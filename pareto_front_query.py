#!/usr/bin/env python3
"""
Generate a Pareto front from the saved active-learning pool and query geometry
parameters for points on that front.

Main capabilities:
1. Read the final pool checkpoint or training CSV.
2. Extract the feasible true Pareto front and save:
   - pareto_front_points.csv
   - pareto_front.png
3. Select a point on the front by:
   - exact front index
   - fractional position along the front polyline
   - explicit target (efficiency, pressure ratio)
4. For interpolated/target points, optionally use the trained forward neural
   network to solve an inverse-design style search for geometry parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from design_variables import lower_bounds, load_variable_specs, upper_bounds, variable_names

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


ALL_OUTPUT_NAMES = ["Efficiency", "totalpressureratio", "Power", "MassFlow"]
SURROGATE_OUTPUT_NAMES = ["Efficiency", "totalpressureratio", "MassFlow"]
_VARIABLE_SPECS = load_variable_specs()
VAR_NAMES = variable_names(_VARIABLE_SPECS)
L_BOUNDS = lower_bounds(_VARIABLE_SPECS)
U_BOUNDS = upper_bounds(_VARIABLE_SPECS)

MAX_LE_SWEEP_DIFF = 52.0
MIN_RAKE_TE_S_BY_NBL = {
    9: -18.0,
    10: -19.0,
    11: -20.0,
    12: -21.0,
}

DEFAULT_POOL_CSV = "al_training_pool_checkpoint.csv"
DEFAULT_TRAINING_CSV = "Compressor_Training_Data.csv"
DEFAULT_MODEL_PATH = "best_regressor.pth"
DEFAULT_GEOM_WARN_CLF_PATH = "geometry_warning_clf.pkl"
DEFAULT_PARETO_CSV = "pareto_front_points.csv"
DEFAULT_PLOT_PATH = "pareto_front.png"
DEFAULT_SELECTION_JSON = "pareto_selected_point.json"
DEFAULT_ENGINEERING_CSV = "pareto_engineering_ranked.csv"
DEFAULT_ENGINEERING_JSON = "pareto_engineering_report.json"
GEOM_WARN_FEATURE_NAMES = [name for name in VAR_NAMES if name != "P_out"]


def configure_runtime(design_variables_path: str | None = None):
    globals_dict = globals()
    specs = load_variable_specs(design_variables_path) if design_variables_path else load_variable_specs()
    globals_dict["_VARIABLE_SPECS"] = specs
    globals_dict["VAR_NAMES"] = variable_names(specs)
    globals_dict["L_BOUNDS"] = lower_bounds(specs)
    globals_dict["U_BOUNDS"] = upper_bounds(specs)
    globals_dict["GEOM_WARN_FEATURE_NAMES"] = [name for name in globals_dict["VAR_NAMES"] if name != "P_out"]


@dataclass
class SimpleMinMaxScaler:
    data_min_: np.ndarray
    data_max_: np.ndarray

    @classmethod
    def fit(cls, arr: np.ndarray) -> "SimpleMinMaxScaler":
        arr = np.asarray(arr, dtype=float)
        return cls(data_min_=arr.min(axis=0), data_max_=arr.max(axis=0))

    def transform(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        scale = self.data_max_ - self.data_min_
        scale = np.where(scale == 0.0, 1.0, scale)
        return (arr - self.data_min_) / scale

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        scale = self.data_max_ - self.data_min_
        scale = np.where(scale == 0.0, 1.0, scale)
        return arr * scale + self.data_min_


if torch is not None:
    class PerformanceSurrogate(nn.Module):
        def __init__(self, input_dim: int = 14, output_dim: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.12),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.12),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.08),
                nn.Linear(32, output_dim),
            )

        def forward(self, x):
            return self.net(x)


def snap_discrete_vars(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float, copy=True)
    if x.ndim == 1:
        x = x[None, :]
    x[:, 11] = np.clip(np.round(x[:, 11]), 9, 12)
    return x


def geometry_rule_violations(x: np.ndarray) -> np.ndarray:
    x = snap_discrete_vars(x)
    (
        d1s,
        dH,
        beta1hb,
        beta1sb,
        d2,
        b2,
        beta2hb,
        beta2sb,
        Lz,
        t,
        tip_clear,
        n_bl,
        rake_te_s,
        p_out,
    ) = x.T

    g1 = 0.070 - (d2 - d1s)
    g2 = 0.044 - b2
    g3 = t - 0.0035
    g4 = 0.0007 - tip_clear
    g5 = tip_clear - 0.0015
    g6 = (beta1hb - beta1sb) - MAX_LE_SWEEP_DIFF
    g7 = np.abs(beta2hb - beta2sb) - 13.5
    g8 = 0.185 - Lz
    g9 = Lz - 0.215
    g10 = rake_te_s + 15.0
    g11 = -25.0 - rake_te_s

    return np.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])


def overlap_proxy_violation(x: np.ndarray) -> np.ndarray:
    x = snap_discrete_vars(x)
    n_bl = np.clip(np.round(x[:, 11]), 9, 12).astype(int)
    rake_te_s = x[:, 12]
    min_rake_for_overlap = np.array(
        [MIN_RAKE_TE_S_BY_NBL[int(v)] for v in n_bl],
        dtype=float,
    )
    return np.maximum(0.0, min_rake_for_overlap - rake_te_s)


def load_geometry_warning_classifier(path: str):
    if joblib is None or not path or not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def predict_geometry_safe_prob(geom_warn_clf, x_raw: np.ndarray) -> np.ndarray:
    if geom_warn_clf is None:
        return np.ones(len(x_raw))
    x_raw = snap_discrete_vars(x_raw)
    x_geom = x_raw[:, :len(GEOM_WARN_FEATURE_NAMES)]
    p_bad = geom_warn_clf.predict_proba(x_geom)[:, 1]
    return 1.0 - p_bad


def non_dominated_mask_maximize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates_i = np.all(y[i] >= y, axis=1) & np.any(y[i] > y, axis=1)
        dominated_by_i = np.all(y >= y[i], axis=1) & np.any(y > y[i], axis=1)
        if np.any(dominated_by_i):
            keep[i] = False
        else:
            keep[dominates_i] = False
            keep[i] = True
    return keep


def resolve_input_csv(explicit_path: str | None) -> str:
    candidates = [explicit_path, DEFAULT_POOL_CSV, DEFAULT_TRAINING_CSV]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No input CSV found. Tried: {explicit_path!r}, "
        f"{DEFAULT_POOL_CSV!r}, {DEFAULT_TRAINING_CSV!r}"
    )


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = VAR_NAMES + ALL_OUTPUT_NAMES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    if "is_boundary" not in df.columns:
        df["is_boundary"] = 0.0
    df = df[VAR_NAMES + ALL_OUTPUT_NAMES + ["is_boundary"]].copy()
    df["nBl"] = np.clip(np.round(df["nBl"]), 9, 12).astype(int)
    df = df.drop_duplicates(subset=VAR_NAMES, keep="first").reset_index(drop=True)
    return df


def build_front_dataframe(df: pd.DataFrame, geom_warn_clf=None, geom_safe_threshold: float = 0.45) -> pd.DataFrame:
    x_pool = snap_discrete_vars(df[VAR_NAMES].to_numpy(dtype=float))
    y_pool = df[ALL_OUTPUT_NAMES].to_numpy(dtype=float)
    p_geom_safe = predict_geometry_safe_prob(geom_warn_clf, x_pool)

    geom_ok = np.all(geometry_rule_violations(x_pool) <= 0.0, axis=1)
    feas_mask = (
        geom_ok
        & (p_geom_safe >= geom_safe_threshold)
        & (y_pool[:, 0] >= 0.45)
        & (y_pool[:, 1] >= 1.60)
        & (y_pool[:, 1] <= 2.85)
        & (y_pool[:, 3] >= 3.60)
    )

    df_feas = df.loc[feas_mask].copy().reset_index(drop=True)
    if len(df_feas) == 0:
        return df_feas

    y_obj = df_feas[["Efficiency", "totalpressureratio"]].to_numpy(dtype=float)
    nd_mask = non_dominated_mask_maximize(y_obj)
    front = df_feas.loc[nd_mask].copy()
    front["pred_geom_safe_prob"] = predict_geometry_safe_prob(geom_warn_clf, front[VAR_NAMES].to_numpy(dtype=float))
    front = front.sort_values(["Efficiency", "totalpressureratio"], ascending=[True, True]).reset_index(drop=True)
    front.insert(0, "front_index", np.arange(len(front)))
    return front


def save_pareto_plot(front: pd.DataFrame, plot_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(front["Efficiency"], front["totalpressureratio"], "o-", lw=2, ms=5)
    ax.set_xlabel("Efficiency")
    ax.set_ylabel("Pressure Ratio")
    ax.set_title("Pareto Front")
    ax.grid(True, alpha=0.3)
    for _, row in front.iterrows():
        ax.annotate(str(int(row["front_index"])), (row["Efficiency"], row["totalpressureratio"]), fontsize=8, alpha=0.75)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def normalized_distance_to_dataset(x_ref: np.ndarray, x_pool: np.ndarray) -> np.ndarray:
    ranges = np.where((U_BOUNDS - L_BOUNDS) == 0.0, 1.0, (U_BOUNDS - L_BOUNDS))
    x_ref_n = x_ref / ranges
    x_pool_n = x_pool / ranges
    diff = x_ref_n[:, None, :] - x_pool_n[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def front_knee_scores(front: pd.DataFrame) -> np.ndarray:
    xy = front[["Efficiency", "totalpressureratio"]].to_numpy(dtype=float)
    n = len(xy)
    if n <= 2:
        return np.ones(n, dtype=float)
    start = xy[0]
    end = xy[-1]
    line = end - start
    line_norm = float(np.linalg.norm(line))
    if line_norm <= 0.0:
        return np.ones(n, dtype=float)
    rel = xy - start
    cross = np.abs(rel[:, 0] * line[1] - rel[:, 1] * line[0])
    return cross / line_norm


def local_output_stability(front: pd.DataFrame, full_df: pd.DataFrame, k_neighbors: int = 8) -> np.ndarray:
    x_front = front[VAR_NAMES].to_numpy(dtype=float)
    x_pool = full_df[VAR_NAMES].to_numpy(dtype=float)
    y_pool = full_df[["Efficiency", "totalpressureratio", "MassFlow"]].to_numpy(dtype=float)
    dist = normalized_distance_to_dataset(x_front, x_pool)
    scores = np.zeros(len(front), dtype=float)
    for i in range(len(front)):
        nn_idx = np.argsort(dist[i])[: max(2, min(k_neighbors, len(full_df)))]
        y_local = y_pool[nn_idx]
        eff_std = float(np.std(y_local[:, 0]))
        pr_std = float(np.std(y_local[:, 1]))
        mf_std = float(np.std(y_local[:, 2]))
        scores[i] = eff_std / 0.02 + pr_std / 0.05 + mf_std / 0.08
    return scores


def compute_engineering_front_scores(front: pd.DataFrame, full_df: pd.DataFrame, geom_warn_clf=None) -> pd.DataFrame:
    ranked = front.copy()
    x_front = ranked[VAR_NAMES].to_numpy(dtype=float)
    x_pool = full_df[VAR_NAMES].to_numpy(dtype=float)
    y = ranked[["Efficiency", "totalpressureratio", "MassFlow"]].to_numpy(dtype=float)

    geom_v = geometry_rule_violations(x_front)
    geom_margin = -np.max(geom_v, axis=1)
    geom_margin = np.maximum(0.0, geom_margin)
    geom_safe_prob = predict_geometry_safe_prob(geom_warn_clf, x_front)

    all_dist = normalized_distance_to_dataset(x_front, x_pool)
    nearest_success_dist = np.partition(all_dist, 1, axis=1)[:, 1] if len(full_df) > 1 else np.zeros(len(ranked))

    knee_score = front_knee_scores(ranked)
    stability_penalty = local_output_stability(ranked, full_df)

    flow_margin = np.maximum(0.0, y[:, 2] - 3.60)
    eff_margin = np.maximum(0.0, y[:, 0] - 0.60)
    pr_upper_margin = np.maximum(0.0, 2.85 - y[:, 1])
    pr_lower_margin = np.maximum(0.0, y[:, 1] - 1.60)
    pr_margin = np.minimum(pr_upper_margin, pr_lower_margin)

    ranges = {
        "geom_margin": max(float(np.max(geom_margin)), 1e-8),
        "nearest_success_dist": max(float(np.max(nearest_success_dist)), 1e-8),
        "knee_score": max(float(np.max(knee_score)), 1e-8),
        "stability_penalty": max(float(np.max(stability_penalty)), 1e-8),
        "flow_margin": max(float(np.max(flow_margin)), 1e-8),
        "eff_margin": max(float(np.max(eff_margin)), 1e-8),
        "pr_margin": max(float(np.max(pr_margin)), 1e-8),
        "geom_safe_prob": max(float(np.max(geom_safe_prob)), 1e-8),
    }

    engineering_score = (
        0.26 * (geom_margin / ranges["geom_margin"])
        + 0.18 * (geom_safe_prob / ranges["geom_safe_prob"])
        + 0.16 * (flow_margin / ranges["flow_margin"])
        + 0.12 * (eff_margin / ranges["eff_margin"])
        + 0.10 * (pr_margin / ranges["pr_margin"])
        + 0.20 * (knee_score / ranges["knee_score"])
        - 0.10 * (nearest_success_dist / ranges["nearest_success_dist"])
        - 0.06 * (stability_penalty / ranges["stability_penalty"])
    )

    ranked["geom_margin"] = geom_margin
    ranked["pred_geom_safe_prob"] = geom_safe_prob
    ranked["nearest_success_dist"] = nearest_success_dist
    ranked["knee_score"] = knee_score
    ranked["stability_penalty"] = stability_penalty
    ranked["flow_margin"] = flow_margin
    ranked["eff_margin"] = eff_margin
    ranked["pr_margin"] = pr_margin
    ranked["engineering_score"] = engineering_score
    ranked = ranked.sort_values(["engineering_score", "Efficiency", "totalpressureratio"], ascending=[False, False, False]).reset_index(drop=True)
    ranked.insert(0, "engineering_rank", np.arange(1, len(ranked) + 1))
    return ranked


def polyline_fraction_point(front: pd.DataFrame, frac: float) -> dict:
    frac = float(np.clip(frac, 0.0, 1.0))
    xy = front[["Efficiency", "totalpressureratio"]].to_numpy(dtype=float)
    if len(xy) == 1:
        return {
            "target_efficiency": float(xy[0, 0]),
            "target_pressure_ratio": float(xy[0, 1]),
            "segment_start_index": 0,
            "segment_end_index": 0,
            "segment_alpha": 0.0,
        }

    seg = np.diff(xy, axis=0)
    seg_len = np.sqrt((seg ** 2).sum(axis=1))
    total_len = float(seg_len.sum())
    if total_len <= 0.0:
        return {
            "target_efficiency": float(xy[0, 0]),
            "target_pressure_ratio": float(xy[0, 1]),
            "segment_start_index": 0,
            "segment_end_index": 1,
            "segment_alpha": 0.0,
        }

    target_len = frac * total_len
    cumsum = np.cumsum(seg_len)
    seg_idx = int(np.searchsorted(cumsum, target_len, side="left"))
    prev_len = 0.0 if seg_idx == 0 else float(cumsum[seg_idx - 1])
    alpha = 0.0 if seg_len[seg_idx] <= 0.0 else (target_len - prev_len) / float(seg_len[seg_idx])
    point = xy[seg_idx] * (1.0 - alpha) + xy[seg_idx + 1] * alpha
    return {
        "target_efficiency": float(point[0]),
        "target_pressure_ratio": float(point[1]),
        "segment_start_index": seg_idx,
        "segment_end_index": seg_idx + 1,
        "segment_alpha": float(alpha),
    }


def make_scalers(df_source: pd.DataFrame) -> tuple[SimpleMinMaxScaler, SimpleMinMaxScaler]:
    x_scaler = SimpleMinMaxScaler.fit(df_source[VAR_NAMES].to_numpy(dtype=float))
    y_scaler = SimpleMinMaxScaler.fit(df_source[SURROGATE_OUTPUT_NAMES].to_numpy(dtype=float))
    return x_scaler, y_scaler


def load_surrogate_model(model_path: str):
    if torch is None:
        raise RuntimeError("PyTorch is not available in the current environment.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = PerformanceSurrogate(input_dim=14, output_dim=3)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_surrogate(model, x_raw: np.ndarray, scaler_x: SimpleMinMaxScaler, scaler_y: SimpleMinMaxScaler) -> np.ndarray:
    x_raw = snap_discrete_vars(x_raw)
    x_norm = scaler_x.transform(x_raw)
    x_t = torch.tensor(x_norm, dtype=torch.float32)
    with torch.no_grad():
        y_norm = model(x_t).cpu().numpy()
    return scaler_y.inverse_transform(y_norm)


def score_inverse_candidates(
    model,
    y_pred: np.ndarray,
    x_raw: np.ndarray,
    target_eff: float,
    target_pr: float,
    scaler_x: SimpleMinMaxScaler,
    scaler_y: SimpleMinMaxScaler,
    train_x_raw: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    eff = y_pred[:, 0]
    pr = y_pred[:, 1]
    mf = y_pred[:, 2]

    scale_eff = 0.03
    scale_pr = 0.08
    perf_err = ((eff - target_eff) / scale_eff) ** 2 + ((pr - target_pr) / scale_pr) ** 2

    geom_v = geometry_rule_violations(x_raw)
    geom_penalty = 200.0 * np.sum(np.maximum(0.0, geom_v) ** 2, axis=1)
    geom_margin_bonus = -0.15 * np.maximum(0.0, -np.max(geom_v, axis=1))

    mf_penalty = 80.0 * np.maximum(0.0, 3.60 - mf) ** 2
    eff_low_penalty = 80.0 * np.maximum(0.0, 0.60 - eff) ** 2
    pr_low_penalty = 60.0 * np.maximum(0.0, 1.60 - pr) ** 2
    pr_high_penalty = 60.0 * np.maximum(0.0, pr - 2.85) ** 2

    x_norm = scaler_x.transform(x_raw)
    train_x_norm = scaler_x.transform(train_x_raw)
    diff = x_norm[:, None, :] - train_x_norm[None, :, :]
    train_dist = np.sqrt(np.sum(diff ** 2, axis=2)).min(axis=1)
    train_dist_penalty = 0.7 * train_dist

    sigma = (U_BOUNDS - L_BOUNDS) * 0.015
    noise = rng.normal(0.0, sigma, size=(len(x_raw), 6, len(VAR_NAMES)))
    x_pert = np.clip(x_raw[:, None, :] + noise, L_BOUNDS, U_BOUNDS)
    x_pert = snap_discrete_vars(x_pert.reshape(-1, len(VAR_NAMES)))
    y_pert = predict_surrogate(model, x_pert, scaler_x, scaler_y)
    y_pert = y_pert.reshape(len(x_raw), 6, -1)
    robustness_penalty = (
        np.std(y_pert[:, :, 0], axis=1) / 0.015
        + np.std(y_pert[:, :, 1], axis=1) / 0.03
        + np.std(y_pert[:, :, 2], axis=1) / 0.06
    )

    return (
        perf_err
        + geom_penalty
        + mf_penalty
        + eff_low_penalty
        + pr_low_penalty
        + pr_high_penalty
        + train_dist_penalty
        + 0.35 * robustness_penalty
        + geom_margin_bonus
    )


def inverse_design_search(
    model,
    scaler_x: SimpleMinMaxScaler,
    scaler_y: SimpleMinMaxScaler,
    target_eff: float,
    target_pr: float,
    front: pd.DataFrame,
    df_all: pd.DataFrame,
    random_samples: int,
    local_rounds: int,
    local_samples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    front_xy = front[["Efficiency", "totalpressureratio"]].to_numpy(dtype=float)
    front_x = front[VAR_NAMES].to_numpy(dtype=float)
    train_x = df_all[VAR_NAMES].to_numpy(dtype=float)
    dist = np.sqrt((front_xy[:, 0] - target_eff) ** 2 + (front_xy[:, 1] - target_pr) ** 2)
    nearest_order = np.argsort(dist)[: min(5, len(front))]

    x_rand = rng.uniform(L_BOUNDS, U_BOUNDS, size=(random_samples, len(VAR_NAMES)))
    x_rand = snap_discrete_vars(x_rand)
    x_seed = np.vstack([front_x[nearest_order], x_rand])

    y_seed = predict_surrogate(model, x_seed, scaler_x, scaler_y)
    score = score_inverse_candidates(model, y_seed, x_seed, target_eff, target_pr, scaler_x, scaler_y, train_x, rng)

    best_idx = int(np.argmin(score))
    best_x = x_seed[best_idx].copy()
    best_y = y_seed[best_idx].copy()
    best_score = float(score[best_idx])

    sigma = (U_BOUNDS - L_BOUNDS) * 0.15
    for _ in range(local_rounds):
        x_local = best_x + rng.normal(0.0, sigma, size=(local_samples, len(VAR_NAMES)))
        x_local = np.clip(x_local, L_BOUNDS, U_BOUNDS)
        x_local = snap_discrete_vars(x_local)

        x_eval = np.vstack([best_x[None, :], front_x[nearest_order], x_local])
        y_eval = predict_surrogate(model, x_eval, scaler_x, scaler_y)
        score_eval = score_inverse_candidates(model, y_eval, x_eval, target_eff, target_pr, scaler_x, scaler_y, train_x, rng)

        round_best = int(np.argmin(score_eval))
        if float(score_eval[round_best]) < best_score:
            best_score = float(score_eval[round_best])
            best_x = x_eval[round_best].copy()
            best_y = y_eval[round_best].copy()

        sigma *= 0.6

    return {
        "geometry": dict(zip(VAR_NAMES, [float(v) if i != 11 else int(round(v)) for i, v in enumerate(best_x)])),
        "predicted_outputs": {
            "Efficiency": float(best_y[0]),
            "totalpressureratio": float(best_y[1]),
            "MassFlow": float(best_y[2]),
        },
        "engineering_objective_score": best_score,
        "nearest_front_indices_used_as_seeds": [int(i) for i in nearest_order.tolist()],
    }


def exact_front_selection(front_ranked: pd.DataFrame, idx: int) -> dict:
    matched = front_ranked.loc[front_ranked["front_index"] == idx]
    if len(matched) == 0:
        raise IndexError(f"front-index not found: {idx}")
    row = matched.iloc[0]
    return {
        "selection_mode": "front_index",
        "front_index": int(row["front_index"]),
        "engineering_rank": int(row["engineering_rank"]),
        "engineering_score": float(row["engineering_score"]),
        "geometry": {name: (int(row[name]) if name == "nBl" else float(row[name])) for name in VAR_NAMES},
        "true_outputs": {name: float(row[name]) for name in ALL_OUTPUT_NAMES},
    }


def save_selection_json(result: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Pareto front and query geometry parameters from saved active-learning artifacts."
    )
    parser.add_argument("--input-csv", default=None, help="Input CSV. Defaults to pool checkpoint, then training CSV.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to best_regressor.pth")
    parser.add_argument("--geom-warn-clf-path", default=DEFAULT_GEOM_WARN_CLF_PATH, help="Optional geometry warning classifier path.")
    parser.add_argument("--geom-safe-threshold", type=float, default=0.45, help="Minimum predicted geometry-safe probability when building the Pareto front.")
    parser.add_argument("--output-csv", default=DEFAULT_PARETO_CSV, help="Output CSV for Pareto front points.")
    parser.add_argument("--plot-path", default=DEFAULT_PLOT_PATH, help="Output image path for Pareto curve.")
    parser.add_argument("--selection-json", default=DEFAULT_SELECTION_JSON, help="JSON file for the selected query result.")
    parser.add_argument("--engineering-csv", default=DEFAULT_ENGINEERING_CSV, help="Ranked engineering output CSV.")
    parser.add_argument("--engineering-json", default=DEFAULT_ENGINEERING_JSON, help="Engineering report JSON.")
    parser.add_argument("--front-index", type=int, default=None, help="Pick an exact existing Pareto point by index.")
    parser.add_argument("--curve-frac", type=float, default=None, help="Pick a point by fractional arc length on the Pareto curve, in [0, 1].")
    parser.add_argument("--target-eff", type=float, default=None, help="Target efficiency for inverse query.")
    parser.add_argument("--target-pr", type=float, default=None, help="Target pressure ratio for inverse query.")
    parser.add_argument("--random-samples", type=int, default=4000, help="Random samples for inverse search.")
    parser.add_argument("--local-rounds", type=int, default=6, help="Number of local refinement rounds.")
    parser.add_argument("--local-samples", type=int, default=500, help="Samples per local refinement round.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for inverse search.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = resolve_input_csv(args.input_csv)
    df = load_dataset(input_csv)
    geom_warn_clf = load_geometry_warning_classifier(args.geom_warn_clf_path)
    front = build_front_dataframe(df, geom_warn_clf=geom_warn_clf, geom_safe_threshold=args.geom_safe_threshold)

    if len(front) == 0:
        print("No feasible Pareto front could be extracted from the input data.", file=sys.stderr)
        return 2

    front_ranked = compute_engineering_front_scores(front, df, geom_warn_clf=geom_warn_clf)

    front.to_csv(args.output_csv, index=False)
    front_ranked.to_csv(args.engineering_csv, index=False)
    save_pareto_plot(front, args.plot_path)

    engineering_report = {
        "input_csv": input_csv,
        "front_size": int(len(front)),
        "geometry_warning_classifier_loaded": bool(geom_warn_clf is not None),
        "geometry_safe_threshold": float(args.geom_safe_threshold),
        "recommended_front_index": int(front_ranked.iloc[0]["front_index"]),
        "recommended_engineering_rank": int(front_ranked.iloc[0]["engineering_rank"]),
        "recommended_engineering_score": float(front_ranked.iloc[0]["engineering_score"]),
        "top_recommendations": [
            {
                "engineering_rank": int(row["engineering_rank"]),
                "front_index": int(row["front_index"]),
                "engineering_score": float(row["engineering_score"]),
                "Efficiency": float(row["Efficiency"]),
                "totalpressureratio": float(row["totalpressureratio"]),
                "MassFlow": float(row["MassFlow"]),
            }
            for _, row in front_ranked.head(5).iterrows()
        ],
    }
    save_selection_json(engineering_report, args.engineering_json)

    print(f"Pareto front extracted from: {input_csv}")
    print(f"Pareto front size: {len(front)}")
    print(f"Saved Pareto points CSV: {args.output_csv}")
    print(f"Saved engineering-ranked CSV: {args.engineering_csv}")
    print(f"Saved engineering report JSON: {args.engineering_json}")
    print(f"Saved Pareto plot: {args.plot_path}")
    print(
        "Engineering recommendation: "
        f"front_index={engineering_report['recommended_front_index']}, "
        f"score={engineering_report['recommended_engineering_score']:.4f}"
    )

    selection_result = None
    if args.front_index is not None:
        selection_result = exact_front_selection(front_ranked, args.front_index)

    elif args.curve_frac is not None:
        target = polyline_fraction_point(front, args.curve_frac)
        selection_result = {
            "selection_mode": "curve_fraction",
            "curve_fraction": float(args.curve_frac),
            **target,
        }

    elif args.target_eff is not None or args.target_pr is not None:
        if args.target_eff is None or args.target_pr is None:
            raise ValueError("--target-eff and --target-pr must be provided together.")
        selection_result = {
            "selection_mode": "target_point",
            "target_efficiency": float(args.target_eff),
            "target_pressure_ratio": float(args.target_pr),
        }

    if selection_result is None:
        return 0

    if selection_result["selection_mode"] == "front_index":
        save_selection_json(selection_result, args.selection_json)
        print(json.dumps(selection_result, ensure_ascii=False, indent=2))
        print(f"Saved selection JSON: {args.selection_json}")
        return 0

    if torch is None:
        raise RuntimeError(
            "PyTorch is not available, so the script can compute the Pareto curve but cannot run the inverse NN query."
        )

    model = load_surrogate_model(args.model_path)
    scaler_x, scaler_y = make_scalers(df)
    inverse_result = inverse_design_search(
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        target_eff=float(selection_result["target_efficiency"]),
        target_pr=float(selection_result["target_pressure_ratio"]),
        front=front,
        df_all=df,
        random_samples=args.random_samples,
        local_rounds=args.local_rounds,
        local_samples=args.local_samples,
        seed=args.seed,
    )

    merged = {**selection_result, **inverse_result}
    save_selection_json(merged, args.selection_json)
    print(json.dumps(merged, ensure_ascii=False, indent=2))
    print(f"Saved selection JSON: {args.selection_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
