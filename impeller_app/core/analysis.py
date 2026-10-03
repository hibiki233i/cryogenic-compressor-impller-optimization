"""Offline descriptive analysis of saved observations and prediction audits.

No refitting, CFD, or changes to the acquisition/CV protocol occur here.
Prediction pairs are checked against schema-v2 observed targets, not merely
accepted because their CSV has a column named ``true_pr``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

from design_variables import geometry_variable_names, load_variable_specs, variable_names
from ..config import AppConfig
from .live_results import LiveResultsReader, _csv, _json, _stamp

TARGETS = {"eff": "Efficiency", "pr": "totalpressureratio", "mf": "MassFlow"}


@dataclass
class AnalysisSnapshot:
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    audits: dict[str, pd.DataFrame] = field(default_factory=dict)
    pairs: dict[str, pd.DataFrame] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    stage: str = "legacy"
    geometry_names: list[str] = field(default_factory=list)


def stable_csv(path: Path) -> pd.DataFrame:
    stamp = _stamp(path)
    frame = _csv(path)
    if stamp != _stamp(path):
        raise ValueError(f"File changed while reading: {path}")
    return frame


def finite_rows(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not set(columns).issubset(frame.columns):
        return frame.iloc[:0].copy()
    result = frame.copy()
    result[columns] = result[columns].apply(pd.to_numeric, errors="coerce")
    return result.loc[np.isfinite(result[columns].to_numpy()).all(axis=1)].copy()


def filter_observations(frame, boundary="all", blade_count=None):
    result = frame
    if boundary != "all" and "is_boundary" in result:
        result = result.loc[result.is_boundary == int(boundary)]
    if blade_count is not None and "nBl" in result:
        result = result.loc[result.nBl == blade_count]
    return result.copy()


def correlations(frame, target, geometry_names):
    rows = []
    for name in geometry_names:
        if name == "nBl":
            continue  # Categorical blade count is analyzed by groups, not Pearson r.
        paired = finite_rows(frame, [name, target])
        if len(paired) < 3 or paired[name].nunique() < 2 or paired[target].nunique() < 2:
            continue
        value = float(paired[name].corr(paired[target]))
        if np.isfinite(value):
            rows.append({"variable": name, "r": value, "n": len(paired)})
    return pd.DataFrame(sorted(rows, key=lambda row: -abs(row["r"])), columns=["variable", "r", "n"])


def blade_groups(frame, target):
    data = finite_rows(frame, ["nBl", target])
    if data.empty:
        return pd.DataFrame(columns=["nBl", "count", "mean", "std", "min", "max"])
    return data.groupby("nBl")[target].agg(["count", "mean", "std", "min", "max"]).reset_index()


def prediction_metrics(pairs: pd.DataFrame, suffix: str) -> dict:
    data = finite_rows(pairs, [f"true_{suffix}", f"pred_{suffix}"])
    if data.empty:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "r2": None}
    truth = data[f"true_{suffix}"].to_numpy(float)
    errors = data[f"pred_{suffix}"].to_numpy(float) - truth
    spread = float(np.sum((truth - truth.mean()) ** 2))
    return {"n": len(data), "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.mean(errors ** 2))), "bias": float(errors.mean()),
            "r2": float(1 - np.sum(errors ** 2) / spread) if len(data) > 1 and spread > 0 else None}


def verified_pairs(audit, reference, names, stage, kind):
    """Match historical truth to known observations; never recompute predictions.

    Older untagged audits remain usable only in an untagged/legacy workspace
    and after numeric truth matching. In a named stage, untagged rows cannot
    be attributed to that stage and stay available only as raw audit records.
    """
    required = names + [f"{prefix}_{suffix}" for suffix in TARGETS for prefix in ("true", "pred")]
    data = finite_rows(audit, required + ["iter"])
    if data.empty or reference.empty:
        return data.iloc[:0].copy()
    tags = data.get("stage_id", pd.Series("legacy", index=data.index)).fillna("legacy").astype(str)
    data = data.loc[tags == stage].copy()
    data = data.loc[(data["iter"] >= 0) & (data["iter"] % 1 == 0)]
    if kind == "query":
        if "status" not in data or "run_id" not in data:
            return data.iloc[:0]
        data = data.loc[data.status.astype(str).str.lower().eq("success")]
        if "result_valid" in data:
            data = data.loc[data.result_valid.astype(str).str.lower().isin(["true", "1", "1.0"])]
        data = data.drop_duplicates("run_id", keep="last")
    elif "test_index" in data:
        data = data.drop_duplicates(["iter", "test_index"], keep="last")
    else:
        return data.iloc[:0]
    observed = {}
    for _, row in reference.iterrows():
        key = tuple(np.round(row[names].to_numpy(float), 10))
        observed.setdefault(key, []).append(row)
    keep = []
    for index, row in data.iterrows():
        key = tuple(np.round(row[names].to_numpy(float), 10))
        for truth in observed.get(key, []):
            if (np.allclose(row[names].to_numpy(float), truth[names].to_numpy(float), rtol=0, atol=1e-10)
                    and all(np.isclose(row[f"true_{suffix}"], truth[column], rtol=1e-8, atol=1e-10)
                            for suffix, column in TARGETS.items())):
                keep.append(index)
                break
    return data.loc[keep].copy()


class AnalysisReader:
    def __init__(self, config: AppConfig):
        self.config = config.resolved()

    def read(self) -> AnalysisSnapshot:
        ws = self.config.workspace
        snapshot = AnalysisSnapshot()
        specs = load_variable_specs(ws.design_variables_json)
        names = variable_names(specs)
        snapshot.geometry_names = geometry_variable_names(specs)
        stage_path = ws.checkpoint_meta_json.parent / "al_stage.json"
        if stage_path.exists():
            stage = _json(stage_path)
            snapshot.stage = str(stage["stage_id"])
        for name, path in (("doe", ws.training_csv), ("pool", ws.pool_checkpoint_csv),
                           ("fixed", ws.project_root / "fixed_test_set.csv")):
            snapshot.sources[name] = str(path)
            try:
                frame, excluded, duplicates, _ = LiveResultsReader(self.config).performance_frame(path)
                # Invalid categorical codes must not become continuous blade sizes.
                frame = frame.loc[frame.nBl.eq(frame.nBl.round())]
                snapshot.frames[name] = frame
                snapshot.counts[name] = (excluded, duplicates)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                snapshot.frames[name] = pd.DataFrame()
                snapshot.issues.append(f"{name}: {exc}")
        for name, path in (("query", ws.al_query_validation_csv), ("fixed", ws.fixed_test_predictions_csv),
                           ("cv", ws.cv_fold_metrics_csv), ("validation", ws.surrogate_metrics_csv)):
            snapshot.sources[name + "_audit"] = str(path)
            try:
                snapshot.audits[name] = stable_csv(path)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                snapshot.audits[name] = pd.DataFrame()
                snapshot.issues.append(f"{name}: {exc}")
        for kind, reference in (("query", snapshot.frames["pool"]), ("fixed", snapshot.frames["fixed"])):
            snapshot.pairs[kind] = verified_pairs(snapshot.audits[kind], reference, names, snapshot.stage, kind)
        return snapshot
