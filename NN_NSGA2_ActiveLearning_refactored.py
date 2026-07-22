import os
import json
import argparse
import subprocess
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from cfx_runner import run_cfx_pipeline
warnings.filterwarnings("ignore", category=UserWarning)
import joblib
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from design_variables import (
    PERFORMANCE_DATA_SCHEMA_VERSION,
    TOTAL_PRESSURE_DEFINITION,
    is_current_performance_data,
    lower_bounds,
    load_variable_specs,
    require_current_performance_data,
    upper_bounds,
    variable_names,
    write_performance_data_metadata,
)
from failure_records import (
    classify_cfx_failure_stage,
    classify_geometry_failure_stage,
    load_structured_failure_status,
    load_failure_training_sets,
    record_run_outcome,
)

CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0


def _env_or_default(name: str, default):
    value = os.environ.get(name)
    return value if value else default


# =============================================================================
# 全局配置
# =============================================================================
VAR_NAMES = [
    'd1s', 'dH', 'beta1hb', 'beta1sb', 'd2', 'b2', 'beta2hb', 'beta2sb',
    'Lz', 't', 'TipClear', 'nBl', 'rake_te_s', 'P_out'
]

# 这里统一压比字段，只允许一个
TARGET_PR_NAME = "totalpressureratio"
ALL_OUTPUT_NAMES = ['Efficiency', TARGET_PR_NAME, 'Power', 'MassFlow']
SURROGATE_OUTPUT_NAMES = ['Efficiency', TARGET_PR_NAME, 'MassFlow']
SURROGATE_OUTPUT_IDX = [0, 1, 3]  
TRUE_HV_REF_EFF = 0.6
TRUE_HV_REF_PR  = 1.0
DESIGN_VARIABLES_PATH = _env_or_default("IMPELLER_DESIGN_VARIABLES_PATH", "design_variables.json")
_VARIABLE_SPECS = load_variable_specs(DESIGN_VARIABLES_PATH)
VAR_NAMES = variable_names(_VARIABLE_SPECS)
L_BOUNDS = lower_bounds(_VARIABLE_SPECS)
U_BOUNDS = upper_bounds(_VARIABLE_SPECS)
P_OUT_NAME = "P_out"
P_OUT_IDX = VAR_NAMES.index(P_OUT_NAME)
GEOMETRY_VAR_NAMES = [name for name in VAR_NAMES if name != P_OUT_NAME]
GEOMETRY_VAR_IDX = np.array([VAR_NAMES.index(name) for name in GEOMETRY_VAR_NAMES], dtype=int)

# nBl remains an input of the performance surrogate and the explicit overlap
# rule, but it is deliberately excluded from learned failure classifiers. P_out
# is also excluded here because this workflow pins every optimization query to
# one operating point; its tiny DOE/AL batch offset would otherwise leak the
# run phase into the failure label instead of describing physical feasibility.
FAILURE_CLASSIFIER_EXCLUDED_FEATURES = frozenset({"nBl", "P_out"})
FEASIBILITY_FEATURE_NAMES = [
    name for name in VAR_NAMES
    if name not in FAILURE_CLASSIFIER_EXCLUDED_FEATURES
]
FEASIBILITY_VAR_IDX = np.array(
    [VAR_NAMES.index(name) for name in FEASIBILITY_FEATURE_NAMES], dtype=int
)
GEOM_WARN_FEATURE_NAMES = [
    name for name in GEOMETRY_VAR_NAMES
    if name not in FAILURE_CLASSIFIER_EXCLUDED_FEATURES
]
GEOM_WARN_VAR_IDX = np.array(
    [VAR_NAMES.index(name) for name in GEOM_WARN_FEATURE_NAMES], dtype=int
)

MIN_VALID_FLOW_G_S = float(_env_or_default("IMPELLER_MIN_VALID_FLOW_G_S", 0.1))
MIN_DISCARD_FLOW_G_S = float(_env_or_default("IMPELLER_MIN_DISCARD_FLOW_G_S", 0.0001))
BOUNDARY_FLOW_G_S = float(_env_or_default("IMPELLER_BOUNDARY_FLOW_G_S", 3.60))
MIN_EFFICIENCY = float(_env_or_default("IMPELLER_MIN_EFFICIENCY", 0.60))
MIN_POWER = float(_env_or_default("IMPELLER_MIN_POWER", 60.0))
OPTIMIZATION_P_OUT = float(_env_or_default("IMPELLER_OPTIMIZATION_P_OUT", 12.0))
OPERATING_POINT_P_OUT_TOLERANCE = float(
    _env_or_default("IMPELLER_OPERATING_POINT_P_OUT_TOLERANCE", 0.25)
)

# 默认工程几何阈值，可由 GUI/配置文件覆盖，用于当前叶轮族的可行域筛选。
MIN_D2_D1S_GAP = float(_env_or_default("IMPELLER_MIN_D2_D1S_GAP", 0.070))
MAX_LE_SWEEP_DIFF = float(_env_or_default("IMPELLER_MAX_LE_SWEEP_DIFF", 52.0))
MAX_EXIT_ANGLE_DIFF = float(_env_or_default("IMPELLER_MAX_EXIT_ANGLE_DIFF", 13.5))

# 相邻叶片 overlap 的代理约束：
# nBl 越少，节距角 360/nBl 越大，同样的后掠越容易造成 overlap 不足，
# 因此对 rake_te_s 的允许下限做成随 nBl 自适应变化。
MIN_RAKE_TE_S_BY_NBL = {
    9: -18.0,
    10: -19.0,
    11: -20.0,
    12: -21.0,
}


def min_rake_te_s_for_nbl(n_bl: int) -> float:
    if n_bl in MIN_RAKE_TE_S_BY_NBL:
        return MIN_RAKE_TE_S_BY_NBL[n_bl]
    nearest = min(MIN_RAKE_TE_S_BY_NBL, key=lambda value: abs(value - n_bl))
    return MIN_RAKE_TE_S_BY_NBL[nearest]

# overlap 先作为软惩罚而不是硬约束处理，避免直接把大量已算出的 DOE/训练点全部判死。
OVERLAP_PENALTY_COEFF = 0.035
OVERLAP_EHVI_DECAY_DEG = 3.0
GEOM_WARN_PENALTY_COEFF = 0.20
GEOM_WARN_RUNS_DIR_FALLBACK = "ActiveLearning_Runs"

PS_SCRIPT_PATH  = _env_or_default("IMPELLER_PS_SCRIPT_PATH", r"F:\optimazition\Run-GeometryMeshing.ps1")
TURBOGRID_TEMPLATE = _env_or_default(
    "IMPELLER_TURBOGRID_TEMPLATE", r"F:\optimazition\Templates\new_base.tst"
)
AL_WORKING_BASE = _env_or_default("IMPELLER_AL_WORKING_BASE", r"F:\optimazition\ActiveLearning_Runs")
TRAINING_CSV    = _env_or_default("IMPELLER_TRAINING_CSV", "Compressor_Training_Data.csv")
SCALER_X_PATH = _env_or_default("IMPELLER_SCALER_X_PATH", "scaler_X.pkl")
SCALER_Y_PATH = _env_or_default("IMPELLER_SCALER_Y_PATH", "scaler_Y.pkl")
BEST_REG_PATH   = _env_or_default("IMPELLER_BEST_REG_PATH", "best_regressor.pth")
GEOM_FEAS_CLF_PATH = _env_or_default(
    "IMPELLER_GEOM_FEAS_CLF_PATH", "geometry_feasibility_clf.pkl"
)
HV_CSV_PATH     = _env_or_default("IMPELLER_HV_CSV_PATH", "hv_history.csv")
HV_PLOT_PATH    = _env_or_default("IMPELLER_HV_PLOT_PATH", "hv_convergence.png")
FAILED_POINTS_PATH = _env_or_default("IMPELLER_FAILED_POINTS_PATH", "failed_points.npy")  # 保留失败样本池
FAILURE_RECORDS_CSV = _env_or_default(
    "IMPELLER_FAILURE_RECORDS_CSV", "failure_records.csv"
)
SURROGATE_METRICS_CSV = _env_or_default(
    "IMPELLER_SURROGATE_METRICS_CSV", "surrogate_validation_history.csv"
)
CV_FOLD_METRICS_CSV = _env_or_default(
    "IMPELLER_CV_FOLD_METRICS_CSV", "surrogate_cv_fold_history.csv"
)
FIXED_TEST_PREDICTIONS_CSV = _env_or_default(
    "IMPELLER_FIXED_TEST_PREDICTIONS_CSV", "fixed_test_predictions_history.csv"
)
AL_QUERY_VALIDATION_CSV = _env_or_default(
    "IMPELLER_AL_QUERY_VALIDATION_CSV", "al_query_validation.csv"
)
TEST_SPLIT_PATH = _env_or_default("IMPELLER_TEST_SPLIT_PATH", "fixed_test_set.npz")  # 固定测试集，避免随机划分引入评估波动
TEST_SET_CSV = _env_or_default("IMPELLER_TEST_SET_CSV", "fixed_test_set.csv")
POOL_CHECKPOINT_CSV = _env_or_default("IMPELLER_POOL_CHECKPOINT_CSV", "al_training_pool_checkpoint.csv")
CHECKPOINT_META_PATH = _env_or_default("IMPELLER_CHECKPOINT_META_PATH", "al_checkpoint_meta.json")
GEOMETRY_SUMMARY_PATH = _env_or_default("IMPELLER_GEOMETRY_SUMMARY_PATH", "geometry_summary.json")
ENABLE_INTERACTIVE_PLOT = _env_or_default("IMPELLER_ENABLE_PLOT", "0").lower() in {"1", "true", "yes"}

HV_HISTORY_COLUMNS = [
    "iter", "hv_policy_version", "n_samples", "train_samples_before_cfd",
    "true_hv", "surrogate_hv",
    "mse_eff", "mse_pr", "mse_mf", "rmse_eff", "rmse_pr", "rmse_mf",
    "mae_eff", "mae_pr", "mae_mf", "r2_eff", "r2_pr", "r2_mf",
    "cv_rmse_eff_mean", "cv_rmse_pr_mean", "cv_rmse_mf_mean",
    "cv_r2_eff_mean", "cv_r2_pr_mean", "cv_r2_mf_mean",
]
HV_POLICY_VERSION = 2  # v2: verified CFD front is not vetoed by soft overlap proxy


def read_optional_csv(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV that may exist before its first data row is written."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def write_hv_history(path: str, records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records, columns=HV_HISTORY_COLUMNS)
    frame.to_csv(path, index=False)
    return frame


def recover_query_counters(
    query_csv_path: str,
    total_attempts: int = 0,
    total_success: int = 0,
) -> tuple[int, int]:
    """Prevent run-id reuse when query CSV is newer than checkpoint metadata."""
    history = read_optional_csv(query_csv_path)
    if history.empty or "run_id" not in history.columns:
        return int(total_attempts), int(total_success)

    attempt_ids = pd.to_numeric(
        history["run_id"].astype(str).str.extract(r"_A(\d+)$", expand=False),
        errors="coerce",
    ).dropna()
    if len(attempt_ids) > 0:
        total_attempts = max(int(total_attempts), int(attempt_ids.max()))
    if "status" in history.columns:
        logged_success = int(
            history.loc[
                history["status"].astype(str).str.lower() == "success",
                "run_id",
            ].nunique()
        )
        total_success = max(int(total_success), logged_success)
    return int(total_attempts), int(total_success)


MAX_AL_ITERS    = 80
MC_SAMPLES      = 30
N_CANDIDATES    = 5000

BOUNDARY_WEIGHT = 2.0
DEVICE = "cpu"


# =============================================================================
# 配置数据类
# =============================================================================
@dataclass
class ALConfig:
    max_al_iters: int = MAX_AL_ITERS
    mc_samples: int = MC_SAMPLES
    n_candidates: int = N_CANDIDATES

    feasible_prob_threshold_opt: float = 0.65
    feasible_prob_threshold_pick: float = 0.55
    geom_safe_prob_threshold_opt: float = 0.55
    geom_safe_prob_threshold_pick: float = 0.45

    train_distance_soft_limit: float = 0.16

    pop_size: int = 120
    n_gen: int = 120

    n_eval_candidates_per_iter: int = 4
    cv_folds: int = 5
    cv_max_epochs: int = 1200
    cv_patience: int = 40
    density_k_neighbors: int = 10
    sample_weight_min: float = 0.25
    sample_weight_max: float = 4.0

    # --- 新增：局部采样比例 ---
    local_sample_ratio: float = 0.4

    # --- 新增：多样性约束最小归一化距离 ---
    diversity_min_dist: float = 0.08


CFG = ALConfig()


def configure_runtime(**overrides):
    """Allow GUI/services to override legacy script globals without editing the script."""
    globals_dict = globals()
    for key, value in overrides.items():
        if key in globals_dict and value is not None:
            globals_dict[key] = value
    if globals_dict.get("DESIGN_VARIABLES_PATH"):
        specs = load_variable_specs(globals_dict["DESIGN_VARIABLES_PATH"])
        globals_dict["VAR_NAMES"] = variable_names(specs)
        globals_dict["L_BOUNDS"] = lower_bounds(specs)
        globals_dict["U_BOUNDS"] = upper_bounds(specs)
        globals_dict["P_OUT_IDX"] = globals_dict["VAR_NAMES"].index(P_OUT_NAME)
        globals_dict["GEOMETRY_VAR_NAMES"] = [
            name for name in globals_dict["VAR_NAMES"] if name != P_OUT_NAME
        ]
        globals_dict["GEOMETRY_VAR_IDX"] = np.array(
            [globals_dict["VAR_NAMES"].index(name) for name in globals_dict["GEOMETRY_VAR_NAMES"]],
            dtype=int,
        )
        globals_dict["FEASIBILITY_FEATURE_NAMES"] = [
            name for name in globals_dict["VAR_NAMES"]
            if name not in FAILURE_CLASSIFIER_EXCLUDED_FEATURES
        ]
        globals_dict["FEASIBILITY_VAR_IDX"] = np.array(
            [
                globals_dict["VAR_NAMES"].index(name)
                for name in globals_dict["FEASIBILITY_FEATURE_NAMES"]
            ],
            dtype=int,
        )
        globals_dict["GEOM_WARN_FEATURE_NAMES"] = [
            name for name in globals_dict["GEOMETRY_VAR_NAMES"]
            if name not in FAILURE_CLASSIFIER_EXCLUDED_FEATURES
        ]
        globals_dict["GEOM_WARN_VAR_IDX"] = np.array(
            [
                globals_dict["VAR_NAMES"].index(name)
                for name in globals_dict["GEOM_WARN_FEATURE_NAMES"]
            ],
            dtype=int,
        )

    p_out_lower = float(globals_dict["L_BOUNDS"][globals_dict["P_OUT_IDX"]])
    p_out_upper = float(globals_dict["U_BOUNDS"][globals_dict["P_OUT_IDX"]])
    fixed_p_out = float(globals_dict["OPTIMIZATION_P_OUT"])
    if not p_out_lower <= fixed_p_out <= p_out_upper:
        raise ValueError(
            f"OPTIMIZATION_P_OUT={fixed_p_out} Pa must be within "
            f"the surrogate training bounds [{p_out_lower}, {p_out_upper}] Pa."
        )


def parse_runtime_args():
    parser = argparse.ArgumentParser(
        description="Surrogate-assisted compressor active learning."
    )
    parser.add_argument(
        "--max-al-iters",
        type=int,
        default=None,
        help="Absolute total active-learning iterations to run to.",
    )
    parser.add_argument(
        "--additional-iters",
        type=int,
        default=0,
        help="Continue for N extra iterations beyond the completed checkpoint.",
    )
    parser.add_argument(
        "--nsga2-only",
        action="store_true",
        help="Train the surrogate from the current LHS/DOE data and run NSGA-II without EHVI/CFD active-learning updates.",
    )
    parser.add_argument(
        "--nsga2-output-csv",
        default="nsga2_surrogate_pareto.csv",
        help="Output CSV for the NSGA-II surrogate Pareto front.",
    )
    parser.add_argument(
        "--nsga2-use-pool-checkpoint",
        action="store_true",
        help="Use al_training_pool_checkpoint.csv as the NSGA-II training source when it exists. Defaults to pure TRAINING_CSV/LHS data.",
    )
    return parser.parse_args()


#断点需跑的辅助函数
def get_resume_iter(
    checkpoint_meta_path=None,
    hv_csv_path=None
):
    checkpoint_meta_path = CHECKPOINT_META_PATH if checkpoint_meta_path is None else checkpoint_meta_path
    hv_csv_path = HV_CSV_PATH if hv_csv_path is None else hv_csv_path
    if os.path.exists(checkpoint_meta_path):
        try:
            with open(checkpoint_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if (
                int(meta.get("performance_data_schema_version", 0))
                != PERFORMANCE_DATA_SCHEMA_VERSION
                or meta.get("total_pressure_definition")
                != TOTAL_PRESSURE_DEFINITION
            ):
                print("[断点续跑] 忽略旧总压定义对应的 checkpoint 元信息。")
                return 0
            completed_iters = int(meta.get("completed_iters", 0) or 0)
            in_progress_iter = meta.get("in_progress_iter")
            if in_progress_iter is not None:
                in_progress_iter = int(in_progress_iter)
                # If metadata is inconsistent, prefer restarting the interrupted
                # 1-based iteration instead of dropping back to an older checkpoint.
                completed_iters = max(completed_iters, in_progress_iter - 1)
            return completed_iters
        except Exception as e:
            print(f"[断点续跑] 读取 checkpoint 元信息失败，将回退到 HV 历史: {e}")

    if os.path.exists(hv_csv_path):
        hv_df = read_optional_csv(hv_csv_path, HV_HISTORY_COLUMNS)
        if len(hv_df) > 0 and 'iter' in hv_df.columns:
            return int(hv_df['iter'].max())
    return 0


def load_pool_checkpoint(pool_csv=None):
    pool_csv = POOL_CHECKPOINT_CSV if pool_csv is None else pool_csv
    if not os.path.exists(pool_csv):
        return None
    if os.path.getsize(pool_csv) == 0:
        print(f"[断点续跑] 忽略空训练池 checkpoint: {pool_csv}")
        return None
    if not is_current_performance_data(pool_csv):
        print(
            f"[断点续跑] 忽略旧版训练池 checkpoint（总压定义不匹配）: {pool_csv}"
        )
        return None

    df_pool = pd.read_csv(pool_csv)
    expected_cols = VAR_NAMES + ALL_OUTPUT_NAMES + ['is_boundary']
    missing_cols = [col for col in expected_cols if col not in df_pool.columns]
    if missing_cols:
        raise ValueError(f"训练池 checkpoint 缺少必要列: {missing_cols}")

    df_pool = df_pool[expected_cols].copy()
    nbl_idx = VAR_NAMES.index("nBl")
    df_pool['nBl'] = np.clip(np.round(df_pool['nBl']), L_BOUNDS[nbl_idx], U_BOUNDS[nbl_idx]).astype(int)
    df_pool = df_pool.drop_duplicates(subset=VAR_NAMES, keep='first').reset_index(drop=True)
    return df_pool


def save_checkpoint(
    al_iter: int,
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    W_pool: np.ndarray,
    X_test_fixed: np.ndarray,
    Y_test_fixed: np.ndarray,
    W_test_fixed: np.ndarray,
    failed_points: list,
    hv_history: list,
    total_attempts: int,
    total_success: int,
    completed_iters: int = None,
    in_progress_iter: int = None,
    pool_csv=None,
    failed_points_path=None,
    hv_csv_path=None,
    checkpoint_meta_path=None
):
    pool_csv = POOL_CHECKPOINT_CSV if pool_csv is None else pool_csv
    failed_points_path = FAILED_POINTS_PATH if failed_points_path is None else failed_points_path
    hv_csv_path = HV_CSV_PATH if hv_csv_path is None else hv_csv_path
    checkpoint_meta_path = CHECKPOINT_META_PATH if checkpoint_meta_path is None else checkpoint_meta_path
    df_pool = pd.DataFrame(X_pool, columns=VAR_NAMES)
    for j, col in enumerate(ALL_OUTPUT_NAMES):
        df_pool[col] = Y_pool[:, j]
    df_pool['is_boundary'] = W_pool
    df_pool.to_csv(pool_csv, index=False)
    write_performance_data_metadata(pool_csv)

    np.save(failed_points_path, np.array(failed_points, dtype=float))
    write_hv_history(hv_csv_path, hv_history)

    if completed_iters is None:
        completed_iters = al_iter + 1

    meta = {
        "completed_iters": int(completed_iters),
        "in_progress_iter": None if in_progress_iter is None else int(in_progress_iter),
        "pool_samples": int(len(X_pool)),
        "test_samples": int(len(X_test_fixed)),
        "failed_points": int(len(failed_points)),
        "total_attempts": int(total_attempts),
        "total_success": int(total_success),
        "pool_checkpoint_csv": pool_csv,
        "fixed_test_set_csv": TEST_SET_CSV,
        "surrogate_metrics_csv": SURROGATE_METRICS_CSV,
        "cv_fold_metrics_csv": CV_FOLD_METRICS_CSV,
        "fixed_test_predictions_csv": FIXED_TEST_PREDICTIONS_CSV,
        "al_query_validation_csv": AL_QUERY_VALIDATION_CSV,
        "failure_records_csv": FAILURE_RECORDS_CSV,
        "hv_policy_version": HV_POLICY_VERSION,
        "performance_data_schema_version": PERFORMANCE_DATA_SCHEMA_VERSION,
        "total_pressure_definition": TOTAL_PRESSURE_DEFINITION,
    }
    with open(checkpoint_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
# =============================================================================
# 基础工具
# =============================================================================
def snap_discrete_vars(X: np.ndarray) -> np.ndarray:
    """
    处理离散变量：nBl 必须是整数
    """
    X = np.array(X, dtype=float, copy=True)
    if X.ndim == 1:
        X = X[None, :]
    nbl_idx = VAR_NAMES.index("nBl")
    X[:, nbl_idx] = np.clip(np.round(X[:, nbl_idx]), L_BOUNDS[nbl_idx], U_BOUNDS[nbl_idx])
    return X


def expand_geometry_decisions(
    X_geometry: np.ndarray,
    fixed_p_out: float | None = None,
) -> np.ndarray:
    """Expand 13 geometry decisions to the 14-input conditional surrogate vector."""
    X_geometry = np.asarray(X_geometry, dtype=float)
    if X_geometry.ndim == 1:
        X_geometry = X_geometry[None, :]
    if X_geometry.shape[1] == len(VAR_NAMES):
        X_full = np.array(X_geometry, dtype=float, copy=True)
    elif X_geometry.shape[1] == len(GEOMETRY_VAR_NAMES):
        X_full = np.empty((len(X_geometry), len(VAR_NAMES)), dtype=float)
        X_full[:, GEOMETRY_VAR_IDX] = X_geometry
    else:
        raise ValueError(
            f"Expected {len(GEOMETRY_VAR_NAMES)} geometry values or "
            f"{len(VAR_NAMES)} full surrogate inputs, got {X_geometry.shape[1]}."
        )
    active_p_out = OPTIMIZATION_P_OUT if fixed_p_out is None else fixed_p_out
    X_full[:, P_OUT_IDX] = float(active_p_out)
    return snap_discrete_vars(X_full)


def pin_optimization_operating_point(
    X: np.ndarray,
    fixed_p_out: float | None = None,
) -> np.ndarray:
    """Keep P_out as a surrogate context input, never as an optimization decision."""
    return expand_geometry_decisions(X, fixed_p_out=fixed_p_out)


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / (rng + 1e-8)


def calc_distance_to_set(X_norm: np.ndarray, ref_norm: np.ndarray) -> np.ndarray:
    return cdist(X_norm, ref_norm).min(axis=1)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert_csv_records(
    csv_path: str,
    records: list[dict] | dict,
    key_columns: list[str],
) -> None:
    """Crash-tolerant CSV upsert used by validation and CFD query logs."""
    if isinstance(records, dict):
        records = [records]
    if not records:
        return

    new_df = pd.DataFrame(records)
    missing_keys = [column for column in key_columns if column not in new_df.columns]
    if missing_keys:
        raise ValueError(f"CSV upsert records missing key columns: {missing_keys}")

    if os.path.exists(csv_path):
        existing = read_optional_csv(csv_path)
    else:
        existing = pd.DataFrame()

    merged = pd.concat([existing, new_df], ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=key_columns, keep="last")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    tmp_path = f"{csv_path}.tmp"
    merged.to_csv(tmp_path, index=False)
    os.replace(tmp_path, csv_path)
    write_performance_data_metadata(csv_path)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict:
    """Return per-output and macro MSE/RMSE/MAE/R² metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Metric arrays must have the same shape, got {y_true.shape} and {y_pred.shape}."
        )
    if y_true.ndim != 2 or y_true.shape[1] != len(SURROGATE_OUTPUT_NAMES):
        raise ValueError(
            f"Expected metric arrays with shape (N, {len(SURROGATE_OUTPUT_NAMES)})."
        )

    suffixes = ["eff", "pr", "mf"]
    result = {}
    per_metric = {"mse": [], "rmse": [], "mae": [], "r2": []}
    for output_idx, suffix in enumerate(suffixes):
        true_col = y_true[:, output_idx]
        pred_col = y_pred[:, output_idx]
        mse = float(mean_squared_error(true_col, pred_col))
        values = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(true_col, pred_col)),
            "r2": float(r2_score(true_col, pred_col)) if len(true_col) >= 2 else np.nan,
        }
        for metric_name, value in values.items():
            result[f"{prefix}{metric_name}_{suffix}"] = value
            per_metric[metric_name].append(value)

    for metric_name, values in per_metric.items():
        result[f"{prefix}{metric_name}_macro"] = float(np.nanmean(values))
    return result


def compute_density_nbl_sample_weights(
    X_norm: np.ndarray,
    nbl_values: np.ndarray,
    k_neighbors: int = 10,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
) -> tuple[np.ndarray, dict]:
    """
    Combine local inverse-density weighting with inverse-frequency nBl weighting.

    Sparse points receive larger k-nearest-neighbour distance weights; each nBl
    class receives a tempered inverse-frequency weight. Square-root tempering,
    clipping and mean-one normalization prevent rare classes from destabilizing
    the optimizer.
    """
    X_norm = np.asarray(X_norm, dtype=float)
    nbl_values = np.rint(np.asarray(nbl_values, dtype=float)).astype(int)
    n_samples = len(X_norm)
    if n_samples != len(nbl_values):
        raise ValueError("X_norm and nbl_values must contain the same sample count.")
    if n_samples == 0:
        return np.empty(0, dtype=float), {
            "weight_min": np.nan,
            "weight_max": np.nan,
            "weight_mean": np.nan,
            "effective_sample_size": 0.0,
            "density_k": 0,
            "nbl_weight_means": "{}",
        }

    if n_samples == 1:
        density_weight = np.ones(1)
        density_k = 0
    else:
        density_k = min(max(1, int(k_neighbors)), n_samples - 1)
        distances = cdist(X_norm, X_norm)
        np.fill_diagonal(distances, np.inf)
        nearest = np.partition(distances, density_k - 1, axis=1)[:, :density_k]
        mean_neighbor_distance = nearest.mean(axis=1)
        reference_distance = max(float(np.median(mean_neighbor_distance)), 1e-12)
        density_weight = np.sqrt(mean_neighbor_distance / reference_distance)

    unique_nbl, nbl_counts = np.unique(nbl_values, return_counts=True)
    n_classes = max(len(unique_nbl), 1)
    nbl_weight_map = {
        int(nbl): np.sqrt(n_samples / (n_classes * int(count)))
        for nbl, count in zip(unique_nbl, nbl_counts)
    }
    nbl_weight = np.array([nbl_weight_map[int(nbl)] for nbl in nbl_values])

    weights = density_weight * nbl_weight
    weights /= max(float(np.mean(weights)), 1e-12)
    weights = np.clip(weights, float(min_weight), float(max_weight))
    weights /= max(float(np.mean(weights)), 1e-12)

    effective_n = float(weights.sum() ** 2 / max(np.square(weights).sum(), 1e-12))
    nbl_weight_means = {
        str(int(nbl)): float(weights[nbl_values == nbl].mean())
        for nbl in unique_nbl
    }
    diagnostics = {
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "effective_sample_size": effective_n,
        "density_k": int(density_k),
        "nbl_weight_means": json.dumps(nbl_weight_means, ensure_ascii=False),
    }
    return weights, diagnostics


def infer_boundary_from_outputs(y: np.ndarray) -> float:
    eff, pr, power, mf = y
    return float(
        (mf < BOUNDARY_FLOW_G_S) or
        (eff < MIN_EFFICIENCY) or
        (power < MIN_POWER)
    )


def geometry_rule_violations(X: np.ndarray) -> np.ndarray:
    """
    显式几何规则约束。
    返回 G_rule, shape = (N, k), 每列 <= 0 表示满足约束。
    这些规则你需要按工程经验继续细化。
    """
    X = snap_discrete_vars(X)
    d1s, dH, beta1hb, beta1sb, d2, b2, beta2hb, beta2sb, Lz, t, TipClear, nBl, rake_te_s, P_out = X.T
    lower = {name: L_BOUNDS[idx] for idx, name in enumerate(VAR_NAMES)}
    upper = {name: U_BOUNDS[idx] for idx, name in enumerate(VAR_NAMES)}

    g1 = MIN_D2_D1S_GAP - (d2 - d1s)
    g2 = lower["b2"] - b2
    g3 = t - upper["t"]
    g4 = lower["TipClear"] - TipClear
    g5 = TipClear - upper["TipClear"]
    g6 = (beta1hb - beta1sb) - MAX_LE_SWEEP_DIFF
    g7 = np.abs(beta2hb - beta2sb) - MAX_EXIT_ANGLE_DIFF
    g8 = lower["Lz"] - Lz
    g9 = Lz - upper["Lz"]
    g10 = rake_te_s - upper["rake_te_s"]
    g11 = lower["rake_te_s"] - rake_te_s

    return np.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])


def overlap_proxy_violation(X: np.ndarray) -> np.ndarray:
    """
    overlap 风险代理量，>0 表示后掠过大导致相邻叶片 overlap 可能偏低。
    目前只作为软惩罚使用，不直接判为硬不可行。
    """
    X = snap_discrete_vars(X)
    nbl_idx = VAR_NAMES.index("nBl")
    nBl = np.clip(np.round(X[:, nbl_idx]), L_BOUNDS[nbl_idx], U_BOUNDS[nbl_idx]).astype(int)
    rake_te_s = X[:, 12]
    min_rake_for_overlap = np.array(
        [min_rake_te_s_for_nbl(int(n_bl)) for n_bl in nBl],
        dtype=float
    )
    return np.maximum(0.0, min_rake_for_overlap - rake_te_s)


def geometry_safe_mask(
    X: np.ndarray,
    geom_warn_clf=None,
    geom_safe_threshold: float | None = None,
    exclude_overlap_proxy: bool = True,
) -> np.ndarray:
    """
    用于 Pareto / HV / 局部采样参考前沿的安全样本掩码。
    历史样本仍然保留用于 surrogate 训练，但几何上高风险的点不再作为
    “继续学习”的参考前沿，避免 EHVI 继续朝错误方向扩张。
    """
    X = snap_discrete_vars(X)
    safe = np.all(geometry_rule_violations(X) <= 0.0, axis=1)

    if exclude_overlap_proxy:
        safe &= (overlap_proxy_violation(X) <= 0.0)

    if geom_warn_clf is not None and geom_safe_threshold is not None:
        safe &= (predict_geometry_safe_prob(geom_warn_clf, X) >= geom_safe_threshold)

    return safe


def load_and_clean_data(csv_path: str):
    require_current_performance_data(csv_path)
    df = read_optional_csv(csv_path)
    if len(df.columns) == 0:
        raise ValueError(f"TRAINING_CSV 为空或没有表头: {csv_path}")

    required_cols = VAR_NAMES + ALL_OUTPUT_NAMES + ['is_boundary']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSV 缺少必须列: {c}")

    if TARGET_PR_NAME not in df.columns:
        raise ValueError(
            f"CSV 中没有目标压比列 {TARGET_PR_NAME}。"
            f"请统一为 PressureRatio 或 totalpressureratio 之一。"
        )

    # 只保留统一目标
    use_cols = VAR_NAMES + ALL_OUTPUT_NAMES + ['is_boundary']
    df = df[use_cols].copy()

    # 保证 nBl 是整数
    nbl_idx = VAR_NAMES.index("nBl")
    df['nBl'] = np.clip(np.round(df['nBl']), L_BOUNDS[nbl_idx], U_BOUNDS[nbl_idx]).astype(int)

    # 简单去重
    df = df.drop_duplicates(subset=VAR_NAMES, keep='first').reset_index(drop=True)

    return df
# =============================================================================
# 数据划分：固定测试集
# =============================================================================
def split_with_fixed_testset(df: pd.DataFrame, test_csv=None, test_size=0.15):
    """
    固定测试集：
    - 第一次运行：从当前 df 中划分测试集并保存到 test_csv
    - 后续运行：读取 test_csv，并从当前 df 中按设计变量匹配剔除测试样本
    """
    test_csv = TEST_SET_CSV if test_csv is None else test_csv
    df = df.copy()

    if len(df) == 0:
        raise ValueError(
            "TRAINING_CSV 中没有可用于主动学习的样本。"
            "请先运行 DOE / Recover Runs 生成训练数据，并确认 Compressor_Training_Data.csv 不是空文件。"
        )

    if len(df) < 2:
        raise ValueError(
            f"TRAINING_CSV 中只有 {len(df)} 条可用样本，无法划分固定测试集。"
            "请先补充更多 DOE 样本后再运行主动学习。"
        )

    if (
        os.path.exists(test_csv)
        and os.path.getsize(test_csv) > 0
        and is_current_performance_data(test_csv)
    ):
        test_df = pd.read_csv(test_csv)

        # 用设计变量做匹配键
        def make_key(dataframe):
            return dataframe[VAR_NAMES].round(10).astype(str).agg('|'.join, axis=1)

        all_keys = make_key(df)
        test_keys = set(make_key(test_df))

        is_test = all_keys.isin(test_keys)

        df_test = df[is_test].copy()
        df_train = df[~is_test].copy()

        print(f"[加载固定测试集] {test_csv} | 测试集 {len(df_test)} | 训练池 {len(df_train)}")

        if len(df_test) == 0:
            raise ValueError("固定测试集未能在当前 TRAINING_CSV 中匹配到任何样本，请检查 fixed_test_set.csv 是否与当前数据一致。")

    else:
        stratify_labels = df['is_boundary'] if len(np.unique(df['is_boundary'])) > 1 else None
        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=stratify_labels
        )

        df_test.to_csv(test_csv, index=False)
        write_performance_data_metadata(test_csv)
        print(f"[保存固定测试集] {test_csv} | 测试集 {len(df_test)} | 训练池 {len(df_train)}")

    X_pool = df_train[VAR_NAMES].values.astype(float)
    Y_pool = df_train[ALL_OUTPUT_NAMES].values.astype(float)
    W_pool = df_train['is_boundary'].values.astype(float)

    X_test_fixed = df_test[VAR_NAMES].values.astype(float)
    Y_test_fixed = df_test[ALL_OUTPUT_NAMES].values.astype(float)
    W_test_fixed = df_test['is_boundary'].values.astype(float)

    return X_pool, X_test_fixed, Y_pool, Y_test_fixed, W_pool, W_test_fixed
# =============================================================================
# EHVI 内部辅助：2D 精确超体积与增量计算
# =============================================================================
def _hv2d_exact(Y: np.ndarray, ref: np.ndarray) -> float:
    """
    2D 最大化超体积，允许输入任意点集（内部自动提取非支配前沿）。
    Y: (K, 2) [eff, pr]
    ref: (2,) [ref_eff, ref_pr]
    """
    if len(Y) == 0:
        return 0.0
    mask = (Y[:, 0] > ref[0]) & (Y[:, 1] > ref[1])
    Yv = Y[mask]
    if len(Yv) == 0:
        return 0.0
    # 扫描线提取非支配前沿（O(N log N)）
    order = np.argsort(-Yv[:, 0])
    max_pr = -np.inf
    front_rows = []
    for i in order:
        if Yv[i, 1] > max_pr:
            front_rows.append(Yv[i])
            max_pr = Yv[i, 1]
    F = np.array(front_rows)[::-1]   # 按 eff 升序
    hv, prev_eff = 0.0, ref[0]
    for f in F:
        hv += (f[0] - prev_eff) * (f[1] - ref[1])
        prev_eff = f[0]
    return hv


def _batch_hvi_2d(
    Y_new: np.ndarray,
    pareto_Y: np.ndarray,
    ref: np.ndarray,
    hv_base: float
) -> np.ndarray:
    """
    批量计算 N 个候选点各自的 HVI（vectorized 支配检查 + 逐点精确计算）。
    Y_new: (N, 2)
    pareto_Y: (K, 2) 当前真实 Pareto 前沿
    ref: (2,)
    hv_base: 当前前沿的 HV 基准值
    Returns: (N,) HVI，被支配的点返回 0
    """
    N = len(Y_new)
    hvi = np.zeros(N)

    if len(pareto_Y) == 0:
        # 无当前前沿：HVI = 高于参考点的矩形面积
        hvi = (np.maximum(0.0, Y_new[:, 0] - ref[0]) *
               np.maximum(0.0, Y_new[:, 1] - ref[1]))
        return hvi

    # 向量化支配检查：pareto_Y[None,:,:] >= Y_new[:,None,:]
    dominated = np.any(
        np.all(pareto_Y[None, :, :] >= Y_new[:, None, :], axis=2),
        axis=1
    )   # (N,)

    for i in np.where(~dominated)[0]:
        y = Y_new[i]
        # 移除被 y 支配的前沿点
        not_dom = ~np.all(y[None, :] >= pareto_Y, axis=1)
        new_front = np.vstack([pareto_Y[not_dom], y[None, :]])
        hv_new = _hv2d_exact(new_front, ref)
        hvi[i] = max(0.0, hv_new - hv_base)

    return hvi


def _generate_candidates_mixed(
    current_pareto_X: np.ndarray,
    cfg: ALConfig
) -> np.ndarray:
    """
    混合候选点生成：
    (1-ratio) 全局 LHS + ratio 在当前 Pareto 前沿设计变量附近高斯扰动。
    """
    n_local  = int(cfg.n_candidates * cfg.local_sample_ratio)
    n_global = cfg.n_candidates - n_local

    sampler = qmc.LatinHypercube(d=len(VAR_NAMES), seed=None)
    X_global = snap_discrete_vars(
        qmc.scale(sampler.random(n_global), L_BOUNDS, U_BOUNDS)
    )
    X_global = pin_optimization_operating_point(X_global)

    if current_pareto_X is not None and len(current_pareto_X) > 0:
        sigma = (U_BOUNDS - L_BOUNDS) * 0.06   # 各维度范围的 6%
        idx = np.random.choice(len(current_pareto_X), n_local, replace=True)
        X_local = current_pareto_X[idx] + np.random.randn(n_local, len(VAR_NAMES)) * sigma
        X_local = np.clip(X_local, L_BOUNDS, U_BOUNDS)
        X_local = pin_optimization_operating_point(X_local)
    else:
        sampler2 = qmc.LatinHypercube(d=len(VAR_NAMES), seed=None)
        X_local = snap_discrete_vars(
            qmc.scale(sampler2.random(n_local), L_BOUNDS, U_BOUNDS)
        )
        X_local = pin_optimization_operating_point(X_local)

    return np.vstack([X_global, X_local])
def compute_true_cumulative_hv(
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    W_pool: np.ndarray,
    geom_warn_clf=None,
    geom_safe_threshold: float | None = None,
    exclude_overlap_proxy: bool = False,
    ref_eff: float = TRUE_HV_REF_EFF,
    ref_pr: float = TRUE_HV_REF_PR,
    fixed_p_out: float | None = None,
    p_out_tolerance: float | None = None,
):
    """
    返回 (hv_val, front_Y, front_X)。已完成 CFD 的成功点默认不再被
    overlap 软代理否决；该代理只用于未计算候选的采集惩罚。
    front_Y: (K, 2) 非支配前沿的 [Efficiency, PR]
    front_X: (K, 14) 对应的设计变量，供局部采样使用
    """
    X_pool = snap_discrete_vars(X_pool)
    active_p_out = OPTIMIZATION_P_OUT if fixed_p_out is None else fixed_p_out
    active_tolerance = (
        OPERATING_POINT_P_OUT_TOLERANCE
        if p_out_tolerance is None
        else p_out_tolerance
    )
    operating_point_ok = (
        np.abs(X_pool[:, P_OUT_IDX] - float(active_p_out)) <= float(active_tolerance)
    )
    geom_ok = geometry_safe_mask(
        X_pool,
        geom_warn_clf=geom_warn_clf,
        geom_safe_threshold=geom_safe_threshold,
        exclude_overlap_proxy=exclude_overlap_proxy,
    )

    feas_mask = (
        operating_point_ok &
        geom_ok &
        (Y_pool[:, 0] >= MIN_EFFICIENCY) &
        (Y_pool[:, 3] >= BOUNDARY_FLOW_G_S)
    )

    Y_feas = Y_pool[feas_mask][:, :2]
    X_feas = X_pool[feas_mask]

    if len(Y_feas) == 0:
        return np.nan, None, None

    ref = np.array([ref_eff, ref_pr])
    pareto_idx = NonDominatedSorting().do(-Y_feas, only_non_dominated_front=True)
    front_Y = Y_feas[pareto_idx]
    front_X = X_feas[pareto_idx]          # ← 新增返回值

    hv_val = HV(ref_point=np.array([-ref_eff, -ref_pr])).do(-front_Y)
    return hv_val, front_Y, front_X

def extract_surrogate_front_and_hv(
    res,
    reg_model,
    scaler_X,
    scaler_Y,
    ref_eff: float = TRUE_HV_REF_EFF,
    ref_pr: float = TRUE_HV_REF_PR
):
    """
    辅指标：基于 NSGA-II 返回的 res.X，重新做纯 surrogate 性能预测，
    再提取 [Efficiency, PR] 的非支配前沿并计算 surrogate HV。
    """
    if res.X is None or len(res.X) == 0:
        return None, None, np.nan

    X_pf = expand_geometry_decisions(np.atleast_2d(res.X))
    X_pf_norm = scaler_X.transform(X_pf)

    mean_real, _ = mc_dropout_predict(reg_model, X_pf_norm, scaler_Y, n_samples=50)
    Y_perf = mean_real[:, :2]   # [eff, pr]

    pareto_idx = NonDominatedSorting().do(-Y_perf, only_non_dominated_front=True)
    pareto_X = X_pf[pareto_idx]
    pareto_Y = Y_perf[pareto_idx]

    if len(pareto_Y) == 0:
        return None, None, np.nan

    surrogate_hv = HV(ref_point=np.array([-ref_eff, -ref_pr])).do(-pareto_Y)
    return pareto_X, pareto_Y, surrogate_hv
# =============================================================================
# 网络模型
# =============================================================================
class PerformanceSurrogate(nn.Module):
    """
    回归模型：输出 3 个性能量
    """
    def __init__(self, input_dim=14, output_dim=3):
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

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class BoundaryClassifierNN(nn.Module):
    """
    可选：边界分类网络
    """
    def __init__(self, input_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# =============================================================================
# 损失函数
# =============================================================================
def weighted_regression_loss(
    pred,
    target,
    is_boundary,
    sample_weights=None,
):
    per_sample = ((pred - target) ** 2).mean(dim=1)
    weights = 1.0 + (BOUNDARY_WEIGHT - 1.0) * is_boundary
    if sample_weights is not None:
        weights = weights * sample_weights
    return (per_sample * weights).mean()


# =============================================================================
# 训练与推断
# =============================================================================
def train_regressor(
    X_train, Y_train, W_train,
    X_val, Y_val, W_val,
    save_path=BEST_REG_PATH,
    sample_weights_train=None,
    max_epochs: int = 1200,
    patience: int = 40,
    random_seed: int | None = None,
):
    if random_seed is not None:
        torch.manual_seed(int(random_seed))

    model = PerformanceSurrogate(
        input_dim=len(VAR_NAMES),
        output_dim=len(SURROGATE_OUTPUT_NAMES),
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-5)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    Ytr = torch.tensor(Y_train, dtype=torch.float32, device=DEVICE)
    Wtr = torch.tensor(W_train, dtype=torch.float32, device=DEVICE)

    Xva = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    Yva = torch.tensor(Y_val, dtype=torch.float32, device=DEVICE)
    Wva = torch.tensor(W_val, dtype=torch.float32, device=DEVICE)
    Str = (
        torch.tensor(sample_weights_train, dtype=torch.float32, device=DEVICE)
        if sample_weights_train is not None
        else None
    )

    best_val = np.inf
    counter = 0
    best_state = None

    history = []

    for epoch in range(int(max_epochs)):
        model.train()
        optimizer.zero_grad()
        pred = model(Xtr)
        loss = weighted_regression_loss(pred, Ytr, Wtr, Str)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva)
            val_loss = weighted_regression_loss(val_pred, Yva, Wva).item()

        history.append((epoch, loss.item(), val_loss))

        if val_loss < best_val:
            best_val = val_loss
            counter = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            counter += 1

        if counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model, history


def train_feasibility_classifier(X_success, X_failed, min_failed_required=4):
    """
    改进后的可行性分类器：引入最小启动阈值与动态正则化，解决早期类别极度不平衡问题。
    """
    n_success = len(X_success)
    n_failed = len(X_failed)
    
    # 1. 启动阈值判断：如果失败样本太少，返回 None。
    # 此时预测概率 p_feas 将全为 1.0，依靠 acquisition 函数中的 d_fail 距离惩罚来避开失败点。
    if n_failed < min_failed_required:
        if n_failed > 0:
            print(f"  [可行性分类器] 失败样本过少 ({n_failed} < {min_failed_required})，暂不启动分类器，依靠距离惩罚兜底。")
        else:
            print("  [可行性分类器] 当前无失败样本，默认全部可行。")
        return None
        
    print(f"  [可行性分类器] 满足启动条件 (成功: {n_success}, 失败: {n_failed})，开始训练...")

    X_all_raw = snap_discrete_vars(np.vstack([X_success, X_failed]))
    X_all = X_all_raw[:, FEASIBILITY_VAR_IDX]
    y_all = np.hstack([np.ones(n_success), np.zeros(n_failed)])

    # 2. 动态正则化：失败样本越少，对树模型的约束越强，防止过拟合
    if n_failed < 10:
        # 极少失败样本：限制为极其简单的弱分类器集合
        max_depth = 4
        min_samples_leaf = 2
        min_samples_split = 4
    elif n_failed < 30:
        # 中等失败样本：稍微放宽
        max_depth = 8
        min_samples_leaf = 3
        min_samples_split = 4
    else:
        max_depth = 12
        min_samples_leaf = 2
        min_samples_split = 2

    # 3. 训练 RF 分类器
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=42,
        class_weight="balanced_subsample", 
        max_features="sqrt"
    )
    
    clf.fit(X_all, y_all)
    clf.impeller_feature_names_ = tuple(FEASIBILITY_FEATURE_NAMES)
    clf.impeller_excluded_feature_names_ = tuple(
        sorted(FAILURE_CLASSIFIER_EXCLUDED_FEATURES)
    )
    clf.impeller_classifier_schema_version_ = 2
    return clf


def train_boundary_classifier(X_norm, y_boundary):
    if len(np.unique(y_boundary)) < 2:
        return None

    Xtr, Xva, ytr, yva = train_test_split(
        X_norm, y_boundary, test_size=0.2, random_state=42, stratify=y_boundary
    )

    model = BoundaryClassifierNN(input_dim=14).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.001)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)

    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=DEVICE)

    bce = nn.BCEWithLogitsLoss()

    best_auc = -np.inf
    best_state = None
    patience = 30
    counter = 0

    for epoch in range(500):
        model.train()
        opt.zero_grad()
        logits = model(Xtr_t)
        loss = bce(logits, ytr_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva_t).cpu().numpy()
            val_prob = 1 / (1 + np.exp(-val_logits))
            auc = roc_auc_score(yva, val_prob)

        if auc > best_auc:
            best_auc = auc
            counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            counter += 1

        if counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def mc_dropout_predict(model, X_norm, scaler_Y, n_samples=MC_SAMPLES):
    model.train()
    X_t = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)

    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(X_t).cpu().numpy())

    preds = np.stack(preds, axis=0)              # (S, N, 3)
    mean_norm = preds.mean(axis=0)
    std_norm = preds.std(axis=0)

    mean_real = scaler_Y.inverse_transform(mean_norm)
    return mean_real, std_norm


def deterministic_predict(model, X_norm, scaler_Y):
    model.eval()
    X_t = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        y_norm = model(X_t).cpu().numpy()
    return scaler_Y.inverse_transform(y_norm)


def normalized_std_to_real(std_norm: np.ndarray, scaler_Y) -> np.ndarray:
    """Convert MinMax-scaled predictive standard deviation to physical units."""
    scale = np.asarray(scaler_Y.scale_, dtype=float)
    safe_scale = np.where(np.abs(scale) > 1e-12, np.abs(scale), np.nan)
    return np.asarray(std_norm, dtype=float) / safe_scale


def run_kfold_surrogate_validation(
    X_raw: np.ndarray,
    Y_surr: np.ndarray,
    W_boundary: np.ndarray,
    al_iter: int,
    cfg: ALConfig,
) -> tuple[list[dict], dict]:
    """Leakage-safe K-fold validation with fold-local scalers and train weights."""
    X_raw = snap_discrete_vars(X_raw)
    Y_surr = np.asarray(Y_surr, dtype=float)
    W_boundary = np.asarray(W_boundary, dtype=float)
    n_samples = len(X_raw)
    n_splits = min(int(cfg.cv_folds), n_samples)
    if n_splits < 2:
        return [], {
            "cv_status": "insufficient_samples",
            "cv_folds": int(n_splits),
        }

    nbl_idx = VAR_NAMES.index("nBl")
    nbl_values = np.rint(X_raw[:, nbl_idx]).astype(int)
    strat_labels = np.array(
        [f"{nbl}:{int(boundary)}" for nbl, boundary in zip(nbl_values, W_boundary)]
    )
    _, strat_counts = np.unique(strat_labels, return_counts=True)
    if len(strat_counts) > 0 and int(strat_counts.min()) >= n_splits:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )
        split_iter = splitter.split(X_raw, strat_labels)
        split_method = "stratified_nBl_boundary"
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(X_raw)
        split_method = "kfold"

    fold_rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
        fold_scaler_X = MinMaxScaler()
        fold_scaler_Y = MinMaxScaler()
        X_train_norm = fold_scaler_X.fit_transform(X_raw[train_idx])
        X_val_norm = fold_scaler_X.transform(X_raw[val_idx])
        Y_train_norm = fold_scaler_Y.fit_transform(Y_surr[train_idx])
        Y_val_norm = fold_scaler_Y.transform(Y_surr[val_idx])

        train_weights, weight_diag = compute_density_nbl_sample_weights(
            X_train_norm,
            nbl_values[train_idx],
            k_neighbors=cfg.density_k_neighbors,
            min_weight=cfg.sample_weight_min,
            max_weight=cfg.sample_weight_max,
        )
        fold_model, fold_history = train_regressor(
            X_train_norm,
            Y_train_norm,
            W_boundary[train_idx],
            X_val_norm,
            Y_val_norm,
            W_boundary[val_idx],
            save_path=None,
            sample_weights_train=train_weights,
            max_epochs=cfg.cv_max_epochs,
            patience=cfg.cv_patience,
            random_seed=10000 + int(al_iter) * 100 + fold_idx,
        )
        fold_pred = deterministic_predict(fold_model, X_val_norm, fold_scaler_Y)
        row = {
            "iter": int(al_iter),
            "fold": int(fold_idx),
            "n_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "split_method": split_method,
            "epochs": int(len(fold_history)),
            "recorded_at_utc": utc_timestamp(),
            **regression_metrics(Y_surr[val_idx], fold_pred),
            **weight_diag,
        }
        fold_rows.append(row)

    metric_columns = [
        f"{metric}_{suffix}"
        for metric in ("mse", "rmse", "mae", "r2")
        for suffix in ("eff", "pr", "mf", "macro")
    ]
    summary = {
        "cv_status": "completed",
        "cv_folds": int(n_splits),
        "cv_split_method": split_method,
    }
    for column in metric_columns:
        values = np.array([row[column] for row in fold_rows], dtype=float)
        summary[f"cv_mean_{column}"] = float(np.nanmean(values))
        summary[f"cv_std_{column}"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return fold_rows, summary


def predict_boundary_prob(boundary_model, X_norm):
    if boundary_model is None:
        return np.zeros(len(X_norm))
    boundary_model.eval()
    X_t = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = boundary_model(X_t).cpu().numpy()
    return 1 / (1 + np.exp(-logits))


def _classifier_positive_probability(clf, features, positive_label=1):
    probabilities = clf.predict_proba(features)
    classes = np.asarray(clf.classes_)
    matches = np.flatnonzero(classes == positive_label)
    if len(matches) == 0:
        return np.zeros(len(features), dtype=float)
    return probabilities[:, int(matches[0])]


def _legacy_classifier_probability_marginalized_over_nbl(
    clf,
    X_raw,
    feature_idx,
    positive_label=1,
):
    """Remove direct nBl dependence from a persisted pre-v2 classifier.

    Old classifiers cannot be transformed in place.  Averaging their output at
    every allowed nBl value preserves their geometry/operating knowledge while
    making the result invariant to the candidate's actual blade count.
    """
    X_raw = snap_discrete_vars(X_raw)
    nbl_idx = VAR_NAMES.index("nBl")
    allowed_nbl = np.arange(
        int(np.ceil(L_BOUNDS[nbl_idx])),
        int(np.floor(U_BOUNDS[nbl_idx])) + 1,
        dtype=float,
    )
    variants = np.repeat(X_raw[None, :, :], len(allowed_nbl), axis=0)
    variants[:, :, nbl_idx] = allowed_nbl[:, None]
    features = variants.reshape(-1, len(VAR_NAMES))[:, feature_idx]
    probabilities = _classifier_positive_probability(
        clf,
        features,
        positive_label=positive_label,
    )
    return probabilities.reshape(len(allowed_nbl), len(X_raw)).mean(axis=0)


def predict_feasible_prob(feas_clf, X_raw):
    if feas_clf is None:
        return np.ones(len(X_raw))
    X_raw = snap_discrete_vars(X_raw)
    n_features = int(getattr(feas_clf, "n_features_in_", -1))
    if n_features == len(FEASIBILITY_FEATURE_NAMES):
        return _classifier_positive_probability(
            feas_clf,
            X_raw[:, FEASIBILITY_VAR_IDX],
            positive_label=1,
        )
    if n_features == len(VAR_NAMES):
        return _legacy_classifier_probability_marginalized_over_nbl(
            feas_clf,
            X_raw,
            np.arange(len(VAR_NAMES), dtype=int),
            positive_label=1,
        )
    raise ValueError(
        "Unsupported feasibility-classifier feature count: "
        f"{n_features}; expected {len(FEASIBILITY_FEATURE_NAMES)} (current) "
        f"or {len(VAR_NAMES)} (legacy)."
    )


def load_geometry_summary(work_dir: str):
    path = os.path.join(work_dir, GEOMETRY_SUMMARY_PATH)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"     [几何摘要] 读取失败: {e}")
        return None


def resolve_geometry_runs_dir() -> str | None:
    candidates = [AL_WORKING_BASE, os.path.join(os.getcwd(), GEOM_WARN_RUNS_DIR_FALLBACK)]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return None


def extract_geometry_warning_case(case_dir: str):
    summary = load_geometry_summary(case_dir)

    path = os.path.join(case_dir, "0908-2.cft-res")
    log_path = os.path.join(case_dir, "run_cfturbo.log")
    if not os.path.exists(path):
        return None
    if summary is None:
        summary = {
            "warn_sweep": False,
            "warn_overlap": False,
            "warn_internal_blade_thickness": False,
        }
        if os.path.exists(log_path):
            try:
                log_text = open(log_path, "r", encoding="utf-8", errors="ignore").read()
                summary["warn_sweep"] = "Very high tangential leading edge sweep angle" in log_text
                summary["warn_overlap"] = "Overlapping of adjacent blades might be too low" in log_text
                summary["warn_internal_blade_thickness"] = "Internal blade thickness is lower than specified" in log_text
            except Exception:
                pass

    try:
        xml = ET.parse(path)
        root = xml.getroot()
        values = {
            "d1s": float(root.findtext(".//Updates//dS")),
            "dH": float(root.findtext(".//Updates//dH")),
            "beta1hb": np.degrees(float(root.find(".//Updates//Beta1/Value[@Index='0']").text)),
            "beta1sb": np.degrees(float(root.find(".//Updates//Beta1/Value[@Index='1']").text)),
            "d2": float(root.findtext(".//Updates//d2")),
            "b2": float(root.findtext(".//Updates//b2")),
            "beta2hb": np.degrees(float(root.find(".//Updates//Beta2/Value[@Index='0']").text)),
            "beta2sb": np.degrees(float(root.find(".//Updates//Beta2/Value[@Index='1']").text)),
            "Lz": float(root.findtext(".//Updates//DeltaZ")),
            "t": float(root.findtext(".//Updates//sLEH")),
            "TipClear": float(root.findtext(".//Updates//xTipInlet")),
            "nBl": int(round(float(root.findtext(".//Updates//nBl")))),
            "rake_te_s": np.degrees(float(root.find(".//Output//RakeTE/Value[@Index='1']").text)),
        }
    except Exception:
        return None

    x = np.array([values[name] for name in GEOM_WARN_FEATURE_NAMES], dtype=float)
    y_bad = float(
        bool(summary.get("warn_sweep"))
        or bool(summary.get("warn_overlap"))
        or bool(summary.get("warn_internal_blade_thickness"))
    )
    return x, y_bad


def load_geometry_warning_dataset():
    runs_dir = resolve_geometry_runs_dir()
    if runs_dir is None:
        return None, None

    cases = []
    for entry in sorted(os.listdir(runs_dir)):
        if not entry.startswith("AL_"):
            continue
        parsed = extract_geometry_warning_case(os.path.join(runs_dir, entry))
        if parsed is not None:
            cases.append(parsed)

    if not cases:
        return None, None

    X = np.array([c[0] for c in cases], dtype=float)
    y = np.array([c[1] for c in cases], dtype=float)
    return X, y


def load_recorded_failure_datasets():
    """Load confirmed DOE/AL failures using the correct feature spaces."""
    return load_failure_training_sets(
        FAILURE_RECORDS_CSV,
        VAR_NAMES,
        GEOMETRY_VAR_NAMES,
    )


def build_geometry_classifier_dataset(
    X_success_full: np.ndarray,
    X_geometry_failed: np.ndarray,
):
    """Build an nBl-neutral geometry classifier from confirmed failures."""
    frames = []
    X_success_full = np.asarray(X_success_full, dtype=float)
    if len(X_success_full) > 0:
        frames.append(
            pd.DataFrame(
                np.column_stack(
                    [X_success_full[:, GEOMETRY_VAR_IDX], np.zeros(len(X_success_full))]
                ),
                columns=GEOMETRY_VAR_NAMES + ["is_bad"],
            )
        )

    X_geometry_failed = np.asarray(X_geometry_failed, dtype=float)
    if X_geometry_failed.size > 0:
        X_geometry_failed = X_geometry_failed.reshape(-1, len(GEOMETRY_VAR_NAMES))
        frames.append(
            pd.DataFrame(
                np.column_stack(
                    [X_geometry_failed, np.ones(len(X_geometry_failed))]
                ),
                columns=GEOMETRY_VAR_NAMES + ["is_bad"],
            )
        )

    if not frames:
        return None, None
    combined = pd.concat(frames, ignore_index=True)
    # nBl is intentionally absent from the learned classifier.  If otherwise
    # identical feature rows occur at different blade counts, retain the
    # conservative label while the explicit nBl/rake overlap rule handles the
    # known blade-count physics separately.
    combined = (
        combined.groupby(GEOM_WARN_FEATURE_NAMES, as_index=False, dropna=False)["is_bad"]
        .max()
    )
    return (
        combined[GEOM_WARN_FEATURE_NAMES].to_numpy(dtype=float),
        combined["is_bad"].to_numpy(dtype=float),
    )


def train_geometry_feasibility_classifier(X_geom, y_bad, min_bad_required=4):
    if X_geom is None or y_bad is None or len(X_geom) == 0:
        return None
    if len(np.unique(y_bad)) < 2 or int(y_bad.sum()) < min_bad_required:
        return None

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=4,
        random_state=42,
        class_weight="balanced_subsample",
        max_features="sqrt"
    )
    clf.fit(X_geom, y_bad.astype(int))
    clf.impeller_feature_names_ = tuple(GEOM_WARN_FEATURE_NAMES)
    clf.impeller_excluded_feature_names_ = tuple(
        sorted(FAILURE_CLASSIFIER_EXCLUDED_FEATURES)
    )
    clf.impeller_classifier_schema_version_ = 2
    return clf


def predict_geometry_safe_prob(geom_warn_clf, X_raw):
    if geom_warn_clf is None:
        return np.ones(len(X_raw))
    X_raw = snap_discrete_vars(X_raw)
    n_features = int(getattr(geom_warn_clf, "n_features_in_", -1))
    if n_features == len(GEOM_WARN_FEATURE_NAMES):
        p_bad = _classifier_positive_probability(
            geom_warn_clf,
            X_raw[:, GEOM_WARN_VAR_IDX],
            positive_label=1,
        )
    elif n_features == len(GEOMETRY_VAR_NAMES):
        p_bad = _legacy_classifier_probability_marginalized_over_nbl(
            geom_warn_clf,
            X_raw,
            GEOMETRY_VAR_IDX,
            positive_label=1,
        )
    else:
        raise ValueError(
            "Unsupported geometry-classifier feature count: "
            f"{n_features}; expected {len(GEOM_WARN_FEATURE_NAMES)} (current) "
            f"or {len(GEOMETRY_VAR_NAMES)} (legacy)."
        )
    return 1.0 - p_bad


# =============================================================================
# CFD 调用
# =============================================================================
def run_single_cfd(x_cand: np.ndarray, run_id: str):
    current_work_dir = os.path.join(AL_WORKING_BASE, run_id)
    os.makedirs(current_work_dir, exist_ok=True)

    x_cand = snap_discrete_vars(x_cand)[0]
    param_dict = dict(zip(VAR_NAMES, x_cand))

    cmd = [
        r"C:\Program Files\PowerShell\7\pwsh.exe",
        "-ExecutionPolicy", "Bypass",
        "-File", PS_SCRIPT_PATH,
        "-WorkingDir", current_work_dir,
        "-TurboGridTemplate", TURBOGRID_TEMPLATE,
    ]

    for name in VAR_NAMES:
        if name == 'P_out':
            continue
        val = int(round(param_dict[name])) if name == 'nBl' else param_dict[name]
        cmd.extend([f"-{name}", str(val)])

    cmd.extend(["-mFlow", "0.0036", "-N_rpm", "10000", "-alpha0", "0.0"])

    try:
        print(f"     [{run_id}] 正在生成几何与网格...")
        ps_result = subprocess.run(
            cmd, capture_output=True, text=True, creationflags=CREATE_NO_WINDOW
        )

        geometry_summary = load_geometry_summary(current_work_dir)

        if ps_result.returncode != 0:
            reason = f"几何/网格失败，Exit Code: {ps_result.returncode}"
            print(f"     [{run_id}] {reason}")
            structured_failure = load_structured_failure_status(current_work_dir)
            return False, None, geometry_summary, {
                "stage": (
                    structured_failure["stage"]
                    if structured_failure is not None
                    else classify_geometry_failure_stage(
                        reason,
                        f"{ps_result.stdout}\n{ps_result.stderr}",
                    )
                ),
                "reason": (
                    structured_failure["reason"]
                    if structured_failure is not None
                    else reason
                ),
            }

        p_out_val = param_dict['P_out']
        nbl_val = int(round(param_dict['nBl']))

        print(f"     [{run_id}] 网格完成，启动 CFX（背压: {p_out_val:.3f} Pa）...")
        success_cfx, cfx_res, msg = run_cfx_pipeline(
            current_work_dir, run_id, p_out=p_out_val, cores=8, n_blades=nbl_val
        )

        if not success_cfx:
            print(f"     [{run_id}] CFD 失败: {msg}")
            return False, None, geometry_summary, {
                "stage": classify_cfx_failure_stage(msg),
                "reason": msg,
            }

        true_y = np.array([
            cfx_res['Efficiency'],
            cfx_res[TARGET_PR_NAME] if TARGET_PR_NAME in cfx_res else cfx_res['totalpressureratio'],
            cfx_res['Power'],
            cfx_res['MassFlow'],
        ], dtype=float)

        if not np.all(np.isfinite(true_y)):
            return False, None, geometry_summary, {
                "stage": "physical_invalid",
                "reason": "CFD returned NaN or infinite performance values.",
            }
        if float(true_y[3]) < MIN_DISCARD_FLOW_G_S:
            return False, None, geometry_summary, {
                "stage": "physical_invalid",
                "reason": (
                    f"MassFlow={true_y[3]:.6g} g/s is below the discard "
                    f"threshold {MIN_DISCARD_FLOW_G_S:.6g} g/s."
                ),
            }

        return True, true_y, geometry_summary, None

    except Exception as e:
        print(f"     [{run_id}] 未知异常: {e}")
        return False, None, None, {
            "stage": "infrastructure",
            "reason": str(e),
        }


def print_geometry_summary(geometry_summary: dict | None):
    if not geometry_summary:
        return

    parts = []
    if geometry_summary.get("parsed"):
        if geometry_summary.get("beta1_diff_deg") is not None:
            parts.append(f"beta1差={geometry_summary['beta1_diff_deg']:.2f} deg")
        if geometry_summary.get("overlap_factor_min") is not None:
            parts.append(f"overlap最小因子={geometry_summary['overlap_factor_min']:.3f}")
    if geometry_summary.get("warn_sweep"):
        parts.append("CFturbo警告:sweep")
    if geometry_summary.get("warn_overlap"):
        parts.append("CFturbo警告:overlap")
    if geometry_summary.get("warn_internal_blade_thickness"):
        parts.append("CFturbo警告:thickness")

    if parts:
        print("     [几何摘要] " + " | ".join(parts))


# =============================================================================
# 主动学习候选采集
# =============================================================================
# =============================================================================
# 主动学习采集：EHVI
# =============================================================================
def compute_ehvi_acquisition(
    reg_model,
    feas_clf,
    geom_warn_clf,
    scaler_X,
    scaler_Y,
    X_pool_raw: np.ndarray,
    failed_points_raw: list,
    current_pareto_Y: np.ndarray,   # 真实 Pareto 前沿 Y，None 表示尚无
    current_pareto_X: np.ndarray,   # 真实 Pareto 前沿 X，用于局部采样
    ref_eff: float,
    ref_pr: float,
    cfg: ALConfig
):
    ref = np.array([ref_eff, ref_pr])

    # ------------------------------------------------------------------
    # 1. 混合候选点生成
    # ------------------------------------------------------------------
    X_cand = _generate_candidates_mixed(current_pareto_X, cfg)
    X_norm = scaler_X.transform(X_cand)
    N = len(X_cand)

    # ------------------------------------------------------------------
    # 2. 预筛选（只用 5 次 MC 快速估计，避免对无效点做完整 EHVI）
    # ------------------------------------------------------------------
    p_feas = predict_feasible_prob(feas_clf, X_cand)
    p_geom_safe = predict_geometry_safe_prob(geom_warn_clf, X_cand)
    geom_g = geometry_rule_violations(X_cand)
    invalid_geom = np.any(geom_g > 0.0, axis=1)

    # 快速均值预测（5 次 MC，仅用于剪枝）
    mean_quick, std_quick_norm = mc_dropout_predict(
        reg_model, X_norm, scaler_Y, n_samples=5
    )
    pred_eff_q = mean_quick[:, 0]
    pred_mf_q  = mean_quick[:, 2]

    # Exploitation is deliberately conservative: it uses both learned
    # feasibility models and the surrogate's predicted performance limits.
    exploitation_mask = (
        (p_feas >= cfg.feasible_prob_threshold_pick) &
        (p_geom_safe >= cfg.geom_safe_prob_threshold_pick) &
        (~invalid_geom) &
        (pred_mf_q >= max(MIN_DISCARD_FLOW_G_S, BOUNDARY_FLOW_G_S - 0.05)) &
        (pred_eff_q >= MIN_EFFICIENCY) &
        (pred_eff_q <= 0.85)
    )

    # Exploration must be able to challenge the learned models.  Restrict it
    # only by explicit geometry rules, finite surrogate output and proximity
    # to confirmed failures; otherwise a wrong low prediction would prevent
    # the uncertainty/coverage slots from ever collecting corrective CFD data.
    exploration_mask = (
        (~invalid_geom)
        & np.all(np.isfinite(mean_quick), axis=1)
        & np.all(np.isfinite(std_quick_norm), axis=1)
    )

    # 失败点距离过滤（太近的直接排除）
    failed_norm = None
    if len(failed_points_raw) > 0:
        failed_raw  = snap_discrete_vars(np.array(failed_points_raw))
        failed_norm = scaler_X.transform(failed_raw)
        d_fail_all  = calc_distance_to_set(X_norm, failed_norm)
        sufficiently_far_from_failure = d_fail_all > 0.04
        exploitation_mask &= sufficiently_far_from_failure
        exploration_mask &= sufficiently_far_from_failure

    valid_idx = np.where(exploitation_mask)[0]

    ehvi = np.full(N, -np.inf)

    if len(valid_idx) == 0:
        print(
            "  [EHVI] 警告：预筛选后无开发候选点；"
            f"仍保留 {int(exploration_mask.sum())} 个探索候选。"
        )
        return X_cand, ehvi, {
            "pred_mean": mean_quick,
            "pred_std_norm": std_quick_norm,
            "p_feas": p_feas,
            "p_geom_safe": p_geom_safe,
            "valid_mask": exploitation_mask,
            "exploration_mask": exploration_mask,
            "overlap_proxy_violation": overlap_proxy_violation(X_cand),
        }

    # ------------------------------------------------------------------
    # 3. 对有效候选点做完整 MC Dropout（cfg.mc_samples 次）
    # ------------------------------------------------------------------
    X_valid      = X_cand[valid_idx]
    X_valid_norm = X_norm[valid_idx]

    reg_model.train()
    X_t = torch.tensor(X_valid_norm, dtype=torch.float32, device=DEVICE)
    mc_preds_norm = []
    with torch.no_grad():
        for _ in range(cfg.mc_samples):
            mc_preds_norm.append(reg_model(X_t).cpu().numpy())
    mc_preds_norm = np.stack(mc_preds_norm, axis=0)  # (S, N_valid, 3)
    mc_preds = scaler_Y.inverse_transform(
        mc_preds_norm.reshape(-1, mc_preds_norm.shape[-1])
    ).reshape(mc_preds_norm.shape)

    # ------------------------------------------------------------------
    # 4. EHVI 计算
    # ------------------------------------------------------------------
    pareto_Y_cur = (current_pareto_Y
                    if current_pareto_Y is not None and len(current_pareto_Y) > 0
                    else np.zeros((0, 2)))
    hv_base = _hv2d_exact(pareto_Y_cur, ref)

    S = mc_preds.shape[0]
    ehvi_valid = np.zeros(len(valid_idx))

    for s in range(S):
        Y_s   = mc_preds[s, :, :2]                              # (N_valid, 2)
        hvi_s = _batch_hvi_2d(Y_s, pareto_Y_cur, ref, hv_base)  # (N_valid,)
        ehvi_valid += hvi_s

    ehvi_valid /= S

    # 可行性概率加权
    ehvi_valid *= p_feas[valid_idx]
    ehvi_valid *= p_geom_safe[valid_idx]

    # overlap 风险软惩罚：保留这些点，但显著降低其采样优先级。
    overlap_violation_valid = overlap_proxy_violation(X_valid)
    ehvi_valid *= np.exp(-overlap_violation_valid / OVERLAP_EHVI_DECAY_DEG)

    # 失败点软惩罚
    if failed_norm is not None:
        d_fail_valid = calc_distance_to_set(X_valid_norm, failed_norm)
        ehvi_valid  *= (1.0 - 0.5 * np.exp(-d_fail_valid / 0.08))

    ehvi[valid_idx] = ehvi_valid

    print(f"  [EHVI] 有效候选点: {len(valid_idx)}/{N} | "
          f"HV基准: {hv_base:.5f} | "
          f"最大EHVI: {ehvi_valid.max():.6f}")

    info = {
        "pred_mean": mean_quick,
        "pred_std_norm": std_quick_norm,
        "p_feas": p_feas,
        "p_geom_safe": p_geom_safe,
        "valid_mask": exploitation_mask,
        "exploration_mask": exploration_mask,
        "overlap_proxy_violation": overlap_proxy_violation(X_cand),
    }
    info["pred_std_norm"][valid_idx] = mc_preds_norm.std(axis=0)
    return X_cand, ehvi, info



# =============================================================================
# NSGA-II 优化问题
# =============================================================================
class CompressorMOOProblem(Problem):
    def __init__(
        self,
        reg_model,
        feas_clf,
        geom_warn_clf,
        scaler_X,
        scaler_Y,
        X_pool_raw,
        Y_pool_raw,
        cfg: ALConfig,
        fixed_p_out: float | None = None,
    ):
        self.reg_model = reg_model
        self.feas_clf = feas_clf
        self.geom_warn_clf = geom_warn_clf
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.X_pool_raw = snap_discrete_vars(X_pool_raw)
        self.Y_pool_raw = Y_pool_raw
        self.cfg = cfg
        self.fixed_p_out = float(
            OPTIMIZATION_P_OUT if fixed_p_out is None else fixed_p_out
        )

       
        
        self.eff_max_phys = 0.84
        self.eff_min_phys = 0.6
        n_rules = geometry_rule_violations(self.X_pool_raw[:1]).shape[1]

        super().__init__(

            n_var=len(GEOMETRY_VAR_NAMES),
            n_obj=2,
            n_ieq_constr=5 + n_rules,
            xl=L_BOUNDS[GEOMETRY_VAR_IDX],
            xu=U_BOUNDS[GEOMETRY_VAR_IDX],
        )

    def _evaluate(self, X, out, *args, **kwargs):
        X_full = expand_geometry_decisions(X, fixed_p_out=self.fixed_p_out)
        X_norm = self.scaler_X.transform(X_full)

        mean_real, std_norm = mc_dropout_predict(
            self.reg_model, X_norm, self.scaler_Y, n_samples=20
        )

        eff_raw = mean_real[:, 0]
        pr_raw  = mean_real[:, 1]
        mf      = mean_real[:, 2]
        unc = std_norm[:, :2].mean(axis=1)
        p_feas = predict_feasible_prob(self.feas_clf, X_full)
        p_geom_safe = predict_geometry_safe_prob(self.geom_warn_clf, X_full)
        overlap_violation = overlap_proxy_violation(X_full)
        eff_obj = np.clip(eff_raw, 0.45, 0.84)
        out["F"] = np.column_stack([
            -eff_obj + 0.12 * unc + OVERLAP_PENALTY_COEFF * overlap_violation + GEOM_WARN_PENALTY_COEFF * (1.0 - p_geom_safe),
            -pr_raw  + 0.12 * unc + OVERLAP_PENALTY_COEFF * overlap_violation + GEOM_WARN_PENALTY_COEFF * (1.0 - p_geom_safe),
        ])
        g_basic = np.column_stack([
            BOUNDARY_FLOW_G_S - mf,
            eff_raw - self.eff_max_phys,
            self.eff_min_phys - eff_raw,
            self.cfg.feasible_prob_threshold_opt - p_feas,
            self.cfg.geom_safe_prob_threshold_opt - p_geom_safe,
        ])

        # 几何规则约束
        g_geom = geometry_rule_violations(X_full)

        out["G"] = np.column_stack([g_basic, g_geom])


# =============================================================================
# 选点策略
# =============================================================================
# =============================================================================
# 选点策略：2 个 EHVI + 1 个最大不确定性 + 1 个欠采样 nBl/空间填充
# =============================================================================
def select_candidates_diverse(
    acq_X: np.ndarray,
    ehvi_vals: np.ndarray,
    scaler_X,
    acq_info: dict | None = None,
    X_pool_raw: np.ndarray | None = None,
    X_failed_raw: np.ndarray | list | None = None,
    n_pick: int = 4,
    min_dist_norm: float = 0.08,
    n_ehvi: int = 2,
):
    """
    固定配额批量选点：
      1) 2 个 Pareto/EHVI 点；
      2) 1 个 MC-Dropout 最大不确定性点；
      3) 1 个训练池中欠采样 nBl 类别内的空间填充点。

    EHVI 使用保守开发掩码；不确定性与覆盖槽位使用仅含显式规则和
    已确认失败距离的探索掩码，使它们能够纠正分类器/代理模型误判。
    所有策略共享批内最小距离约束。
    """
    X_cand = snap_discrete_vars(np.asarray(acq_X, dtype=float))
    X_cand_norm = scaler_X.transform(X_cand)
    n_candidates = len(X_cand)
    if n_candidates == 0 or n_pick <= 0:
        return [], []

    valid_mask = np.ones(n_candidates, dtype=bool)
    exploration_mask = None
    exploration_support = np.ones(n_candidates, dtype=float)
    pred_std_norm = np.zeros((n_candidates, len(SURROGATE_OUTPUT_NAMES)))
    if acq_info is not None:
        if "valid_mask" in acq_info:
            valid_mask = np.asarray(acq_info["valid_mask"], dtype=bool).copy()
        if "exploration_mask" in acq_info:
            exploration_mask = np.asarray(
                acq_info["exploration_mask"], dtype=bool
            ).copy()
        if "pred_std_norm" in acq_info:
            pred_std_norm = np.asarray(acq_info["pred_std_norm"], dtype=float)
        if "p_feas" in acq_info and "p_geom_safe" in acq_info:
            p_feas = np.asarray(acq_info["p_feas"], dtype=float)
            p_geom_safe = np.asarray(acq_info["p_geom_safe"], dtype=float)
            if p_feas.shape != (n_candidates,) or p_geom_safe.shape != (n_candidates,):
                raise ValueError(
                    "acq_info feasibility probabilities must match the candidate count."
                )
            exploration_support = np.sqrt(
                np.clip(p_feas, 0.0, 1.0)
                * np.clip(p_geom_safe, 0.0, 1.0)
            )
            exploration_support = np.nan_to_num(
                exploration_support, nan=0.0, posinf=0.0, neginf=0.0
            )
    if valid_mask.shape != (n_candidates,):
        raise ValueError("acq_info['valid_mask'] must match the candidate count.")
    if exploration_mask is None:
        exploration_mask = valid_mask.copy()
    if exploration_mask.shape != (n_candidates,):
        raise ValueError(
            "acq_info['exploration_mask'] must match the candidate count."
        )
    if pred_std_norm.shape[0] != n_candidates:
        raise ValueError("acq_info['pred_std_norm'] must match the candidate count.")

    X_pool = (
        snap_discrete_vars(np.asarray(X_pool_raw, dtype=float))
        if X_pool_raw is not None and len(X_pool_raw) > 0
        else np.empty((0, X_cand.shape[1]), dtype=float)
    )
    X_pool_norm = (
        scaler_X.transform(X_pool)
        if len(X_pool) > 0
        else np.empty((0, X_cand.shape[1]), dtype=float)
    )
    X_failed = (
        snap_discrete_vars(np.asarray(X_failed_raw, dtype=float))
        if X_failed_raw is not None and len(X_failed_raw) > 0
        else np.empty((0, X_cand.shape[1]), dtype=float)
    )

    selected = []
    labels = []
    selected_idx = []
    exploitation_available = valid_mask.copy()
    exploration_available = exploration_mask.copy()

    # 不重复查询训练池中已有的设计点。
    if len(X_pool_norm) > 0:
        not_in_training_pool = (
            calc_distance_to_set(X_cand_norm, X_pool_norm) > 1e-10
        )
        exploitation_available &= not_in_training_pool
        exploration_available &= not_in_training_pool

    def refresh_batch_distance():
        if not selected_idx:
            return
        d_selected = calc_distance_to_set(
            X_cand_norm, X_cand_norm[np.asarray(selected_idx, dtype=int)]
        )
        too_close = d_selected < min_dist_norm
        exploitation_available[too_close] = False
        exploration_available[too_close] = False

    def add_candidate(idx: int, label: str):
        selected_idx.append(int(idx))
        selected.append(X_cand[idx].copy())
        labels.append(label)
        exploitation_available[idx] = False
        exploration_available[idx] = False
        refresh_batch_distance()

    # 1) Pareto exploitation: exactly two EHVI slots when possible.
    ehvi_scores = np.asarray(ehvi_vals, dtype=float)
    for slot in range(min(n_ehvi, n_pick)):
        # Zero-EHVI candidates are still Pareto-directed samples from the
        # acquisition pool. Keeping them eligible preserves the fixed 2-slot
        # exploitation quota when the current front yields no positive HVI.
        eligible = (
            exploitation_available
            & np.isfinite(ehvi_scores)
            & (ehvi_scores >= 0.0)
        )
        if not np.any(eligible):
            break
        idx = int(np.argmax(np.where(eligible, ehvi_scores, -np.inf)))
        add_candidate(
            idx,
            f"Pareto/EHVI#{slot + 1} (ehvi={ehvi_scores[idx]:.5f})",
        )

    # 2) Exploration: largest normalized MC-Dropout uncertainty.
    if len(selected) < min(n_pick, n_ehvi + 1):
        uncertainty = np.nanmean(pred_std_norm, axis=1)
        # Learned safety remains a soft preference, never a veto. The 0.25
        # floor preserves counterexample collection in regions the classifier
        # currently considers risky.
        support_weight = 0.25 + 0.75 * exploration_support
        uncertainty_score = uncertainty * support_weight
        eligible = exploration_available & np.isfinite(uncertainty_score)
        if np.any(eligible):
            idx = int(
                np.argmax(np.where(eligible, uncertainty_score, -np.inf))
            )
            add_candidate(
                idx,
                f"最大不确定性 (u_norm={uncertainty[idx]:.5f}, "
                f"safety_support={exploration_support[idx]:.3f})",
            )

    # 3) Coverage: least represented blade-count class, then maximin fill.
    if len(selected) < min(n_pick, n_ehvi + 2):
        nbl_idx = VAR_NAMES.index("nBl")
        allowed_nbl = np.arange(
            int(np.ceil(L_BOUNDS[nbl_idx])),
            int(np.floor(U_BOUNDS[nbl_idx])) + 1,
        )
        pool_nbl = (
            np.rint(X_pool[:, nbl_idx]).astype(int)
            if len(X_pool) > 0
            else np.empty(0, dtype=int)
        )
        failed_nbl = (
            np.rint(X_failed[:, nbl_idx]).astype(int)
            if len(X_failed) > 0
            else np.empty(0, dtype=int)
        )
        success_counts = {
            int(nbl): int(np.count_nonzero(pool_nbl == nbl))
            for nbl in allowed_nbl
        }
        failure_counts = {
            int(nbl): int(np.count_nonzero(failed_nbl == nbl))
            for nbl in allowed_nbl
        }
        evaluated_counts = {
            int(nbl): success_counts[int(nbl)] + failure_counts[int(nbl)]
            for nbl in allowed_nbl
        }
        candidate_nbl = np.rint(X_cand[:, nbl_idx]).astype(int)
        target_nbl = None
        eligible = np.zeros(n_candidates, dtype=bool)
        for nbl in sorted(
            evaluated_counts,
            key=lambda value: (
                evaluated_counts[value],
                success_counts[value],
                value,
            ),
        ):
            eligible_for_nbl = exploration_available & (candidate_nbl == nbl)
            if np.any(eligible_for_nbl):
                target_nbl = nbl
                eligible = eligible_for_nbl
                break

        if target_nbl is not None:
            same_nbl_pool = X_pool_norm[pool_nbl == target_nbl]
            references = same_nbl_pool
            if len(references) == 0:
                references = X_pool_norm
            if selected_idx:
                selected_norm = X_cand_norm[np.asarray(selected_idx, dtype=int)]
                references = (
                    np.vstack([references, selected_norm])
                    if len(references) > 0
                    else selected_norm
                )
            fill_distance = (
                calc_distance_to_set(X_cand_norm, references)
                if len(references) > 0
                else np.ones(n_candidates)
            )
            fill_score = fill_distance * (
                0.25 + 0.75 * exploration_support
            )
            idx = int(np.argmax(np.where(eligible, fill_score, -np.inf)))
            add_candidate(
                idx,
                f"欠采样nBl/空间填充 (nBl={target_nbl}, "
                f"success={success_counts[target_nbl]}, "
                f"failed={failure_counts[target_nbl]}, "
                f"evaluated={evaluated_counts[target_nbl]}, "
                f"d={fill_distance[idx]:.4f}, "
                f"safety_support={exploration_support[idx]:.3f})",
            )

    # Any unavailable quota is filled by a valid maximin point.
    while len(selected) < n_pick and np.any(exploration_available):
        references = X_pool_norm
        if selected_idx:
            selected_norm = X_cand_norm[np.asarray(selected_idx, dtype=int)]
            references = (
                np.vstack([references, selected_norm])
                if len(references) > 0
                else selected_norm
            )
        fill_distance = (
            calc_distance_to_set(X_cand_norm, references)
            if len(references) > 0
            else np.ones(n_candidates)
        )
        fill_score = fill_distance * (0.25 + 0.75 * exploration_support)
        idx = int(
            np.argmax(
                np.where(exploration_available, fill_score, -np.inf)
            )
        )
        add_candidate(idx, f"有效候选空间回填 (d={fill_distance[idx]:.4f})")

    # 不足时随机补充
    attempts = 0
    while len(selected) < n_pick and attempts < 200:
        attempts += 1
        sampler = qmc.LatinHypercube(d=len(VAR_NAMES), seed=None)
        x_rand  = snap_discrete_vars(
            qmc.scale(sampler.random(1), L_BOUNDS, U_BOUNDS)
        )[0]
        x_rand = pin_optimization_operating_point(x_rand)[0]

        # 即使在随机补充阶段，也尽量不要把明显违反几何规则的点送去 CFD。
        if np.any(geometry_rule_violations(x_rand[None, :]) > 0.0):
            continue
        x_rand_norm = scaler_X.transform(x_rand[None, :])
        if len(X_pool_norm) > 0:
            if calc_distance_to_set(x_rand_norm, X_pool_norm)[0] <= 1e-10:
                continue
        if selected:
            selected_norm = scaler_X.transform(np.asarray(selected))
            if calc_distance_to_set(x_rand_norm, selected_norm)[0] < min_dist_norm:
                continue

        selected.append(x_rand)
        labels.append("几何安全随机回填")

    return selected[:n_pick], labels[:n_pick]


def _load_nsga2_training_dataframe(use_pool_checkpoint: bool = False) -> pd.DataFrame:
    """
    NSGA-II-only mode defaults to TRAINING_CSV so an LHS/DOE baseline is not
    silently contaminated by active-learning checkpoint samples.
    """
    if use_pool_checkpoint and os.path.exists(POOL_CHECKPOINT_CSV):
        print(f"[NSGA-II] 使用主动学习训练池 checkpoint: {POOL_CHECKPOINT_CSV}")
        return load_and_clean_data(POOL_CHECKPOINT_CSV)

    print(f"[NSGA-II] 使用训练数据 CSV: {TRAINING_CSV}")
    return load_and_clean_data(TRAINING_CSV)


def run_nsga2_only_from_lhs(
    output_csv: str = "nsga2_surrogate_pareto.csv",
    summary_json: str | None = None,
    use_pool_checkpoint: bool = False,
) -> dict:
    """
    Continue the LHS/DOE workflow into surrogate-assisted NSGA-II optimization.

    This path intentionally stops before EHVI acquisition and CFD execution. It
    trains the surrogate on the current data, runs NSGA-II on that surrogate, and
    writes the predicted Pareto set for later engineering review or CFD replay.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    if summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(summary_json)), exist_ok=True)

    df = _load_nsga2_training_dataframe(use_pool_checkpoint=use_pool_checkpoint)
    X_pool, X_test_fixed, Y_pool, Y_test_fixed, W_pool, W_test_fixed = split_with_fixed_testset(df)
    X_pool = snap_discrete_vars(X_pool)
    X_test_fixed = snap_discrete_vars(X_test_fixed)

    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    X_pool_norm = scaler_X.fit_transform(X_pool)
    Y_pool_surr = Y_pool[:, SURROGATE_OUTPUT_IDX]
    Y_test_fixed_surr = Y_test_fixed[:, SURROGATE_OUTPUT_IDX]
    Y_pool_norm = scaler_Y.fit_transform(Y_pool_surr)

    stratify_labels = W_pool if len(np.unique(W_pool)) > 1 else None
    train_idx, val_idx = train_test_split(
        np.arange(len(X_pool)),
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels,
    )
    X_tr, X_val = X_pool_norm[train_idx], X_pool_norm[val_idx]
    Y_tr, Y_val = Y_pool_norm[train_idx], Y_pool_norm[val_idx]
    W_tr, W_val = W_pool[train_idx], W_pool[val_idx]
    nbl_idx = VAR_NAMES.index("nBl")
    train_sample_weights, weight_diagnostics = compute_density_nbl_sample_weights(
        X_tr,
        X_pool[train_idx, nbl_idx],
        k_neighbors=CFG.density_k_neighbors,
        min_weight=CFG.sample_weight_min,
        max_weight=CFG.sample_weight_max,
    )

    print(f"[NSGA-II] 训练 surrogate | 训练池: {len(X_pool)} | 固定测试集: {len(X_test_fixed)}")
    reg_model, reg_hist = train_regressor(
        X_tr,
        Y_tr,
        W_tr,
        X_val,
        Y_val,
        W_val,
        sample_weights_train=train_sample_weights,
        random_seed=1000,
    )
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_Y, SCALER_Y_PATH)

    X_test_norm = scaler_X.transform(X_test_fixed)
    Y_test_pred = deterministic_predict(reg_model, X_test_norm, scaler_Y)
    fixed_metrics = regression_metrics(
        Y_test_fixed_surr,
        Y_test_pred,
        prefix="fixed_test_",
    )
    mse_eff = fixed_metrics["fixed_test_mse_eff"]
    mse_pr = fixed_metrics["fixed_test_mse_pr"]
    mse_mf = fixed_metrics["fixed_test_mse_mf"]
    print(
        "[NSGA-II] 固定测试误差 "
        f"Eff RMSE={fixed_metrics['fixed_test_rmse_eff']:.6f}, "
        f"MAE={fixed_metrics['fixed_test_mae_eff']:.6f}, "
        f"R²={fixed_metrics['fixed_test_r2_eff']:.4f} | "
        f"PR RMSE={fixed_metrics['fixed_test_rmse_pr']:.6f}, "
        f"MF RMSE={fixed_metrics['fixed_test_rmse_mf']:.6f}"
    )

    failed_points = []
    geometry_failures, operating_failures, _ = (
        load_recorded_failure_datasets()
    )
    if len(geometry_failures) > 0:
        failed_points.extend(expand_geometry_decisions(geometry_failures).tolist())
    if len(operating_failures) > 0:
        failed_points.extend(operating_failures.tolist())
    feas_clf = train_feasibility_classifier(
        X_pool,
        operating_failures,
    )

    X_geom_warn, y_geom_warn = build_geometry_classifier_dataset(
        X_pool,
        geometry_failures,
    )
    geom_warn_clf = train_geometry_feasibility_classifier(X_geom_warn, y_geom_warn)
    if geom_warn_clf is not None:
        joblib.dump(geom_warn_clf, GEOM_FEAS_CLF_PATH)

    print(f"[NSGA-II] 开始优化 | pop_size={CFG.pop_size} | n_gen={CFG.n_gen}")
    problem = CompressorMOOProblem(
        reg_model=reg_model,
        feas_clf=feas_clf,
        geom_warn_clf=geom_warn_clf,
        scaler_X=scaler_X,
        scaler_Y=scaler_Y,
        X_pool_raw=X_pool,
        Y_pool_raw=Y_pool,
        cfg=CFG,
    )
    algorithm = NSGA2(pop_size=CFG.pop_size, eliminate_duplicates=True)
    res = minimize(problem, algorithm, ("n_gen", CFG.n_gen), verbose=False)

    pareto_X, pareto_Y, surrogate_hv = extract_surrogate_front_and_hv(
        res=res,
        reg_model=reg_model,
        scaler_X=scaler_X,
        scaler_Y=scaler_Y,
        ref_eff=TRUE_HV_REF_EFF,
        ref_pr=TRUE_HV_REF_PR,
    )
    if pareto_X is None or pareto_Y is None or len(pareto_X) == 0:
        raise ValueError("NSGA-II 未能生成可用的 surrogate Pareto 前沿。")

    pareto_X_norm = scaler_X.transform(pareto_X)
    pred_mean, pred_std_norm = mc_dropout_predict(reg_model, pareto_X_norm, scaler_Y, n_samples=CFG.mc_samples)
    out = pd.DataFrame(pareto_X, columns=VAR_NAMES)
    out.insert(0, "front_index", np.arange(1, len(out) + 1))
    out["pred_Efficiency"] = pred_mean[:, 0]
    out[f"pred_{TARGET_PR_NAME}"] = pred_mean[:, 1]
    out["pred_MassFlow"] = pred_mean[:, 2]
    out["uncertainty_norm"] = pred_std_norm[:, :2].mean(axis=1)
    out["surrogate_hv"] = surrogate_hv
    out.to_csv(output_csv, index=False)
    write_performance_data_metadata(output_csv)

    summary = {
        "mode": "nsga2_only",
        "training_csv": TRAINING_CSV,
        "used_pool_checkpoint": bool(use_pool_checkpoint and os.path.exists(POOL_CHECKPOINT_CSV)),
        "train_samples": int(len(X_pool)),
        "test_samples": int(len(X_test_fixed)),
        "front_size": int(len(out)),
        "surrogate_hv": float(surrogate_hv),
        "mse_eff": float(mse_eff),
        "mse_pr": float(mse_pr),
        "mse_mf": float(mse_mf),
        "epochs": int(len(reg_hist)),
        "output_csv": output_csv,
        "fixed_outlet_static_pressure_pa": float(OPTIMIZATION_P_OUT),
        "geometry_decision_variables": list(GEOMETRY_VAR_NAMES),
        "total_pressure_definition": TOTAL_PRESSURE_DEFINITION,
        "pressure_ratio_constraints": "none",
        **fixed_metrics,
        **weight_diagnostics,
    }
    if summary_json:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[NSGA-II] surrogate Pareto 已保存: {output_csv}")
    if summary_json:
        print(f"[NSGA-II] 摘要已保存: {summary_json}")
    return summary

# =============================================================================
# 主流程
# =============================================================================
def main_multiobjective_active_learning(max_al_iters: int | None = None):
    # -------------------------------------------------------------------------
    # 0. 读取并清洗数据
    # -------------------------------------------------------------------------
    df = load_and_clean_data(TRAINING_CSV)

    X_all = df[VAR_NAMES].values.astype(float)
    Y_all = df[ALL_OUTPUT_NAMES].values.astype(float)
    W_all = df['is_boundary'].values.astype(float)

    n_initial = len(df)

   # 固定测试集；训练池始终来自当前最新 CSV
    X_pool, X_test_fixed, Y_pool, Y_test_fixed, W_pool, W_test_fixed = split_with_fixed_testset(df)
    print(f"[断点续跑] 训练池 checkpoint 路径: {os.path.abspath(POOL_CHECKPOINT_CSV)}")
    print(f"[断点续跑] checkpoint 文件存在: {'yes' if os.path.exists(POOL_CHECKPOINT_CSV) else 'no'}")

    pool_checkpoint_df = load_pool_checkpoint()
    if pool_checkpoint_df is not None:
        X_pool = pool_checkpoint_df[VAR_NAMES].values.astype(float)
        Y_pool = pool_checkpoint_df[ALL_OUTPUT_NAMES].values.astype(float)
        W_pool = pool_checkpoint_df['is_boundary'].values.astype(float)
        print(f"[断点续跑] 已从训练池 checkpoint 恢复样本: {len(X_pool)}")
    else:
        print(f"[断点续跑] 未找到可用训练池 checkpoint，继续使用 TRAINING_CSV 划分后的训练池: {len(X_pool)}")

    # 失败点池：主动学习中动态累积
    failed_points = []
    recorded_geometry_failures, recorded_operating_failures, _ = (
        load_recorded_failure_datasets()
    )
    if len(recorded_geometry_failures) > 0:
        failed_points.extend(
            expand_geometry_decisions(recorded_geometry_failures).tolist()
        )
    if len(recorded_operating_failures) > 0:
        failed_points.extend(recorded_operating_failures.tolist())

    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    start_iter = get_resume_iter()
    effective_max_al_iters = CFG.max_al_iters if max_al_iters is None else int(max_al_iters)

    hv_history = read_optional_csv(
        HV_CSV_PATH,
        HV_HISTORY_COLUMNS,
    ).to_dict('records')

    total_attempts = 0
    total_success = 0
    resumed_in_progress_iter = None
    if os.path.exists(CHECKPOINT_META_PATH):
        try:
            with open(CHECKPOINT_META_PATH, "r", encoding="utf-8") as f:
                checkpoint_meta = json.load(f)
            if (
                int(checkpoint_meta.get("performance_data_schema_version", 0))
                == PERFORMANCE_DATA_SCHEMA_VERSION
                and checkpoint_meta.get("total_pressure_definition")
                == TOTAL_PRESSURE_DEFINITION
            ):
                total_attempts = int(checkpoint_meta.get("total_attempts", 0))
                total_success = int(checkpoint_meta.get("total_success", 0))
                resumed_in_progress_iter = checkpoint_meta.get("in_progress_iter")
                if int(checkpoint_meta.get("hv_policy_version", 1)) != HV_POLICY_VERSION:
                    print(
                        "[断点续跑] HV 口径已升级为 v2：软 overlap 代理不再"
                        "否决真实 CFD 点；新旧轮次的 HV 跳变不应解释为模型收敛变化。"
                    )
            else:
                print("[断点续跑] 旧总压定义的 checkpoint 计数已忽略。")
        except Exception as e:
            print(f"[断点续跑] 读取 checkpoint 元信息中的计数失败，将从 0 开始累计: {e}")

    checkpoint_attempts = total_attempts
    total_attempts, total_success = recover_query_counters(
        AL_QUERY_VALIDATION_CSV,
        total_attempts=total_attempts,
        total_success=total_success,
    )
    if total_attempts > checkpoint_attempts:
        print(
            f"[断点续跑] 查询日志比 checkpoint 更新，尝试编号恢复为 "
            f"A{total_attempts:05d}，下一编号将从 A{total_attempts + 1:05d} 开始。"
        )

    print(f"[断点续跑] checkpoint 元信息路径: {os.path.abspath(CHECKPOINT_META_PATH)}")
    print(f"[断点续跑] 已完成轮次: {start_iter} | 本次将运行到总轮次: {effective_max_al_iters}")
    if resumed_in_progress_iter is not None:
        print(f"[断点续跑] 检测到上次可能中断于第 {int(resumed_in_progress_iter)} 轮进行中，本次将基于已落盘训练池重新开始该轮。")
    true_front_Y = None   # 上一轮真实 Pareto 前沿 Y
    true_front_X = None   # 上一轮真实 Pareto 前沿 X（用于局部采样）

    if len(X_pool) > 0:
        _, true_front_Y, true_front_X = compute_true_cumulative_hv(
            X_pool=X_pool,
            Y_pool=Y_pool,
            W_pool=W_pool,
            geom_warn_clf=None,
            geom_safe_threshold=None,
            ref_eff=TRUE_HV_REF_EFF,
            ref_pr=TRUE_HV_REF_PR
        )

    print(f"[初始化] 训练池: {len(X_pool)} | 固定测试集: {len(X_test_fixed)} | 目标压比字段: {TARGET_PR_NAME}")

    # =========================================================================
    # 主动学习循环
    # =========================================================================
    for al_iter in range(start_iter, effective_max_al_iters):
        print("\n" + "=" * 70)
        print(f"[主动学习] 第 {al_iter + 1}/{effective_max_al_iters} 轮 | 当前训练池: {len(X_pool)}")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # 1. 预处理
        # ---------------------------------------------------------------------
        X_pool = snap_discrete_vars(X_pool)
        X_test_fixed = snap_discrete_vars(X_test_fixed)

        X_pool_norm = scaler_X.fit_transform(X_pool)
        Y_pool_surr = Y_pool[:, SURROGATE_OUTPUT_IDX]
        Y_test_fixed_surr = Y_test_fixed[:, SURROGATE_OUTPUT_IDX]
        Y_pool_norm = scaler_Y.fit_transform(Y_pool_surr)

        # 每轮内部验证划分
        stratify_labels = W_pool if len(np.unique(W_pool)) > 1 else None
        train_idx, val_idx = train_test_split(
            np.arange(len(X_pool)),
            test_size=0.2,
            random_state=42 + al_iter,
            stratify=stratify_labels
        )
        X_tr, X_val = X_pool_norm[train_idx], X_pool_norm[val_idx]
        Y_tr, Y_val = Y_pool_norm[train_idx], Y_pool_norm[val_idx]
        W_tr, W_val = W_pool[train_idx], W_pool[val_idx]
        nbl_idx = VAR_NAMES.index("nBl")
        train_sample_weights, weight_diagnostics = compute_density_nbl_sample_weights(
            X_tr,
            X_pool[train_idx, nbl_idx],
            k_neighbors=CFG.density_k_neighbors,
            min_weight=CFG.sample_weight_min,
            max_weight=CFG.sample_weight_max,
        )

        # ---------------------------------------------------------------------
        # 2. 训练性能回归 surrogate
        # ---------------------------------------------------------------------
        reg_model, reg_hist = train_regressor(
            X_tr,
            Y_tr,
            W_tr,
            X_val,
            Y_val,
            W_val,
            sample_weights_train=train_sample_weights,
            random_seed=1000 + al_iter,
        )
        joblib.dump(scaler_X, SCALER_X_PATH)
        joblib.dump(scaler_Y, SCALER_Y_PATH)

        # 固定测试集评估
        X_test_norm = scaler_X.transform(X_test_fixed)
        Y_test_pred = deterministic_predict(reg_model, X_test_norm, scaler_Y)

        fixed_metrics = regression_metrics(
            Y_test_fixed_surr,
            Y_test_pred,
            prefix="fixed_test_",
        )
        mse_eff = fixed_metrics["fixed_test_mse_eff"]
        mse_pr = fixed_metrics["fixed_test_mse_pr"]
        mse_mf = fixed_metrics["fixed_test_mse_mf"]
        print(
            "  [固定测试集] "
            f"Eff RMSE={fixed_metrics['fixed_test_rmse_eff']:.6f}, "
            f"MAE={fixed_metrics['fixed_test_mae_eff']:.6f}, "
            f"R²={fixed_metrics['fixed_test_r2_eff']:.4f} | "
            f"PR RMSE={fixed_metrics['fixed_test_rmse_pr']:.6f}, "
            f"MAE={fixed_metrics['fixed_test_mae_pr']:.6f}, "
            f"R²={fixed_metrics['fixed_test_r2_pr']:.4f} | "
            f"MF RMSE={fixed_metrics['fixed_test_rmse_mf']:.6f}, "
            f"MAE={fixed_metrics['fixed_test_mae_mf']:.6f}, "
            f"R²={fixed_metrics['fixed_test_r2_mf']:.4f}"
        )

        fixed_prediction_rows = []
        for test_idx, (x_test, y_true, y_pred) in enumerate(
            zip(X_test_fixed, Y_test_fixed_surr, Y_test_pred)
        ):
            row = {
                "iter": int(al_iter + 1),
                "test_index": int(test_idx),
                "recorded_at_utc": utc_timestamp(),
            }
            row.update({name: float(value) for name, value in zip(VAR_NAMES, x_test)})
            for output_idx, suffix in enumerate(("eff", "pr", "mf")):
                row[f"true_{suffix}"] = float(y_true[output_idx])
                row[f"pred_{suffix}"] = float(y_pred[output_idx])
                row[f"error_{suffix}"] = float(y_pred[output_idx] - y_true[output_idx])
                row[f"abs_error_{suffix}"] = abs(row[f"error_{suffix}"])
            fixed_prediction_rows.append(row)
        upsert_csv_records(
            FIXED_TEST_PREDICTIONS_CSV,
            fixed_prediction_rows,
            key_columns=["iter", "test_index"],
        )

        print(f"  [K-fold] 开始 {min(CFG.cv_folds, len(X_pool))}-fold 交叉验证...")
        cv_fold_rows, cv_summary = run_kfold_surrogate_validation(
            X_raw=X_pool,
            Y_surr=Y_pool_surr,
            W_boundary=W_pool,
            al_iter=al_iter + 1,
            cfg=CFG,
        )
        if cv_fold_rows:
            upsert_csv_records(
                CV_FOLD_METRICS_CSV,
                cv_fold_rows,
                key_columns=["iter", "fold"],
            )
            print(
                "  [K-fold] "
                f"Eff RMSE={cv_summary['cv_mean_rmse_eff']:.6f}±"
                f"{cv_summary['cv_std_rmse_eff']:.6f} | "
                f"PR RMSE={cv_summary['cv_mean_rmse_pr']:.6f}±"
                f"{cv_summary['cv_std_rmse_pr']:.6f} | "
                f"MF RMSE={cv_summary['cv_mean_rmse_mf']:.6f}±"
                f"{cv_summary['cv_std_rmse_mf']:.6f}"
            )

        validation_row = {
            "iter": int(al_iter + 1),
            "train_pool_samples_before_cfd": int(len(X_pool)),
            "fixed_test_samples": int(len(X_test_fixed)),
            "training_epochs": int(len(reg_hist)),
            "recorded_at_utc": utc_timestamp(),
            **fixed_metrics,
            **cv_summary,
            **weight_diagnostics,
        }
        upsert_csv_records(
            SURROGATE_METRICS_CSV,
            validation_row,
            key_columns=["iter"],
        )

        # ---------------------------------------------------------------------
        # 3. 训练可行性分类器
        # ---------------------------------------------------------------------
        geometry_failures, operating_failures, _ = load_recorded_failure_datasets()
        failed_points = []
        if len(geometry_failures) > 0:
            failed_points.extend(
                expand_geometry_decisions(geometry_failures).tolist()
            )
        if len(operating_failures) > 0:
            failed_points.extend(operating_failures.tolist())
        feas_clf = train_feasibility_classifier(
            X_pool,
            operating_failures,
        )
        if feas_clf is None:
            print("  [可行性分类器] 当前无失败样本，默认全部可行。")
        else:
            print(
                f"  [工况可行性分类器] 已用成功 {len(X_pool)} / "
                f"已确认求解失败 {len(operating_failures)} 样本训练"
                f"（{len(FEASIBILITY_FEATURE_NAMES)}维，不含固定 P_out/nBl）。"
            )

        # ---------------------------------------------------------------------
        # 4. 训练边界分类器
        # ---------------------------------------------------------------------
        boundary_model = train_boundary_classifier(X_pool_norm, W_pool.astype(int))
        if boundary_model is None:
            print("  [边界分类器] 当前类别不足，跳过。")
        else:
            print("  [边界分类器] 已训练。")

        # ---------------------------------------------------------------------
        # 4.5 训练不含 nBl 的几何可生成性分类器（仅使用已确认的 DOE/AL 失败）
        # ---------------------------------------------------------------------
        X_geom_warn, y_geom_warn = build_geometry_classifier_dataset(
            X_pool,
            geometry_failures,
        )
        geom_warn_clf = train_geometry_feasibility_classifier(X_geom_warn, y_geom_warn)
        if geom_warn_clf is None:
            print("  [几何报错分类器] 样本不足或类别不足，跳过。")
        else:
            joblib.dump(geom_warn_clf, GEOM_FEAS_CLF_PATH)
            n_bad = int(y_geom_warn.sum())
            print(
                f"  [几何安全分类器] 已训练。样本={len(y_geom_warn)} | "
                f"警告/失败样本={n_bad}"
                f"（{len(GEOM_WARN_FEATURE_NAMES)}维，不含 P_out/nBl）"
            )

        # X_pool 中的点已经通过 CFD 验证；失败分类器只用于未计算候选点
        # 的风险控制，不能反过来否决真实成功样本或改变真实 Pareto 前沿。
        true_hv_ref_val, true_front_Y, true_front_X = compute_true_cumulative_hv(
            X_pool=X_pool,
            Y_pool=Y_pool,
            W_pool=W_pool,
            geom_warn_clf=None,
            geom_safe_threshold=None,
            ref_eff=TRUE_HV_REF_EFF,
            ref_pr=TRUE_HV_REF_PR
        )
        if np.isfinite(true_hv_ref_val):
            print(f"  [安全前沿] 用于 EHVI 参考的真实 HV = {true_hv_ref_val:.6f}")
        else:
            print("  [安全前沿] 当前无满足几何安全过滤的真实前沿，将退化为冷启动探索。")

        # ---------------------------------------------------------------------
        # 5. NSGA-II 优化
        # ---------------------------------------------------------------------
        problem = CompressorMOOProblem(
            reg_model=reg_model,
            feas_clf=feas_clf,
            geom_warn_clf=geom_warn_clf,
            scaler_X=scaler_X,
            scaler_Y=scaler_Y,
            X_pool_raw=X_pool,
            Y_pool_raw=Y_pool,
            cfg=CFG
        )

        algorithm = NSGA2(pop_size=CFG.pop_size, eliminate_duplicates=True)
        res = minimize(problem, algorithm, ('n_gen', CFG.n_gen), verbose=False)

        pareto_X = None
        pareto_Y = None
        pareto_meta = {}
        surrogate_hv_val = np.nan
        if res.F is not None and len(res.F) > 0:
            pareto_X = expand_geometry_decisions(res.X)
            pareto_Y = -res.F
            pareto_X, pareto_Y, surrogate_hv_val = extract_surrogate_front_and_hv(
                res=res,
                reg_model=reg_model,
                scaler_X=scaler_X,
                scaler_Y=scaler_Y,
                ref_eff=TRUE_HV_REF_EFF,
                ref_pr=TRUE_HV_REF_PR
            )
            if pareto_X is not None and len(pareto_X) > 0:
                print(f"  [辅指标] surrogate HV = {surrogate_hv_val:.6f}")
            else:
                print("  [辅指标] surrogate 前沿为空。")
# ---------------------------------------------------------------------
        # 6. EHVI 采集（替换原 compute_acquisition）
        # ---------------------------------------------------------------------
        acq_X, ehvi_vals, acq_info = compute_ehvi_acquisition(
            reg_model=reg_model,
            feas_clf=feas_clf,
            geom_warn_clf=geom_warn_clf,
            scaler_X=scaler_X,
            scaler_Y=scaler_Y,
            X_pool_raw=X_pool,
            failed_points_raw=failed_points,
            current_pareto_Y=true_front_Y,   # 上一轮的真实前沿
            current_pareto_X=true_front_X,
            ref_eff=TRUE_HV_REF_EFF,
            ref_pr=TRUE_HV_REF_PR,
            cfg=CFG
        )

        # ---------------------------------------------------------------------
        # 7. 多样性贪心选点（替换原 select_candidates_for_cfd）
        # ---------------------------------------------------------------------
        candidates_X, labels = select_candidates_diverse(
            acq_X=acq_X,
            ehvi_vals=ehvi_vals,
            scaler_X=scaler_X,
            acq_info=acq_info,
            X_pool_raw=X_pool,
            X_failed_raw=failed_points,
            n_pick=CFG.n_eval_candidates_per_iter,
            min_dist_norm=CFG.diversity_min_dist,
            n_ehvi=2,
        )
        print("  [选点配额] " + " | ".join(labels))

        if candidates_X:
            selected_array = np.asarray(candidates_X, dtype=float)
            selected_norm = scaler_X.transform(selected_array)
            pre_cfd_mean, pre_cfd_std_norm = mc_dropout_predict(
                reg_model,
                selected_norm,
                scaler_Y,
                n_samples=CFG.mc_samples,
            )
            pre_cfd_std_real = normalized_std_to_real(
                pre_cfd_std_norm,
                scaler_Y,
            )
        else:
            pre_cfd_mean = np.empty((0, len(SURROGATE_OUTPUT_NAMES)))
            pre_cfd_std_norm = np.empty_like(pre_cfd_mean)
            pre_cfd_std_real = np.empty_like(pre_cfd_mean)

        # ---------------------------------------------------------------------
        # 8. CFD 闭环更新
        # ---------------------------------------------------------------------
        for i, (x_cand, label) in enumerate(zip(candidates_X, labels)):
            next_attempt = total_attempts + 1
            run_id = (
                f"AL_Iter{al_iter+1:02d}_P{i+1}_A{next_attempt:05d}"
            )
            print(f"\n  -> [候选 {i+1}/{len(candidates_X)}] {label}")
            total_attempts = next_attempt

            matched = np.where(
                np.all(np.isclose(acq_X, x_cand, rtol=0.0, atol=1e-12), axis=1)
            )[0]
            acq_idx = int(matched[0]) if len(matched) > 0 else None
            query_record = {
                "run_id": run_id,
                "iter": int(al_iter + 1),
                "candidate_slot": int(i + 1),
                "selection_strategy": label,
                "status": "submitted",
                "submitted_at_utc": utc_timestamp(),
                "completed_at_utc": "",
                "acquisition_index": acq_idx,
                "ehvi": (
                    float(ehvi_vals[acq_idx])
                    if acq_idx is not None and np.isfinite(ehvi_vals[acq_idx])
                    else np.nan
                ),
                "pred_feasible_probability": (
                    float(acq_info["p_feas"][acq_idx])
                    if acq_idx is not None and "p_feas" in acq_info
                    else np.nan
                ),
                "pred_geometry_safe_probability": (
                    float(acq_info["p_geom_safe"][acq_idx])
                    if acq_idx is not None and "p_geom_safe" in acq_info
                    else np.nan
                ),
                "pred_eff": float(pre_cfd_mean[i, 0]),
                "pred_pr": float(pre_cfd_mean[i, 1]),
                "pred_mf": float(pre_cfd_mean[i, 2]),
                "uncertainty_norm_eff": float(pre_cfd_std_norm[i, 0]),
                "uncertainty_norm_pr": float(pre_cfd_std_norm[i, 1]),
                "uncertainty_norm_mf": float(pre_cfd_std_norm[i, 2]),
                "uncertainty_norm_mean": float(pre_cfd_std_norm[i].mean()),
                "uncertainty_real_eff": float(pre_cfd_std_real[i, 0]),
                "uncertainty_real_pr": float(pre_cfd_std_real[i, 1]),
                "uncertainty_real_mf": float(pre_cfd_std_real[i, 2]),
            }
            query_record.update(
                {name: float(value) for name, value in zip(VAR_NAMES, x_cand)}
            )
            upsert_csv_records(
                AL_QUERY_VALIDATION_CSV,
                query_record,
                key_columns=["run_id"],
            )

            success = False
            true_y = None
            geometry_summary = None
            failure_info = None
            failure_record = None
            evaluation_attempts = 0
            for evaluation_attempt in range(1, 3):
                evaluation_attempts = evaluation_attempt
                success, true_y, geometry_summary, failure_info = run_single_cfd(
                    x_cand,
                    run_id,
                )
                if success:
                    record_run_outcome(
                        FAILURE_RECORDS_CSV,
                        VAR_NAMES,
                        x_cand,
                        source="active_learning",
                        run_id=run_id,
                        status="succeeded",
                    )
                    break
                failure_record = record_run_outcome(
                    FAILURE_RECORDS_CSV,
                    VAR_NAMES,
                    x_cand,
                    source="active_learning",
                    run_id=run_id,
                    status="failed",
                    failure_stage=(failure_info or {}).get("stage", "infrastructure"),
                    reason=(failure_info or {}).get("reason", ""),
                )
                if bool(failure_record.get("confirmed")):
                    break
                if evaluation_attempt < 2:
                    print("     [失败复核] 将原点重试一次，排除偶发软件或求解故障。")
            print_geometry_summary(geometry_summary)
            query_record["evaluation_attempts"] = int(evaluation_attempts)

            if success:
                total_success += 1
                print(
                    f"     [CFD结果] Eff={true_y[0]*100:.2f}% | "
                    f"PR={true_y[1]:.4f} | Power={true_y[2]:.3f} | MF={true_y[3]:.2f} g/s"
                )

                new_boundary = infer_boundary_from_outputs(true_y)

                X_pool = np.vstack([X_pool, x_cand])
                Y_pool = np.vstack([Y_pool, true_y])
                W_pool = np.append(W_pool, new_boundary)

                true_surr = np.asarray(true_y, dtype=float)[SURROGATE_OUTPUT_IDX]
                query_record.update({
                    "status": "success",
                    "completed_at_utc": utc_timestamp(),
                    "true_eff": float(true_surr[0]),
                    "true_pr": float(true_surr[1]),
                    "true_mf": float(true_surr[2]),
                    "true_power": float(true_y[2]),
                    "is_boundary": float(new_boundary),
                })
                for output_idx, suffix in enumerate(("eff", "pr", "mf")):
                    error = float(pre_cfd_mean[i, output_idx] - true_surr[output_idx])
                    uncertainty = float(pre_cfd_std_real[i, output_idx])
                    query_record[f"error_{suffix}"] = error
                    query_record[f"abs_error_{suffix}"] = abs(error)
                    query_record[f"squared_error_{suffix}"] = error ** 2
                    query_record[f"standardized_abs_error_{suffix}"] = (
                        abs(error) / max(uncertainty, 1e-12)
                    )
                    query_record[f"covered_by_1sigma_{suffix}"] = bool(
                        abs(error) <= uncertainty
                    )
                    query_record[f"covered_by_2sigma_{suffix}"] = bool(
                        abs(error) <= 2.0 * uncertainty
                    )

            else:
                if (
                    failure_record is not None
                    and bool(failure_record.get("confirmed"))
                    and failure_record.get("failure_stage")
                    in {
                        "geometry",
                        "mesh",
                        "geometry_mesh",
                        "fatal_overflow",
                        "blockage",
                        "physical_invalid",
                    }
                ):
                    failed_points.append(x_cand.copy())
                    print(
                        f"     [已确认失败] 已加入距离过滤池，当前累计 "
                        f"{len(failed_points)}"
                    )
                else:
                    print("     [系统性故障] 已记录，但不作为设计不可行标签。")
                query_record.update({
                    "status": "failed",
                    "completed_at_utc": utc_timestamp(),
                    "failure_stage": (failure_info or {}).get(
                        "stage", "infrastructure"
                    ),
                    "failure_reason": (failure_info or {}).get("reason", ""),
                })

            query_record["geometry_summary_json"] = json.dumps(
                geometry_summary or {},
                ensure_ascii=False,
                default=str,
            )
            upsert_csv_records(
                AL_QUERY_VALIDATION_CSV,
                query_record,
                key_columns=["run_id"],
            )

            save_checkpoint(
                al_iter=al_iter,
                X_pool=X_pool,
                Y_pool=Y_pool,
                W_pool=W_pool,
                X_test_fixed=X_test_fixed,
                Y_test_fixed=Y_test_fixed,
                W_test_fixed=W_test_fixed,
                failed_points=failed_points,
                hv_history=hv_history,
                total_attempts=total_attempts,
                total_success=total_success,
                completed_iters=al_iter,
                in_progress_iter=al_iter + 1
            )

        true_hv_val, true_front_Y, true_front_X = compute_true_cumulative_hv(
            X_pool=X_pool,
            Y_pool=Y_pool,
            W_pool=W_pool,
            geom_warn_clf=None,
            geom_safe_threshold=None,
            ref_eff=TRUE_HV_REF_EFF,
            ref_pr=TRUE_HV_REF_PR
        )
        hv_history.append({
            "iter": al_iter + 1,
            "hv_policy_version": HV_POLICY_VERSION,
            "n_samples": len(X_pool),
            "train_samples_before_cfd": validation_row["train_pool_samples_before_cfd"],
            "true_hv": true_hv_val,
            "surrogate_hv": surrogate_hv_val,
            "mse_eff": mse_eff,
            "mse_pr": mse_pr,
            "mse_mf": mse_mf,
            "rmse_eff": fixed_metrics["fixed_test_rmse_eff"],
            "rmse_pr": fixed_metrics["fixed_test_rmse_pr"],
            "rmse_mf": fixed_metrics["fixed_test_rmse_mf"],
            "mae_eff": fixed_metrics["fixed_test_mae_eff"],
            "mae_pr": fixed_metrics["fixed_test_mae_pr"],
            "mae_mf": fixed_metrics["fixed_test_mae_mf"],
            "r2_eff": fixed_metrics["fixed_test_r2_eff"],
            "r2_pr": fixed_metrics["fixed_test_r2_pr"],
            "r2_mf": fixed_metrics["fixed_test_r2_mf"],
            "cv_rmse_eff_mean": cv_summary.get("cv_mean_rmse_eff", np.nan),
            "cv_rmse_pr_mean": cv_summary.get("cv_mean_rmse_pr", np.nan),
            "cv_rmse_mf_mean": cv_summary.get("cv_mean_rmse_mf", np.nan),
            "cv_r2_eff_mean": cv_summary.get("cv_mean_r2_eff", np.nan),
            "cv_r2_pr_mean": cv_summary.get("cv_mean_r2_pr", np.nan),
            "cv_r2_mf_mean": cv_summary.get("cv_mean_r2_mf", np.nan),
        })
        save_checkpoint(
            al_iter=al_iter,
            X_pool=X_pool,
            Y_pool=Y_pool,
            W_pool=W_pool,
            X_test_fixed=X_test_fixed,
            Y_test_fixed=Y_test_fixed,
            W_test_fixed=W_test_fixed,
            failed_points=failed_points,
            hv_history=hv_history,
            total_attempts=total_attempts,
            total_success=total_success,
            completed_iters=al_iter + 1,
            in_progress_iter=None
        )
        print(f"  [主指标] 真实累计 HV = {true_hv_val:.6f}" if np.isfinite(true_hv_val) else "  [主指标] 当前真实可行前沿为空")
        # ---------------------------------------------------------------------
        # 9. 健康检查
        # ---------------------------------------------------------------------
        fail_rate = 1.0 - total_success / max(total_attempts, 1)
        print(f"\n  [健康] 成功率: {total_success}/{total_attempts} = {(1-fail_rate)*100:.1f}% | 失效率: {fail_rate*100:.1f}%")
        if fail_rate > 0.5:
            print("  [警告] 失效率 > 50%，建议继续加强 geometry rules 或 feasibility classifier。")

    # =========================================================================
    # 收尾：HV 历史
    # =========================================================================
    hv_df = write_hv_history(HV_CSV_PATH, hv_history)
    
    print(f"\n[完成] HV 历史已保存: {HV_CSV_PATH}")

    valid_true_hv = hv_df.dropna(subset=['true_hv'])

    if len(valid_true_hv) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 主指标：真实累计 HV
        axes[0].plot(valid_true_hv['iter'], valid_true_hv['true_hv'], 'o-', lw=2, label='True HV')
        if 'surrogate_hv' in hv_df.columns:
            valid_surr = hv_df.dropna(subset=['surrogate_hv'])
            if len(valid_surr) > 0:
                axes[0].plot(valid_surr['iter'], valid_surr['surrogate_hv'], 's--', lw=1.5, label='Surrogate HV')
        axes[0].set_xlabel("AL 轮次")
        axes[0].set_ylabel("HV")
        axes[0].set_title("True HV / Surrogate HV")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    # 真HV vs 样本量
        axes[1].plot(valid_true_hv['n_samples'], valid_true_hv['true_hv'], 'o-', lw=2)
        axes[1].set_xlabel("CFD 样本量")
        axes[1].set_ylabel("True HV")
        axes[1].set_title("True HV vs 样本量")
        axes[1].grid(True, alpha=0.3)

    # surrogate 误差
        axes[2].plot(hv_df['iter'], hv_df['mse_eff'], label='MSE Eff')
        axes[2].plot(hv_df['iter'], hv_df['mse_pr'], label='MSE PR')
        axes[2].plot(hv_df['iter'], hv_df['mse_mf'], label='MSE MF')
        axes[2].set_xlabel("AL 轮次")
        axes[2].set_ylabel("MSE")
        axes[2].set_title("Surrogate 测试误差")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(HV_PLOT_PATH, dpi=150)
        print(f"[完成] HV 曲线已保存: {HV_PLOT_PATH}")
        if ENABLE_INTERACTIVE_PLOT:
            plt.show()
        else:
            plt.close(fig)

    print(
        f"\n[汇总] 初始样本: {n_initial} | "
        f"最终成功样本: {len(X_pool)} | "
        f"失败样本累计: {len(failed_points)}"
    )


if __name__ == "__main__":
    runtime_args = parse_runtime_args()
    if runtime_args.nsga2_only:
        summary_path = os.path.splitext(runtime_args.nsga2_output_csv)[0] + "_summary.json"
        run_nsga2_only_from_lhs(
            output_csv=runtime_args.nsga2_output_csv,
            summary_json=summary_path,
            use_pool_checkpoint=runtime_args.nsga2_use_pool_checkpoint,
        )
        raise SystemExit(0)

    checkpoint_completed = get_resume_iter()
    if runtime_args.max_al_iters is not None:
        target_max_iters = runtime_args.max_al_iters
    else:
        target_max_iters = max(CFG.max_al_iters, checkpoint_completed + max(0, runtime_args.additional_iters))
    print(f"[运行参数] checkpoint={checkpoint_completed} | target_max_iters={target_max_iters}")
    main_multiobjective_active_learning(max_al_iters=target_max_iters)
