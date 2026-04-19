import os
import json
import argparse
import subprocess
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
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
from design_variables import lower_bounds, load_variable_specs, upper_bounds, variable_names

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
TRUE_HV_REF_PR  = 1.8
DESIGN_VARIABLES_PATH = _env_or_default("IMPELLER_DESIGN_VARIABLES_PATH", "design_variables.json")
_VARIABLE_SPECS = load_variable_specs(DESIGN_VARIABLES_PATH)
VAR_NAMES = variable_names(_VARIABLE_SPECS)
L_BOUNDS = lower_bounds(_VARIABLE_SPECS)
U_BOUNDS = upper_bounds(_VARIABLE_SPECS)

# 基于当前设计变量范围和叶轮几何经验的保守几何阈值。
# beta1hb - beta1sb 的理论上限约为 64 deg，原 62 deg 基本不起筛选作用；
# 这里先收紧到 52 deg，避免像 45 deg 那样一下子把大半搜索空间全部切掉，
# 同时比原 62 deg 更能抑制明显过高的前缘切向 sweep。
MAX_LE_SWEEP_DIFF = 52.0

# 相邻叶片 overlap 的代理约束：
# nBl 越少，节距角 360/nBl 越大，同样的后掠越容易造成 overlap 不足，
# 因此对 rake_te_s 的允许下限做成随 nBl 自适应变化。
MIN_RAKE_TE_S_BY_NBL = {
    9: -18.0,
    10: -19.0,
    11: -20.0,
    12: -21.0,
}

# overlap 先作为软惩罚而不是硬约束处理，避免直接把大量已算出的 DOE/训练点全部判死。
OVERLAP_PENALTY_COEFF = 0.035
OVERLAP_EHVI_DECAY_DEG = 3.0
GEOM_WARN_PENALTY_COEFF = 0.20
GEOM_WARN_RUNS_DIR_FALLBACK = "ActiveLearning_Runs"
GEOM_WARN_FEATURE_NAMES = [name for name in VAR_NAMES if name != "P_out"]

PS_SCRIPT_PATH  = _env_or_default("IMPELLER_PS_SCRIPT_PATH", r"F:\optimazition\Run-GeometryMeshing.ps1")
AL_WORKING_BASE = _env_or_default("IMPELLER_AL_WORKING_BASE", r"F:\optimazition\ActiveLearning_Runs")
TRAINING_CSV    = _env_or_default("IMPELLER_TRAINING_CSV", "Compressor_Training_Data.csv")
SCALER_X_PATH = _env_or_default("IMPELLER_SCALER_X_PATH", "scaler_X.pkl")
SCALER_Y_PATH = _env_or_default("IMPELLER_SCALER_Y_PATH", "scaler_Y.pkl")
BEST_REG_PATH   = _env_or_default("IMPELLER_BEST_REG_PATH", "best_regressor.pth")
GEOM_WARN_CLF_PATH = _env_or_default("IMPELLER_GEOM_WARN_CLF_PATH", "geometry_warning_clf.pkl")
HV_CSV_PATH     = _env_or_default("IMPELLER_HV_CSV_PATH", "hv_history.csv")
HV_PLOT_PATH    = _env_or_default("IMPELLER_HV_PLOT_PATH", "hv_convergence.png")
FAILED_POINTS_PATH = _env_or_default("IMPELLER_FAILED_POINTS_PATH", "failed_points.npy")#保留失败样本池
TEST_SPLIT_PATH = _env_or_default("IMPELLER_TEST_SPLIT_PATH", "fixed_test_set.npz")#保留固定测试集，避免每轮随机分割导致评估波动
TEST_SET_CSV = _env_or_default("IMPELLER_TEST_SET_CSV", "fixed_test_set.csv")
POOL_CHECKPOINT_CSV = _env_or_default("IMPELLER_POOL_CHECKPOINT_CSV", "al_training_pool_checkpoint.csv")
CHECKPOINT_META_PATH = _env_or_default("IMPELLER_CHECKPOINT_META_PATH", "al_checkpoint_meta.json")
GEOMETRY_SUMMARY_PATH = _env_or_default("IMPELLER_GEOMETRY_SUMMARY_PATH", "geometry_summary.json")
ENABLE_INTERACTIVE_PLOT = _env_or_default("IMPELLER_ENABLE_PLOT", "0").lower() in {"1", "true", "yes"}


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
        globals_dict["GEOM_WARN_FEATURE_NAMES"] = [name for name in globals_dict["VAR_NAMES"] if name != "P_out"]


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
    return parser.parse_args()


#断点需跑的辅助函数
def get_resume_iter(
    checkpoint_meta_path=CHECKPOINT_META_PATH,
    hv_csv_path=HV_CSV_PATH
):
    if os.path.exists(checkpoint_meta_path):
        try:
            with open(checkpoint_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return int(meta.get("completed_iters", 0))
        except Exception as e:
            print(f"[断点续跑] 读取 checkpoint 元信息失败，将回退到 HV 历史: {e}")

    if os.path.exists(hv_csv_path):
        hv_df = pd.read_csv(hv_csv_path)
        if len(hv_df) > 0 and 'iter' in hv_df.columns:
            return int(hv_df['iter'].max())
    return 0


def load_pool_checkpoint(pool_csv=POOL_CHECKPOINT_CSV):
    if not os.path.exists(pool_csv):
        return None

    df_pool = pd.read_csv(pool_csv)
    expected_cols = VAR_NAMES + ALL_OUTPUT_NAMES + ['is_boundary']
    missing_cols = [col for col in expected_cols if col not in df_pool.columns]
    if missing_cols:
        raise ValueError(f"训练池 checkpoint 缺少必要列: {missing_cols}")

    df_pool = df_pool[expected_cols].copy()
    df_pool['nBl'] = np.clip(np.round(df_pool['nBl']), 9, 12).astype(int)
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
    pool_csv=POOL_CHECKPOINT_CSV,
    failed_points_path=FAILED_POINTS_PATH,
    hv_csv_path=HV_CSV_PATH,
    checkpoint_meta_path=CHECKPOINT_META_PATH
):
    df_pool = pd.DataFrame(X_pool, columns=VAR_NAMES)
    for j, col in enumerate(ALL_OUTPUT_NAMES):
        df_pool[col] = Y_pool[:, j]
    df_pool['is_boundary'] = W_pool
    df_pool.to_csv(pool_csv, index=False)

    np.save(failed_points_path, np.array(failed_points, dtype=float))
    pd.DataFrame(hv_history).to_csv(hv_csv_path, index=False)

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
        "fixed_test_set_csv": TEST_SET_CSV
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
    X[:, 11] = np.clip(np.round(X[:, 11]), 9, 12)
    return X


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / (rng + 1e-8)


def calc_distance_to_set(X_norm: np.ndarray, ref_norm: np.ndarray) -> np.ndarray:
    return cdist(X_norm, ref_norm).min(axis=1)


def infer_boundary_from_outputs(y: np.ndarray) -> float:
    eff, pr, power, mf = y
    return float(
        (mf < 3.60) or      # 质量流量不足，接近失速
        (eff < 0.60) or     # 效率过低
        (power < 60.0)      # 功率异常低
    )


def geometry_rule_violations(X: np.ndarray) -> np.ndarray:
    """
    显式几何规则约束。
    返回 G_rule, shape = (N, k), 每列 <= 0 表示满足约束。
    这些规则你需要按工程经验继续细化。
    """
    X = snap_discrete_vars(X)
    d1s, dH, beta1hb, beta1sb, d2, b2, beta2hb, beta2sb, Lz, t, TipClear, nBl, rake_te_s, P_out = X.T

    g1 = 0.070 - (d2 - d1s)                 # d2 至少比 d1s 大 0.010
    g2 = 0.044 - b2                         # b2 
    g3 = t - 0.0035                         # 厚度
    g4 = 0.0007 - TipClear                 # 间隙
    g5 = TipClear - 0.0015                 # 间隙
    g6 = (beta1hb - beta1sb) - MAX_LE_SWEEP_DIFF
    g7 = np.abs(beta2hb - beta2sb) - 13.5   # 出口角差
    g8 = 0.185 - Lz                         # Lz 
    g9 = Lz - 0.215                         # Lz 
    g10 = rake_te_s + 15.0                  # rake_te_s <= -15 -> rake+15 <=0
    g11 = -25.0 - rake_te_s                 # rake_te_s >= -25 -> -25-rake <=0

    return np.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])


def overlap_proxy_violation(X: np.ndarray) -> np.ndarray:
    """
    overlap 风险代理量，>0 表示后掠过大导致相邻叶片 overlap 可能偏低。
    目前只作为软惩罚使用，不直接判为硬不可行。
    """
    X = snap_discrete_vars(X)
    nBl = np.clip(np.round(X[:, 11]), 9, 12).astype(int)
    rake_te_s = X[:, 12]
    min_rake_for_overlap = np.array(
        [MIN_RAKE_TE_S_BY_NBL[int(n_bl)] for n_bl in nBl],
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
    df = pd.read_csv(csv_path)

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
    df['nBl'] = np.clip(np.round(df['nBl']), 9, 12).astype(int)

    # 简单去重
    df = df.drop_duplicates(subset=VAR_NAMES, keep='first').reset_index(drop=True)

    return df
# =============================================================================
# 数据划分：固定测试集
# =============================================================================
def split_with_fixed_testset(df: pd.DataFrame, test_csv=TEST_SET_CSV, test_size=0.15):
    """
    固定测试集：
    - 第一次运行：从当前 df 中划分测试集并保存到 test_csv
    - 后续运行：读取 test_csv，并从当前 df 中按设计变量匹配剔除测试样本
    """
    df = df.copy()

    if os.path.exists(test_csv):
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

    if current_pareto_X is not None and len(current_pareto_X) > 0:
        sigma = (U_BOUNDS - L_BOUNDS) * 0.06   # 各维度范围的 6%
        idx = np.random.choice(len(current_pareto_X), n_local, replace=True)
        X_local = current_pareto_X[idx] + np.random.randn(n_local, len(VAR_NAMES)) * sigma
        X_local = np.clip(X_local, L_BOUNDS, U_BOUNDS)
        X_local = snap_discrete_vars(X_local)
    else:
        sampler2 = qmc.LatinHypercube(d=len(VAR_NAMES), seed=None)
        X_local = snap_discrete_vars(
            qmc.scale(sampler2.random(n_local), L_BOUNDS, U_BOUNDS)
        )

    return np.vstack([X_global, X_local])
def compute_true_cumulative_hv(
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    W_pool: np.ndarray,
    geom_warn_clf=None,
    geom_safe_threshold: float | None = None,
    exclude_overlap_proxy: bool = True,
    ref_eff: float = TRUE_HV_REF_EFF,
    ref_pr: float = TRUE_HV_REF_PR
):
    """
    返回 (hv_val, front_Y, front_X)
    front_Y: (K, 2) 非支配前沿的 [Efficiency, PR]
    front_X: (K, 14) 对应的设计变量，供局部采样使用
    """
    X_pool = snap_discrete_vars(X_pool)
    geom_ok = geometry_safe_mask(
        X_pool,
        geom_warn_clf=geom_warn_clf,
        geom_safe_threshold=geom_safe_threshold,
        exclude_overlap_proxy=exclude_overlap_proxy,
    )

    feas_mask = (
        geom_ok &
        (Y_pool[:, 0] >= 0.45) &
        (Y_pool[:, 1] >= 1.60) &
        (Y_pool[:, 1] <= 2.85) &
        (Y_pool[:, 3] >= 3.60)
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

    X_pf = snap_discrete_vars(np.atleast_2d(res.X))
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
def weighted_regression_loss(pred, target, is_boundary):
    per_sample = ((pred - target) ** 2).mean(dim=1)
    weights = 1.0 + (BOUNDARY_WEIGHT - 1.0) * is_boundary
    return (per_sample * weights).mean()


# =============================================================================
# 训练与推断
# =============================================================================
def train_regressor(
    X_train, Y_train, W_train,
    X_val, Y_val, W_val,
    save_path=BEST_REG_PATH
):
    model = PerformanceSurrogate(input_dim=14, output_dim=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-5)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    Ytr = torch.tensor(Y_train, dtype=torch.float32, device=DEVICE)
    Wtr = torch.tensor(W_train, dtype=torch.float32, device=DEVICE)

    Xva = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    Yva = torch.tensor(Y_val, dtype=torch.float32, device=DEVICE)
    Wva = torch.tensor(W_val, dtype=torch.float32, device=DEVICE)

    best_val = np.inf
    patience = 40
    counter = 0

    history = []

    for epoch in range(1200):
        model.train()
        optimizer.zero_grad()
        pred = model(Xtr)
        loss = weighted_regression_loss(pred, Ytr, Wtr)
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
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1

        if counter >= patience:
            break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
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

    X_all = np.vstack([X_success, X_failed])
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


def predict_boundary_prob(boundary_model, X_norm):
    if boundary_model is None:
        return np.zeros(len(X_norm))
    boundary_model.eval()
    X_t = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = boundary_model(X_t).cpu().numpy()
    return 1 / (1 + np.exp(-logits))


def predict_feasible_prob(feas_clf, X_raw):
    if feas_clf is None:
        return np.ones(len(X_raw))
    return feas_clf.predict_proba(X_raw)[:, 1]


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


def train_geometry_warning_classifier(X_geom, y_bad, min_bad_required=8):
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
    return clf


def predict_geometry_safe_prob(geom_warn_clf, X_raw):
    if geom_warn_clf is None:
        return np.ones(len(X_raw))
    X_geom = snap_discrete_vars(X_raw)[:, :len(GEOM_WARN_FEATURE_NAMES)]
    p_bad = geom_warn_clf.predict_proba(X_geom)[:, 1]
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
        "-WorkingDir", current_work_dir
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
            print(f"     [{run_id}] 几何/网格失败，Exit Code: {ps_result.returncode}")
            return False, None, geometry_summary

        p_out_val = param_dict['P_out']
        nbl_val = int(round(param_dict['nBl']))

        print(f"     [{run_id}] 网格完成，启动 CFX（背压: {p_out_val:.3f} Pa）...")
        success_cfx, cfx_res, msg = run_cfx_pipeline(
            current_work_dir, run_id, p_out=p_out_val, cores=8, n_blades=nbl_val
        )

        if not success_cfx:
            print(f"     [{run_id}] CFD 失败: {msg}")
            return False, None, geometry_summary

        true_y = np.array([
            cfx_res['Efficiency'],
            cfx_res[TARGET_PR_NAME] if TARGET_PR_NAME in cfx_res else cfx_res['totalpressureratio'],
            cfx_res['Power'],
            cfx_res['MassFlow'],
        ], dtype=float)

        return True, true_y, geometry_summary

    except Exception as e:
        print(f"     [{run_id}] 未知异常: {e}")
        return False, None, None


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
    mean_quick, _ = mc_dropout_predict(reg_model, X_norm, scaler_Y, n_samples=5)
    pred_eff_q = mean_quick[:, 0]
    pred_pr_q  = mean_quick[:, 1]
    pred_mf_q  = mean_quick[:, 2]

    valid_mask = (
        (p_feas >= cfg.feasible_prob_threshold_pick) &
        (p_geom_safe >= cfg.geom_safe_prob_threshold_pick) &
        (~invalid_geom) &
        (pred_mf_q >= 3.55) &
        (pred_eff_q >= 0.60) &
        (pred_eff_q <= 0.85) &
        (pred_pr_q  <= 2.85)
    )

    # 失败点距离过滤（太近的直接排除）
    failed_norm = None
    if len(failed_points_raw) > 0:
        failed_raw  = snap_discrete_vars(np.array(failed_points_raw))
        failed_norm = scaler_X.transform(failed_raw)
        d_fail_all  = calc_distance_to_set(X_norm, failed_norm)
        valid_mask &= (d_fail_all > 0.04)

    valid_idx = np.where(valid_mask)[0]

    ehvi = np.full(N, -np.inf)

    if len(valid_idx) == 0:
        print("  [EHVI] 警告：预筛选后无有效候选点，所有点均无效。")
        return X_cand, ehvi, {"pred_mean": mean_quick, "p_feas": p_feas}

    # ------------------------------------------------------------------
    # 3. 对有效候选点做完整 MC Dropout（cfg.mc_samples 次）
    # ------------------------------------------------------------------
    X_valid      = X_cand[valid_idx]
    X_valid_norm = X_norm[valid_idx]

    reg_model.train()
    X_t = torch.tensor(X_valid_norm, dtype=torch.float32, device=DEVICE)
    mc_preds = []
    with torch.no_grad():
        for _ in range(cfg.mc_samples):
            p = reg_model(X_t).cpu().numpy()
            mc_preds.append(scaler_Y.inverse_transform(p))
    mc_preds = np.stack(mc_preds, axis=0)   # (S, N_valid, 3)

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
        "p_feas": p_feas,
        "p_geom_safe": p_geom_safe,
        "overlap_proxy_violation": overlap_proxy_violation(X_cand),
    }
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
        cfg: ALConfig
    ):
        self.reg_model = reg_model
        self.feas_clf = feas_clf
        self.geom_warn_clf = geom_warn_clf
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.X_pool_raw = snap_discrete_vars(X_pool_raw)
        self.Y_pool_raw = Y_pool_raw
        self.cfg = cfg

       
        
        self.eff_max_phys = 0.84
        self.eff_min_phys = 0.6
        self.pr_min_phys  = 1.80

        n_rules = geometry_rule_violations(self.X_pool_raw[:1]).shape[1]

        super().__init__(

            n_var=14,
            n_obj=2,
            n_ieq_constr=7 + n_rules,
            xl=L_BOUNDS,
            xu=U_BOUNDS
        )

    def _evaluate(self, X, out, *args, **kwargs):
        X = snap_discrete_vars(X)
        X_norm = self.scaler_X.transform(X)

        mean_real, std_norm = mc_dropout_predict(
            self.reg_model, X_norm, self.scaler_Y, n_samples=20
        )

        eff_raw = mean_real[:, 0]
        pr_raw  = mean_real[:, 1]
        mf      = mean_real[:, 2]
        unc = std_norm[:, :2].mean(axis=1)
        p_feas = predict_feasible_prob(self.feas_clf, X)
        p_geom_safe = predict_geometry_safe_prob(self.geom_warn_clf, X)
        overlap_violation = overlap_proxy_violation(X)
        eff_obj = np.clip(eff_raw, 0.45, 0.84)
        pr_obj  = np.clip(pr_raw, 1.60, 2.85)
        out["F"] = np.column_stack([
            -eff_obj + 0.12 * unc + OVERLAP_PENALTY_COEFF * overlap_violation + GEOM_WARN_PENALTY_COEFF * (1.0 - p_geom_safe),
            -pr_obj  + 0.12 * unc + OVERLAP_PENALTY_COEFF * overlap_violation + GEOM_WARN_PENALTY_COEFF * (1.0 - p_geom_safe)
        ])
        g_basic = np.column_stack([
            3.60 - mf,
            eff_raw - self.eff_max_phys,
            self.eff_min_phys - eff_raw,
            pr_raw - 2.85,                  
            self.pr_min_phys - pr_raw,
            self.cfg.feasible_prob_threshold_opt - p_feas,
            self.cfg.geom_safe_prob_threshold_opt - p_geom_safe,
        ])

        # 几何规则约束
        g_geom = geometry_rule_violations(X)

        out["G"] = np.column_stack([g_basic, g_geom])


# =============================================================================
# 选点策略
# =============================================================================
# =============================================================================
# 选点策略：EHVI 贪心 + 多样性约束
# =============================================================================
def select_candidates_diverse(
    acq_X: np.ndarray,
    ehvi_vals: np.ndarray,
    scaler_X,
    n_pick: int = 4,
    min_dist_norm: float = 0.08
):
    """
    贪心批量选点：
    每步选当前 EHVI 最高的点，然后在归一化空间中
    将距离 < min_dist_norm 的候选点排除，再选下一个。
    """
    # 预先归一化所有候选点（只算一次）
    X_cand_norm = scaler_X.transform(snap_discrete_vars(acq_X))

    selected   = []
    labels     = []
    scores     = ehvi_vals.copy()
    sel_norm   = []   # 已选点的归一化坐标

    for slot in range(n_pick):
        if not np.any(np.isfinite(scores) & (scores > 0)):
            break

        idx    = np.argmax(scores)
        x      = snap_discrete_vars(acq_X[idx:idx+1])[0]
        x_norm = X_cand_norm[idx]

        selected.append(x)
        labels.append(
            f"EHVI贪心#{slot+1} (ehvi={ehvi_vals[idx]:.5f})"
        )
        sel_norm.append(x_norm)

        # 排除与已选点过近的候选
        sel_arr   = np.array(sel_norm)               # (n_selected, 14)
        min_dists = cdist(X_cand_norm, sel_arr).min(axis=1)  # (N,)
        scores[min_dists < min_dist_norm] = -np.inf

    # 不足时随机补充
    attempts = 0
    while len(selected) < n_pick and attempts < 200:
        attempts += 1
        sampler = qmc.LatinHypercube(d=len(VAR_NAMES), seed=None)
        x_rand  = snap_discrete_vars(
            qmc.scale(sampler.random(1), L_BOUNDS, U_BOUNDS)
        )[0]

        # 即使在随机补充阶段，也尽量不要把明显违反几何规则的点送去 CFD。
        if np.any(geometry_rule_violations(x_rand[None, :]) > 0.0):
            continue

        selected.append(x_rand)
        labels.append("随机补充点")

    return selected[:n_pick], labels[:n_pick]

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

    pool_checkpoint_df = load_pool_checkpoint()
    if pool_checkpoint_df is not None:
        X_pool = pool_checkpoint_df[VAR_NAMES].values.astype(float)
        Y_pool = pool_checkpoint_df[ALL_OUTPUT_NAMES].values.astype(float)
        W_pool = pool_checkpoint_df['is_boundary'].values.astype(float)
        print(f"[断点续跑] 已从训练池 checkpoint 恢复样本: {len(X_pool)}")

    # 失败点池：主动学习中动态累积
    if os.path.exists(FAILED_POINTS_PATH):
        failed_points = list(np.load(FAILED_POINTS_PATH, allow_pickle=True))
    else:
        failed_points = []

    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    start_iter = get_resume_iter()
    effective_max_al_iters = CFG.max_al_iters if max_al_iters is None else int(max_al_iters)

    if os.path.exists(HV_CSV_PATH):
        hv_history = pd.read_csv(HV_CSV_PATH).to_dict('records')
    else:
        hv_history = []

    total_attempts = 0
    total_success = 0
    resumed_in_progress_iter = None
    if os.path.exists(CHECKPOINT_META_PATH):
        try:
            with open(CHECKPOINT_META_PATH, "r", encoding="utf-8") as f:
                checkpoint_meta = json.load(f)
            total_attempts = int(checkpoint_meta.get("total_attempts", 0))
            total_success = int(checkpoint_meta.get("total_success", 0))
            resumed_in_progress_iter = checkpoint_meta.get("in_progress_iter")
        except Exception as e:
            print(f"[断点续跑] 读取 checkpoint 元信息中的计数失败，将从 0 开始累计: {e}")

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
        X_tr, X_val, Y_tr, Y_val, W_tr, W_val = train_test_split(
            X_pool_norm, Y_pool_norm, W_pool,
            test_size=0.2,
            random_state=42 + al_iter,
            stratify=stratify_labels
        )

        # ---------------------------------------------------------------------
        # 2. 训练性能回归 surrogate
        # ---------------------------------------------------------------------
        reg_model, reg_hist = train_regressor(X_tr, Y_tr, W_tr, X_val, Y_val, W_val)
        joblib.dump(scaler_X, SCALER_X_PATH)
        joblib.dump(scaler_Y, SCALER_Y_PATH)

        # 固定测试集评估
        X_test_norm = scaler_X.transform(X_test_fixed)
        Y_test_pred = deterministic_predict(reg_model, X_test_norm, scaler_Y)

        mse_eff = mean_squared_error(Y_test_fixed_surr[:, 0], Y_test_pred[:, 0])
        mse_pr  = mean_squared_error(Y_test_fixed_surr[:, 1], Y_test_pred[:, 1])
        mse_mf  = mean_squared_error(Y_test_fixed_surr[:, 2], Y_test_pred[:, 2])

        print(f"  [测试误差] Eff={mse_eff:.6f}, PR={mse_pr:.6f}, MF={mse_mf:.6f}")

        # ---------------------------------------------------------------------
        # 3. 训练可行性分类器
        # ---------------------------------------------------------------------
        feas_clf = train_feasibility_classifier(X_pool, np.array(failed_points) if len(failed_points) > 0 else np.empty((0, 14)))
        if feas_clf is None:
            print("  [可行性分类器] 当前无失败样本，默认全部可行。")
        else:
            print(f"  [可行性分类器] 已用成功 {len(X_pool)} / 失败 {len(failed_points)} 样本训练。")

        # ---------------------------------------------------------------------
        # 4. 训练边界分类器
        # ---------------------------------------------------------------------
        boundary_model = train_boundary_classifier(X_pool_norm, W_pool.astype(int))
        if boundary_model is None:
            print("  [边界分类器] 当前类别不足，跳过。")
        else:
            print("  [边界分类器] 已训练。")

        # ---------------------------------------------------------------------
        # 4.5 训练几何报错分类器（使用 ActiveLearning_Runs 中真实 CFturbo warning）
        # ---------------------------------------------------------------------
        X_geom_warn, y_geom_warn = load_geometry_warning_dataset()
        geom_warn_clf = train_geometry_warning_classifier(X_geom_warn, y_geom_warn)
        if geom_warn_clf is None:
            print("  [几何报错分类器] 样本不足或类别不足，跳过。")
        else:
            joblib.dump(geom_warn_clf, GEOM_WARN_CLF_PATH)
            n_bad = int(y_geom_warn.sum())
            print(f"  [几何报错分类器] 已训练。样本={len(y_geom_warn)} | 报错样本={n_bad}")

        true_hv_ref_val, true_front_Y, true_front_X = compute_true_cumulative_hv(
            X_pool=X_pool,
            Y_pool=Y_pool,
            W_pool=W_pool,
            geom_warn_clf=geom_warn_clf,
            geom_safe_threshold=CFG.geom_safe_prob_threshold_pick,
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
            pareto_X = snap_discrete_vars(res.X)
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
            n_pick=CFG.n_eval_candidates_per_iter,
            min_dist_norm=CFG.diversity_min_dist
        )
      

        # ---------------------------------------------------------------------
        # 8. CFD 闭环更新
        # ---------------------------------------------------------------------
        for i, (x_cand, label) in enumerate(zip(candidates_X, labels)):
            run_id = f"AL_Iter{al_iter+1:02d}_P{i+1}"
            print(f"\n  -> [候选 {i+1}/{len(candidates_X)}] {label}")
            total_attempts += 1

            success, true_y, geometry_summary = run_single_cfd(x_cand, run_id)
            print_geometry_summary(geometry_summary)

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

            else:
                failed_points.append(x_cand.copy())
                print(f"     [失败] 已加入失败样本池，当前失败累计 {len(failed_points)}")

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
            geom_warn_clf=geom_warn_clf,
            geom_safe_threshold=CFG.geom_safe_prob_threshold_pick,
            ref_eff=TRUE_HV_REF_EFF,
            ref_pr=TRUE_HV_REF_PR
        )
        hv_history.append({
            "iter": al_iter + 1,
            "n_samples": len(X_pool),
            "true_hv": true_hv_val,
            "surrogate_hv": surrogate_hv_val,
            "mse_eff": mse_eff,
            "mse_pr": mse_pr,
            "mse_mf": mse_mf
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
    hv_df = pd.DataFrame(hv_history)
    hv_df.to_csv(HV_CSV_PATH, index=False)
    
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
    checkpoint_completed = get_resume_iter()
    if runtime_args.max_al_iters is not None:
        target_max_iters = runtime_args.max_al_iters
    else:
        target_max_iters = max(CFG.max_al_iters, checkpoint_completed + max(0, runtime_args.additional_iters))
    print(f"[运行参数] checkpoint={checkpoint_completed} | target_max_iters={target_max_iters}")
    main_multiobjective_active_learning(max_al_iters=target_max_iters)
