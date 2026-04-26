import argparse
import os
import tempfile
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
except ImportError:
    saltelli = None
    sobol = None

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 与主程序统一配置
# =============================================================================
VAR_NAMES = [
    'd1s', 'dH', 'beta1hb', 'beta1sb', 'd2', 'b2', 'beta2hb', 'beta2sb',
    'Lz', 't', 'TipClear', 'nBl', 'rake_te_s', 'P_out'
]
NBL_IDX = VAR_NAMES.index('nBl')
P_OUT_IDX = VAR_NAMES.index('P_out')

TARGET_PR_NAME = "totalpressureratio"
OUTPUT_NAMES = ['Efficiency', TARGET_PR_NAME, 'MassFlow']

L_BOUNDS = np.array([
    0.34, 0.044, 70.0, 20.0, 0.45, 0.044, 40.0, 40.0,
    0.185, 0.0015, 0.0007, 9, -25.0, 8.0
], dtype=float)

U_BOUNDS = np.array([
    0.42, 0.056, 84.0, 35.0, 0.53, 0.056, 54.0, 55.0,
    0.215, 0.0035, 0.0015, 12, -15.0, 13.0
], dtype=float)

TRAINING_CSV = "Compressor_Training_Data.csv"
BEST_MODEL_PATH = "best_regressor.pth"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_Y.pkl"
DOE_TRAINING_CSV = "Compressor_Training_Data.csv"
DOE_MODEL_PATH = "best_regressor_doe.pth"
DOE_SCALER_X_PATH = "scaler_X_doe.pkl"
DOE_SCALER_Y_PATH = "scaler_Y_doe.pkl"
AL_TRAINING_CSV = "al_training_pool_checkpoint.csv"
AL_MODEL_PATH = "best_regressor_al.pth"
AL_SCALER_X_PATH = "scaler_X_al.pkl"
AL_SCALER_Y_PATH = "scaler_Y_al.pkl"

N_SOBOL_BASE = 2048
DEVICE = "cpu"

TRUST_Q_LOW = 5
TRUST_Q_HIGH = 95
DEFAULT_FIXED_NBL = 10
CALC_SECOND_ORDER = False


# =============================================================================
# 基础工具
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="条件 Sobol 分析，支持单次运行与 DOE/AL 前后对比。"
    )
    parser.add_argument("--mode", choices=["single", "compare"], default="single")
    parser.add_argument("--fixed-nbl", type=int, default=DEFAULT_FIXED_NBL)
    parser.add_argument("--base-n", type=int, default=N_SOBOL_BASE)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--csv", default=TRAINING_CSV)
    parser.add_argument("--use-al-samples", action="store_true",
                        help="在主数据集基础上合并主动学习训练池样本，用于 trust region 与参考值统计。")
    parser.add_argument("--al-csv", default=AL_TRAINING_CSV,
                        help="主动学习训练池 CSV 路径，仅在启用 --use-al-samples 时使用。")
    parser.add_argument("--model", default=BEST_MODEL_PATH)
    parser.add_argument("--scaler-x", default=SCALER_X_PATH)
    parser.add_argument("--scaler-y", default=SCALER_Y_PATH)
    parser.add_argument("--compare-output", choices=["效率", "压比", "all"], default="all")
    parser.add_argument("--before-tag", default="DOE")
    parser.add_argument("--before-csv", default=DOE_TRAINING_CSV)
    parser.add_argument("--before-use-al-samples", action="store_true",
                        help="对比前一侧是否额外合并主动学习样本。")
    parser.add_argument("--before-al-csv", default=AL_TRAINING_CSV)
    parser.add_argument("--before-model", default=DOE_MODEL_PATH)
    parser.add_argument("--before-scaler-x", default=DOE_SCALER_X_PATH)
    parser.add_argument("--before-scaler-y", default=DOE_SCALER_Y_PATH)
    parser.add_argument("--after-tag", default="AL")
    parser.add_argument("--after-csv", default=AL_TRAINING_CSV)
    parser.add_argument("--after-use-al-samples", action="store_true",
                        help="对比后一侧是否额外合并主动学习样本。")
    parser.add_argument("--after-al-csv", default=AL_TRAINING_CSV)
    parser.add_argument("--after-model", default=AL_MODEL_PATH)
    parser.add_argument("--after-scaler-x", default=AL_SCALER_X_PATH)
    parser.add_argument("--after-scaler-y", default=AL_SCALER_Y_PATH)
    return parser.parse_args()


def build_output_stem(tag: str, suffix: str) -> str:
    safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag).strip("_")
    safe_tag = safe_tag or "default"
    return f"{safe_tag}_{suffix}"


def ensure_file_exists(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到{label}: {path}")


def ensure_salib_available():
    if saltelli is None or sobol is None:
        raise ImportError("未安装 SALib。请先安装 `SALib` 后再运行 Sobol 分析。")


def snap_discrete_vars(X: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=float, copy=True)
    if X.ndim == 1:
        X = X[None, :]
    X[:, 11] = np.clip(np.round(X[:, 11]), 9, 12)
    return X


def geometry_rule_violations(X: np.ndarray) -> np.ndarray:
    X = snap_discrete_vars(X)
    d1s, dH, beta1hb, beta1sb, d2, b2, beta2hb, beta2sb, Lz, t, TipClear, nBl, rake_te_s, P_out = X.T

    g1 = 0.070 - (d2 - d1s)
    g2 = 0.044 - b2
    g3 = t - 0.0035
    g4 = 0.0007 - TipClear
    g5 = TipClear - 0.0015
    g6 = (beta1hb - beta1sb) - 62.0
    g7 = np.abs(beta2hb - beta2sb) - 13.5
    g8 = 0.185 - Lz
    g9 = Lz - 0.215
    g10 = rake_te_s + 15.0
    g11 = -25.0 - rake_te_s

    return np.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    ensure_file_exists(csv_path, "数据文件")
    df = pd.read_csv(csv_path)

    required_cols = VAR_NAMES + OUTPUT_NAMES + ['is_boundary']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSV 缺少必要列: {c}")

    use_cols = VAR_NAMES + OUTPUT_NAMES + ['is_boundary']
    df = df[use_cols].copy()

    df['nBl'] = np.clip(np.round(df['nBl']), 9, 12).astype(int)
    df = df.drop_duplicates(subset=VAR_NAMES, keep='first').reset_index(drop=True)
    return df


def load_analysis_data(csv_path: str,
                       use_al_samples: bool = False,
                       al_csv_path: str = AL_TRAINING_CSV) -> pd.DataFrame:
    df_main = load_and_clean_data(csv_path)
    if not use_al_samples:
        print(f"[数据] 使用主数据集: {csv_path} (样本数={len(df_main)})")
        return df_main

    df_al = load_and_clean_data(al_csv_path)
    df = pd.concat([df_main, df_al], ignore_index=True)
    df = df.drop_duplicates(subset=VAR_NAMES, keep='first').reset_index(drop=True)
    print(
        f"[数据] 使用主数据集 + 主动学习样本: {csv_path} + {al_csv_path} "
        f"(主={len(df_main)}, AL={len(df_al)}, 合并去重后={len(df)})"
    )
    return df


def build_trust_region_from_success_data(X_success: np.ndarray,
                                         fixed_nBl: int,
                                         q_low: float = TRUST_Q_LOW,
                                         q_high: float = TRUST_Q_HIGH):
    X_success = snap_discrete_vars(X_success)
    mask = (X_success[:, NBL_IDX] == fixed_nBl)
    X_sub = X_success[mask]

    if len(X_sub) < 20:
        raise ValueError(
            f"固定 nBl={fixed_nBl} 后成功样本仅 {len(X_sub)} 个，"
            f"不足以稳定构建 trust region。建议换一个 nBl 或增加样本。"
        )

    cont_idx = [i for i in range(len(VAR_NAMES)) if i not in (NBL_IDX, P_OUT_IDX)]
    cont_names = [VAR_NAMES[i] for i in cont_idx]

    lower_trust = np.percentile(X_sub[:, cont_idx], q_low, axis=0)
    upper_trust = np.percentile(X_sub[:, cont_idx], q_high, axis=0)

    lower_trust = np.maximum(lower_trust, L_BOUNDS[cont_idx])
    upper_trust = np.minimum(upper_trust, U_BOUNDS[cont_idx])

    tiny = 1e-8
    for i in range(len(lower_trust)):
        if upper_trust[i] - lower_trust[i] < tiny:
            c = 0.5 * (upper_trust[i] + lower_trust[i])
            eps = max(1e-4, 0.01 * abs(c))
            lower_trust[i] = max(L_BOUNDS[cont_idx[i]], c - eps)
            upper_trust[i] = min(U_BOUNDS[cont_idx[i]], c + eps)

    return cont_idx, cont_names, lower_trust, upper_trust


# =============================================================================
# 代理模型
# =============================================================================
class PerformanceSurrogate(nn.Module):
    def __init__(self, input_dim=14, output_dim=len(OUTPUT_NAMES)):
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


def deterministic_predict(model, X_norm, scaler_Y):
    model.eval()
    X_t = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        y_norm = model(X_t).cpu().numpy()
    return scaler_Y.inverse_transform(y_norm)


def load_surrogate_and_scalers(model_path: str, scaler_x_path: str, scaler_y_path: str):
    model = PerformanceSurrogate(input_dim=14, output_dim=len(OUTPUT_NAMES)).to(DEVICE)

    ensure_file_exists(model_path, "模型权重")
    ensure_file_exists(scaler_x_path, "输入 scaler")
    ensure_file_exists(scaler_y_path, "输出 scaler")

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"[代理模型] 已加载权重: {model_path}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"未找到模型权重 {model_path}。请先运行主动学习主程序。"
        )

    try:
        scaler_X = joblib.load(scaler_x_path)
        scaler_Y = joblib.load(scaler_y_path)
        print(f"[Scaler] 已加载: {scaler_x_path}, {scaler_y_path}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"未找到 scaler 文件 {scaler_x_path} / {scaler_y_path}。"
        )

    model.eval()
    return model, scaler_X, scaler_Y


# =============================================================================
# Sobol 样本还原
# =============================================================================
def reconstruct_full_X_from_continuous_samples(X_cont: np.ndarray,
                                               cont_idx: list,
                                               fixed_nBl: int,
                                               fixed_p_out: float) -> np.ndarray:
    X_full = np.zeros((len(X_cont), len(VAR_NAMES)), dtype=float)

    for j, idx in enumerate(cont_idx):
        X_full[:, idx] = X_cont[:, j]

    X_full[:, NBL_IDX] = fixed_nBl
    X_full[:, P_OUT_IDX] = fixed_p_out
    X_full = snap_discrete_vars(X_full)
    return X_full


# =============================================================================
# 绘图
# =============================================================================
def plot_sobol_results(results_df: pd.DataFrame, output_name: str, fixed_nBl: int, output_stem: str):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(11, 5))

    sorted_vars = results_df['变量'].tolist()
    s1 = results_df['S1'].values
    st = results_df['ST'].values
    s1_conf = results_df['S1_conf'].values
    st_conf = results_df['ST_conf'].values

    x = np.arange(len(sorted_vars))
    width = 0.36

    ax.bar(x - width/2, s1, width, label='$S_1$(一阶)', color='steelblue',
           alpha=0.85, yerr=s1_conf, capsize=3)
    ax.bar(x + width/2, st, width, label='$S_T$(全效应)', color='tomato',
           alpha=0.85, yerr=st_conf, capsize=3)

    ax.axhline(y=0.01, color='gray', linestyle='--', lw=1.2, label='ST=0.01')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_vars, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Sobol 灵敏度指数")
    ax.set_title(f"条件 Sobol 分析（nBl={fixed_nBl}, 目标={output_name}）")
    ax.legend()
    ax.set_ylim(bottom=-0.02)

    plt.tight_layout()
    plot_path = f"{output_stem}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"[保存] 图表已写入 {plot_path}")
    plt.close(fig)


# =============================================================================
# 条件 Sobol
# =============================================================================
def run_conditional_sobol(output_idx: int = 0,
                          output_name: str = "效率",
                          fixed_nBl: int = DEFAULT_FIXED_NBL,
                          csv_path: str = TRAINING_CSV,
                          use_al_samples: bool = False,
                          al_csv_path: str = AL_TRAINING_CSV,
                          model_path: str = BEST_MODEL_PATH,
                          scaler_x_path: str = SCALER_X_PATH,
                          scaler_y_path: str = SCALER_Y_PATH,
                          output_tag: str = "default",
                          sobol_base_n: int = N_SOBOL_BASE):
    ensure_salib_available()
    df = load_analysis_data(
        csv_path=csv_path,
        use_al_samples=use_al_samples,
        al_csv_path=al_csv_path
    )

    X_data = df[VAR_NAMES].values.astype(float)
    sub_mask = (X_data[:, NBL_IDX] == fixed_nBl)
    if not np.any(sub_mask):
        raise ValueError(f"数据集中不存在 nBl={fixed_nBl} 的样本，无法固定 P_out。")
    p_out_ref = float(np.median(X_data[sub_mask, P_OUT_IDX]))

    model, scaler_X, scaler_Y = load_surrogate_and_scalers(model_path, scaler_x_path, scaler_y_path)

    cont_idx, cont_names, lower_trust, upper_trust = build_trust_region_from_success_data(
        X_data, fixed_nBl=fixed_nBl, q_low=TRUST_Q_LOW, q_high=TRUST_Q_HIGH
    )

    print(f"\n[条件Sobol] 固定 nBl = {fixed_nBl}")
    print(f"[条件Sobol] 固定 P_out = {p_out_ref:.6f}")
    print(f"[条件Sobol] 连续分析变量数 = {len(cont_names)}")
    print(f"[条件Sobol] Trust region（{TRUST_Q_LOW}% ~ {TRUST_Q_HIGH}% 分位数）:")
    for name, lo, hi in zip(cont_names, lower_trust, upper_trust):
        print(f"  {name:12s}: [{lo:.6f}, {hi:.6f}]")

    problem = {
        'num_vars': len(cont_names),
        'names': cont_names,
        'bounds': list(zip(lower_trust, upper_trust))
    }

    print(f"\n[条件Sobol] 数据源: {csv_path}")
    if use_al_samples:
        print(f"[条件Sobol] 附加主动学习样本: {al_csv_path}")
    print(f"[条件Sobol] 模型源: {model_path}")
    print(f"[条件Sobol] 生成 Saltelli 样本（N={sobol_base_n}, second_order={CALC_SECOND_ORDER}）...")
    X_cont = saltelli.sample(problem, sobol_base_n, calc_second_order=CALC_SECOND_ORDER)

    X_full = reconstruct_full_X_from_continuous_samples(
        X_cont, cont_idx=cont_idx, fixed_nBl=fixed_nBl, fixed_p_out=p_out_ref
    )

    geom_g = geometry_rule_violations(X_full)
    invalid_mask = np.any(geom_g > 0.0, axis=1)
    invalid_ratio = invalid_mask.mean()

    print(f"[条件Sobol] Sobol样本总数: {len(X_full)}")
    print(f"[条件Sobol] 几何规则违反比例: {invalid_ratio*100:.2f}%")

    if invalid_ratio > 0.05:
        print("[警告] trust region 内仍有较多样本违反几何规则。")
        print("       建议：缩小 trust region（例如 10%-90%）后再分析。")

    print(f"[条件Sobol] 在 surrogate 上评估 {len(X_full)} 个样本点...")
    X_norm = scaler_X.transform(X_full)
    Y_pred_real = deterministic_predict(model, X_norm, scaler_Y)

    Y_output = Y_pred_real[:, output_idx]

    print(f"[条件Sobol] 计算 Sobol 指数（目标：{output_name}）...")
    Si = sobol.analyze(
        problem,
        Y_output,
        calc_second_order=CALC_SECOND_ORDER,
        print_to_console=False
    )

    results_df = pd.DataFrame({
        '变量': cont_names,
        'S1': Si['S1'],
        'S1_conf': Si['S1_conf'],
        'ST': Si['ST'],
        'ST_conf': Si['ST_conf'],
        '交互效应': Si['ST'] - Si['S1']
    }).sort_values('ST', ascending=False)

    print(f"\n{'='*70}")
    print(f"  条件 Sobol 灵敏度分析结果 — 目标: {output_name} | 固定 nBl={fixed_nBl}")
    print(f"{'='*70}")
    print(results_df.to_string(index=False, float_format='{:.4f}'.format))

    output_stem = build_output_stem(output_tag, f"sobol_conditional_nBl{fixed_nBl}_{output_name}")
    csv_out_path = f"{output_stem}.csv"
    results_df.to_csv(csv_out_path, index=False, encoding='gbk')
    print(f"\n[保存] 结果已写入 {csv_out_path}")

    plot_sobol_results(results_df, output_name, fixed_nBl=fixed_nBl, output_stem=output_stem)
    return results_df


# =============================================================================
# nBl 离散影响
# =============================================================================
def analyze_discrete_nBl_effect(csv_path: str,
                                use_al_samples: bool,
                                al_csv_path: str,
                                model_path: str,
                                scaler_x_path: str,
                                scaler_y_path: str,
                                output_tag: str):
    df = load_analysis_data(
        csv_path=csv_path,
        use_al_samples=use_al_samples,
        al_csv_path=al_csv_path
    )
    X_data = df[VAR_NAMES].values.astype(float)

    model, scaler_X, scaler_Y = load_surrogate_and_scalers(model_path, scaler_x_path, scaler_y_path)

    x_ref = np.median(X_data, axis=0)
    rows = []

    for nbl in [9, 10, 11, 12]:
        x = x_ref.copy()
        x[11] = nbl
        x = snap_discrete_vars(x)[0]

        x_norm = scaler_X.transform(x.reshape(1, -1))
        y_pred = deterministic_predict(model, x_norm, scaler_Y)[0]

        rows.append({
            'nBl': nbl,
            'Efficiency': y_pred[0],
            TARGET_PR_NAME: y_pred[1],
            'MassFlow': y_pred[2]
        })

    res = pd.DataFrame(rows)

    print(f"\n{'='*60}")
    print("  离散变量 nBl 影响分析（代表点条件下）")
    print(f"{'='*60}")
    print(res.to_string(index=False, float_format='{:.5f}'.format))

    csv_out_path = f"{build_output_stem(output_tag, 'nBl_discrete_effect')}.csv"
    res.to_csv(csv_out_path, index=False, encoding='gbk')
    print(f"[保存] 结果已写入 {csv_out_path}")

    return res


# =============================================================================
# 多目标汇总
# =============================================================================
def merge_importance_tables(df_a: pd.DataFrame, name_a: str,
                            df_b: pd.DataFrame, name_b: str,
                            fixed_nBl: int,
                            output_tag: str):
    merged = pd.DataFrame({'变量': df_a['变量']})
    merged = merged.merge(
        df_a[['变量', 'ST']].rename(columns={'ST': f'ST_{name_a}'}),
        on='变量'
    ).merge(
        df_b[['变量', 'ST']].rename(columns={'ST': f'ST_{name_b}'}),
        on='变量'
    )
    merged['ST_综合'] = (merged[f'ST_{name_a}'] + merged[f'ST_{name_b}']) / 2
    merged = merged.sort_values('ST_综合', ascending=False)

    print(f"\n{'='*60}")
    print(f"  综合重要性排序（固定 nBl={fixed_nBl}）")
    print(f"{'='*60}")
    print(merged.to_string(index=False, float_format='{:.4f}'.format))

    out_path = f"{build_output_stem(output_tag, f'sobol_conditional_nBl{fixed_nBl}_综合排序')}.csv"
    merged.to_csv(out_path, index=False, encoding='gbk')
    print(f"[保存] 综合排序已写入 {out_path}")

    return merged


def compare_sobol_tables(df_before: pd.DataFrame,
                         df_after: pd.DataFrame,
                         before_tag: str,
                         after_tag: str) -> pd.DataFrame:
    merged = df_before[['变量', 'ST', 'S1']].rename(
        columns={'ST': f'ST_{before_tag}', 'S1': f'S1_{before_tag}'}
    ).merge(
        df_after[['变量', 'ST', 'S1']].rename(
            columns={'ST': f'ST_{after_tag}', 'S1': f'S1_{after_tag}'}
        ),
        on='变量',
        how='outer'
    )
    merged[f'ST_delta_{after_tag}_minus_{before_tag}'] = (
        merged[f'ST_{after_tag}'] - merged[f'ST_{before_tag}']
    )
    merged[f'S1_delta_{after_tag}_minus_{before_tag}'] = (
        merged[f'S1_{after_tag}'] - merged[f'S1_{before_tag}']
    )
    merged[f'Rank_{before_tag}'] = merged[f'ST_{before_tag}'].rank(ascending=False, method='min')
    merged[f'Rank_{after_tag}'] = merged[f'ST_{after_tag}'].rank(ascending=False, method='min')
    merged['Rank_delta'] = merged[f'Rank_{before_tag}'] - merged[f'Rank_{after_tag}']
    return merged.sort_values(f'ST_delta_{after_tag}_minus_{before_tag}', ascending=False)


def plot_sobol_comparison(compare_df: pd.DataFrame,
                          output_name: str,
                          fixed_nBl: int,
                          before_tag: str,
                          after_tag: str,
                          output_stem: str):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    ordered = compare_df.sort_values(f'ST_{after_tag}', ascending=False)
    x = np.arange(len(ordered))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, ordered[f'ST_{before_tag}'], width, label=f'ST_{before_tag}', color='steelblue', alpha=0.85)
    ax.bar(x + width / 2, ordered[f'ST_{after_tag}'], width, label=f'ST_{after_tag}', color='tomato', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered['变量'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Sobol 全效应指数")
    ax.set_title(f"Sobol 对比（nBl={fixed_nBl}, 目标={output_name}）")
    ax.legend()
    ax.set_ylim(bottom=-0.02)

    plt.tight_layout()
    plot_path = f"{output_stem}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"[保存] 对比图已写入 {plot_path}")
    plt.close(fig)


def run_full_sobol_suite(csv_path: str,
                         use_al_samples: bool,
                         al_csv_path: str,
                         model_path: str,
                         scaler_x_path: str,
                         scaler_y_path: str,
                         fixed_nBl: int,
                         output_tag: str,
                         sobol_base_n: int):
    df_eff = run_conditional_sobol(
        output_idx=0,
        output_name="效率",
        fixed_nBl=fixed_nBl,
        csv_path=csv_path,
        use_al_samples=use_al_samples,
        al_csv_path=al_csv_path,
        model_path=model_path,
        scaler_x_path=scaler_x_path,
        scaler_y_path=scaler_y_path,
        output_tag=output_tag,
        sobol_base_n=sobol_base_n
    )

    df_pr = run_conditional_sobol(
        output_idx=1,
        output_name="压比",
        fixed_nBl=fixed_nBl,
        csv_path=csv_path,
        use_al_samples=use_al_samples,
        al_csv_path=al_csv_path,
        model_path=model_path,
        scaler_x_path=scaler_x_path,
        scaler_y_path=scaler_y_path,
        output_tag=output_tag,
        sobol_base_n=sobol_base_n
    )

    merge_importance_tables(
        df_eff, "效率",
        df_pr, "压比",
        fixed_nBl=fixed_nBl,
        output_tag=output_tag
    )

    analyze_discrete_nBl_effect(
        csv_path=csv_path,
        use_al_samples=use_al_samples,
        al_csv_path=al_csv_path,
        model_path=model_path,
        scaler_x_path=scaler_x_path,
        scaler_y_path=scaler_y_path,
        output_tag=output_tag
    )
    return {"效率": df_eff, "压比": df_pr}


def run_compare_mode(args):
    before_results = run_full_sobol_suite(
        csv_path=args.before_csv,
        use_al_samples=args.before_use_al_samples,
        al_csv_path=args.before_al_csv,
        model_path=args.before_model,
        scaler_x_path=args.before_scaler_x,
        scaler_y_path=args.before_scaler_y,
        fixed_nBl=args.fixed_nbl,
        output_tag=args.before_tag,
        sobol_base_n=args.base_n
    )
    after_results = run_full_sobol_suite(
        csv_path=args.after_csv,
        use_al_samples=args.after_use_al_samples,
        al_csv_path=args.after_al_csv,
        model_path=args.after_model,
        scaler_x_path=args.after_scaler_x,
        scaler_y_path=args.after_scaler_y,
        fixed_nBl=args.fixed_nbl,
        output_tag=args.after_tag,
        sobol_base_n=args.base_n
    )

    outputs = ["效率", "压比"] if args.compare_output == "all" else [args.compare_output]
    for output_name in outputs:
        compare_df = compare_sobol_tables(
            before_results[output_name],
            after_results[output_name],
            before_tag=args.before_tag,
            after_tag=args.after_tag
        )
        output_stem = build_output_stem(
            f"{args.before_tag}_vs_{args.after_tag}",
            f"sobol_compare_nBl{args.fixed_nbl}_{output_name}"
        )
        compare_csv = f"{output_stem}.csv"
        compare_df.to_csv(compare_csv, index=False, encoding='gbk')
        print(f"[保存] 对比表已写入 {compare_csv}")
        plot_sobol_comparison(
            compare_df,
            output_name=output_name,
            fixed_nBl=args.fixed_nbl,
            before_tag=args.before_tag,
            after_tag=args.after_tag,
            output_stem=output_stem
        )


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "single":
        run_full_sobol_suite(
            csv_path=args.csv,
            use_al_samples=args.use_al_samples,
            al_csv_path=args.al_csv,
            model_path=args.model,
            scaler_x_path=args.scaler_x,
            scaler_y_path=args.scaler_y,
            fixed_nBl=args.fixed_nbl,
            output_tag=args.tag,
            sobol_base_n=args.base_n
        )
    else:
        run_compare_mode(args)
