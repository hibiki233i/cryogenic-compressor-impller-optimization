import subprocess
import numpy as np
import pandas as pd
import os
import re
import json
import time
from scipy.stats import qmc
from cfx_runner import run_cfx_pipeline
import os
import glob
from design_variables import ensure_training_csv, load_variable_specs, lower_bounds, training_csv_columns, upper_bounds, variable_names

def clean_old_results(base_dir, filename="CFX_Results.txt"):
    """
    在 DOE 启动前，遍历并删除所有旧的结果文件，强制重新提取。
    """
    print(f"=== 开始清理历史结果文件 ({filename}) ===")
    
    # 构造递归搜索路径，例如：F:/optimazition/**/*.txt
    # 注意：recursive=True 需要 Python 3.5+
    search_pattern = os.path.join(base_dir, '**', filename)
    old_files = glob.glob(search_pattern, recursive=True)
    
    count = 0
    for file_path in old_files:
        try:
            os.remove(file_path)
            print(f"已删除旧文件: {file_path}")
            count += 1
        except Exception as e:
            print(f"⚠ 无法删除 {file_path}: {e}")
            
    print(f"=== 清理完毕，共清除了 {count} 个残留文件 ===\n")

CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0

# =====================================================================
# 1. 定义物理边界
# =====================================================================
num_samples = 300
input_dim = 14
MIN_VALID    = 0.1   # g/s，低于此值：CFD 数值发散，纯噪声
MIN_NORMAL   = 3.6   # g/s，低于此值但高于 MIN_VALID：边界珍贵数据
# 高于 MIN_NORMAL：正常工况数据

# 定义变量名称和物理边界
variable_specs = load_variable_specs()
var_names = variable_names(variable_specs)
L_BOUNDS = lower_bounds(variable_specs)
U_BOUNDS = upper_bounds(variable_specs)

# 初始化 LHS 采样器并生成 [0, 1) 范围内的样本
sampler = qmc.LatinHypercube(d=input_dim, seed=42)
sample_norm = sampler.random(n=num_samples)
sample_real = qmc.scale(sample_norm, L_BOUNDS, U_BOUNDS)
samples = [dict(zip(var_names, row)) for row in sample_real]
# =====================================================================
# 2. Python 循环驱动 PowerShell
# =====================================================================
ps_script_path = r"F:\optimazition\Run-GeometryMeshing.ps1"
working_dir_base = r"F:\optimazition\Runs"
output_csv = "Compressor_Training_Data.csv"
EXTRA_SAMPLES_FILE = "extra_samples.json"
# =====================================================================
# 3. 动态补充点（超出初始 LHS 后）：生成并持久化
# =====================================================================
def get_new_sample():
    """
    生成一个新的随机补充样本，并追加保存到 extra_samples.json，
    保证断点续跑时不丢失已生成的补充点序列。
    """
    norm  = qmc.LatinHypercube(d=input_dim).random(1)   # 无 seed，每次随机
    real  = qmc.scale(norm, L_BOUNDS, U_BOUNDS)
    new_p = dict(zip(var_names, real[0]))

    extras = []
    if os.path.exists(EXTRA_SAMPLES_FILE):
        with open(EXTRA_SAMPLES_FILE, 'r') as f:
            extras = json.load(f)
    extras.append(new_p)
    with open(EXTRA_SAMPLES_FILE, 'w') as f:
        json.dump(extras, f, indent=2)

    return new_p

def load_extra_samples():
    """读取上次已保存的动态补充点列表。"""
    if os.path.exists(EXTRA_SAMPLES_FILE):
        with open(EXTRA_SAMPLES_FILE, 'r') as f:
            return json.load(f)
    return []
# =====================================================================
# 4. 断点恢复：扫描 Runs 目录，重建进度
# =====================================================================
def recover_progress(working_dir_base, output_csv,samples, extra_samples):
    """
    扫描 Runs 目录，找出最大已运行索引（无论成功与否），
    同时统计 CSV 中已记录的成功条数。

    返回：
        successful_count : int  —— CSV 中已有的有效行数
        current_idx      : int  —— 下一个待运行的样本索引
    """
    MIN_DISCARD = 0.0001 
    columns = var_names + ['Efficiency', 'PressureRatio', 'Power', 'MassFlow', 'totalpressureratio', 'is_boundary']

    recovered_rows = []     # 完整可写入 CSV 的数据行
    repost_count   = 0      # ②类：重新提取 Post 的成功数量
    max_run_idx    = -1
    if not os.path.exists(working_dir_base):
        print("[断点恢复] Runs 目录不存在，从零开始。")
        # 全新运行，初始化空 CSV
        ensure_training_csv(output_csv, variable_specs)
        return 0, 0

    # 按编号顺序遍历所有 Run_xxx 文件夹
    all_run_dirs = sorted(
        [d for d in os.listdir(working_dir_base) if re.match(r"Run_(\d+)", d)],
        key=lambda d: int(re.match(r"Run_(\d+)", d).group(1))
    )

    for run_dir in all_run_dirs:
        idx          = int(re.match(r"Run_(\d+)", run_dir).group(1))
        max_run_idx  = max(max_run_idx, idx)
        full_path    = os.path.join(working_dir_base, run_dir)
        result_txt   = os.path.join(full_path, "CFX_Results.txt")
        res_files    = glob.glob(os.path.join(full_path, "*.res"))

        if os.path.exists(result_txt):
            # ── ① 结果完整，重建数据行 ──────────────────────────────────
            try:
                with open(result_txt, 'r') as f:
                    data = f.read().strip().split(',')

                # 恢复输入参数：LHS 点直接从 samples[] 取，动态补充点从 extra_samples 取
                if idx < len(samples):
                    p = samples[idx].copy()
                elif (idx - len(samples)) < len(extra_samples):
                    p = dict(extra_samples[idx - len(samples)])
                else:
                    print(f"  [警告] {run_dir} 索引超出已知样本范围，跳过。")
                    continue

                nbl_val = int(round(p['nBl']))
                mass_flow_single = float(data[3])
                mass_flow_total  = mass_flow_single * nbl_val

                # 喘振/发散过滤（与主循环保持一致）
                if mass_flow_total < MIN_DISCARD:
                    print(f"  [重建过滤] {run_dir}：MassFlow={mass_flow_total:.6f} "
                          f"< {MIN_DISCARD}，发散点，跳过。")
                    continue

                p['Efficiency']    = float(data[0])
                p['PressureRatio'] = float(data[1])
                p['Power']         = float(data[2]) * nbl_val
                p['totalpressureratio'] = float(data[4])      # 总压比
                p['MassFlow']      = mass_flow_total
                p['is_boundary']   = 1 if mass_flow_total < MIN_NORMAL else 0

                recovered_rows.append(p)

            except Exception as e:
                print(f"  [警告] {run_dir} 结果文件读取失败（{e}），跳过。")

        elif res_files:
            # ── ② 有 .res，无 CFX_Results.txt：
            #    求解已完成，txt 被手动删除（提取宏旧版有问题）。
            #    直接在此处调用 run_cfx_pipeline 重新提取，
            #    cfx_runner 的 0-C 节会自动跳过求解，仅重跑 Post。
            print(f"  [重新提取] {run_dir}：发现 .res 但无 CFX_Results.txt，"
                  f"正在重新运行 CFX-Post...")
            try:
                if idx < len(samples):
                    p = samples[idx].copy()
                elif (idx - len(samples)) < len(extra_samples):
                    p = dict(extra_samples[idx - len(samples)])
                else:
                    print(f"  [警告] {run_dir} 索引超出已知样本范围，跳过。")
                    continue

                p_out_val = p['P_out']
                nbl_val   = int(round(p['nBl']))

                success_cfx, cfx_res, msg = run_cfx_pipeline(
                    full_path, f"Recovery-{run_dir}",
                    p_out=p_out_val,
                    cores=8,
                    n_blades=nbl_val
                )

                if success_cfx:
                    mass_flow_total = cfx_res['MassFlow']
                    if mass_flow_total < MIN_DISCARD:
                        print(f"  [重建过滤] {run_dir}：MassFlow={mass_flow_total:.6f}，"
                              f"发散点，跳过。")
                        continue

                    p['Efficiency']    = cfx_res['Efficiency']
                    p['PressureRatio'] = cfx_res['PressureRatio']
                    p['Power']         = cfx_res['Power']
                    p['MassFlow']      = mass_flow_total
                    p['totalpressureratio'] = cfx_res['totalpressureratio']
                    p['is_boundary']   = 1 if mass_flow_total < MIN_NORMAL else 0
                    recovered_rows.append(p)
                    print(f"  [重建成功] {run_dir}：效率={p['Efficiency']:.4f}，"
                          f"压比={p['PressureRatio']:.4f}，"
                          f"流量={p['MassFlow']:.5f} g/s。")
                else:
                    print(f"  [重建失败] {run_dir}：CFX-Post 重提取失败 → {msg}，跳过。")

            except Exception as e:
                print(f"  [重建失败] {run_dir}：发生异常 → {e}，跳过。")


    rebuilt_df = pd.DataFrame(recovered_rows, columns=columns) \
                 if recovered_rows else pd.DataFrame(columns=columns)
    rebuilt_df.to_csv(output_csv, index=False)

    successful_count = len(recovered_rows)

    # current_idx：从最小的"未处理"索引开始
    # ②类待续跑的点编号 < max_run_idx，主循环按顺序推进时会自然遇到它们
    # 因此直接从 max_run_idx+1 开始，②类由 run_single_sample 内部的 5-A 检测处理
    # ——但 ② 类没有 CFX_Results.txt，5-A 不会命中，会进入 cfx_runner 并触发 Post 续跑
    current_idx = max_run_idx + 1

    print(f"\n{'='*60}")
    print(f"[断点恢复] 扫描完成")
    print(f"  Runs 目录最大索引 : Run_{max_run_idx:03d}")
    print(f"  重建有效数据行数  : {successful_count} 条  → 已写入 {output_csv}")
    print(f"  主循环将从        : Run_{current_idx:03d} 继续")
    print(f"{'='*60}\n")

    return successful_count, current_idx

import glob   # recover_progress 内部用到

# 执行断点恢复（在加载 extra_samples 之后调用，保证补充点可被正确重建）
extra_samples = load_extra_samples()
successful_count, current_idx = recover_progress(
    working_dir_base, output_csv, samples, extra_samples
)
# =====================================================================
# 5. 单样本执行函数（含已完成检测 + 软失败自动重试）
# =====================================================================
def run_single_sample(i, p):
    """
    执行单个样本的完整流水线：几何 → 网格 → CFD → 结果提取。

    失败类型区分：
        硬失败：参数畸形 / CFD 发散 → 不重试，直接丢弃
        软失败：CFturbo/TurboGrid 偶发崩溃 → 最多重试 MAX_RETRIES 次

    返回：(success: bool, run_id: str, data: dict | str)
    """
    run_id          = f"Run_{i:03d}"
    current_work_dir = os.path.join(working_dir_base, run_id)
    result_txt       = os.path.join(current_work_dir, "CFX_Results.txt")
    

    # ------------------------------------------------------------------
    # 5-A  已完成检测：若结果文件已存在，直接读取返回，无需重跑
    # ------------------------------------------------------------------
    if os.path.exists(result_txt):
        try:
            with open(result_txt, 'r') as f:
                data = f.read().strip().split(',')
            p['Efficiency']    = float(data[0])
            p['PressureRatio'] = float(data[1])
            p['Power']         = float(data[2])
            p['MassFlow']      = float(data[3])
            p['totalpressureratio'] = float(data[4])  # 总压比
            print(f"[{run_id}] 已有结果文件，直接读取，跳过计算。")
            return True, run_id, p
        except Exception as e:
            print(f"[{run_id}] 结果文件读取失败（{e}），将重新计算。")

    # ------------------------------------------------------------------
    # 5-B  硬失败预检：参数畸形 → 直接返回，不重试，不占用成功名额
    # ------------------------------------------------------------------
    d1s_d2_ratio = p['d1s'] / p['d2']
    if d1s_d2_ratio < 0.50 or d1s_d2_ratio > 0.85:
        return False, run_id, f"[硬失败] d1s/d2={d1s_d2_ratio:.3f} 超出[0.50,0.85]，跳过"
    os.makedirs(current_work_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 5-C  构建 PowerShell 命令（叶片数取整）
    # ------------------------------------------------------------------
    cmd = [
        r"C:\Program Files\PowerShell\7\pwsh.exe",
        "-ExecutionPolicy", "Bypass",
        "-File", ps_script_path,
        "-WorkingDir", current_work_dir
    ]
    for name in var_names:
        if name != 'P_out':
            val = str(int(round(p[name]))) if name == 'nBl' else str(p[name])
            cmd.extend([f"-{name}", val])
    cmd.extend(["-mFlow", "0.0036", "-N_rpm", "10000", "-alpha0", "0.0"])

    # ------------------------------------------------------------------
    # 5-D  软失败重试循环（针对 CFturbo/TurboGrid 偶发崩溃）
    # ------------------------------------------------------------------
    MAX_RETRIES = 2
    last_error  = "未知错误"

    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            print(f"[{run_id}] *** 软失败重试 第 {attempt}/{MAX_RETRIES - 1} 次 "
                  f"（等待 15s 让软件资源释放）... ***")
            time.sleep(15)

        try:
            ps_result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if ps_result.returncode == 2:
    # CFturbo 致命警告，物理不可行，硬失败不重试
                last_error = "[硬失败] CFturbo 致命警告（阻塞/热力学失效），丢弃。"
                print(f"[{run_id}] {last_error}")
                break

            elif ps_result.returncode == 1:
    # CFturbo 无害警告但 curve 文件缺失，软失败可重试
    # （如果 curve 文件存在说明只是警告，PowerShell 已继续到 TurboGrid）
                last_error = f"网格生成失败 Exit Code: 1 (尝试 {attempt+1}/{MAX_RETRIES})"
                print(f"[{run_id}] {last_error}")
                continue

            elif ps_result.returncode != 0:
                last_error = f"网格生成失败 Exit Code: {ps_result.returncode}"
                continue

            # 网格成功，调用 CFX 流水线
            p_out_val = p['P_out']
            nbl_val   = int(round(p['nBl']))   # 传入叶片数，用于整机流量换算

            success_cfx, cfx_res, msg = run_cfx_pipeline(
                current_work_dir, run_id,
                p_out=p_out_val,
                cores=8,
                n_blades=nbl_val    # ← 修复：单流道流量 × nBl = 整机流量
            )

            if success_cfx:
                p['Efficiency']    = cfx_res['Efficiency']
                p['PressureRatio'] = cfx_res['PressureRatio']
                p['Power']         = cfx_res['Power']
                p['MassFlow']      = cfx_res['MassFlow']
                p['totalpressureratio'] = cfx_res['totalpressureratio']

                return True, run_id, p
            else:
                # CFD 发散或崩溃：硬失败，不再重试
                last_error = f"[硬失败] CFD 计算失败：{msg}"
                print(f"[{run_id}] {last_error}")
                break   # → 跳出重试循环，直接返回失败

        except subprocess.TimeoutExpired:
            last_error = "[硬失败] PowerShell 执行超时"
            print(f"[{run_id}] {last_error}")
            break   # 超时视为硬失败，不重试

        except Exception as e:
            last_error = f"[软失败?] 未知异常：{e}"
            print(f"[{run_id}] {last_error}")
            # 未知异常保守处理：继续重试

    return False, run_id, last_error
doe_working_dir = r"F:\optimazition\Runs"
clean_old_results(doe_working_dir, filename="CFX_Results.txt")

# =====================================================================
# 6. 主循环：串行驱动，含喘振点过滤
# =====================================================================
target_samples = 300  
MIN_DISCARD    = 0.0001   

# 加载上次已生成的动态补充点（用于断点续跑时的索引对齐）
extra_ptr = max(0, current_idx - len(samples))

print(f"--- 开始串行评估样本，目标收集 {target_samples} 个有效点 ---")
print(f"--- 当前进度：已成功 {successful_count} 个，从 Run_{current_idx:03d} 继续 ---\n")

while successful_count < target_samples:
    # 取样：优先走 LHS，超出后走持久化的动态补充点，再不够才新生成
    if current_idx < len(samples):
        current_p = samples[current_idx].copy()
    elif extra_ptr < len(extra_samples):
        # 断点续跑：复用上次已生成的补充点，保持序列一致
        current_p = dict(extra_samples[extra_ptr])
        extra_ptr += 1
    else:
        # 真正需要新补充点
        current_p = get_new_sample()
        extra_ptr += 1

    success, run_id, data = run_single_sample(current_idx, current_p)

    if success:
        mass_flow = data.get('MassFlow', 0)

        if mass_flow < MIN_DISCARD:
            # ── CFD 发散点：彻底丢弃 ─────────────────────────────────
            print(f"[{run_id}] ✗ 发散点：MassFlow={mass_flow:.6f} g/s，丢弃。")

        elif mass_flow < MIN_NORMAL:
            data['is_boundary'] = 1
            pd.DataFrame([data]).to_csv(output_csv, mode='a', header=False, index=False)
            successful_count += 1
            print(f"[{run_id}] ⚠ 边界点保留：MassFlow={mass_flow:.4f} g/s "
                  f"（近喘振区）（{successful_count}/{target_samples}）。")

        else:
            # ── 正常工况点 ────────────────────────────────────────────
            data['is_boundary'] = 0
            pd.DataFrame([data]).to_csv(output_csv, mode='a', header=False, index=False)
            successful_count += 1
            print(f"[{run_id}] ✓ 正常点记录（{successful_count}/{target_samples}）。")
    else:
        print(f"[{run_id}] ✗ 失败 → {data}。丢弃，继续下一点。")

    current_idx += 1

print(f"\n采样完成！成功收集到 {successful_count} 组有效数据，已写入 {output_csv}。")
