import subprocess
import os
import glob
import threading
import time
import signal
import psutil 

CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0


def _env_or_default(name, default):
    value = os.environ.get(name)
    return value if value else default


def run_cfx_pipeline(
    working_dir,
    run_id,
    p_out,
    cores=8,
    n_blades=1,
    cfx_bin_dir=None,
    template_cfx=None,
    template_cse=None,
):
    """
    完整的 CFX 自动化流水线：网格替换 -> 求解 -> 结果提取
    引入断点续算与动态 _00X.res 识别
    """
    cfx_bin_dir = cfx_bin_dir or _env_or_default("IMPELLER_CFX_BIN_DIR", r"D:\ANSYS Inc\v251\CFX\bin")
    cfx5pre_exe   = os.path.join(cfx_bin_dir, "cfx5pre.exe")
    cfx5solve_exe = os.path.join(cfx_bin_dir, "cfx5solve.exe")
    cfx5post_exe  = os.path.join(cfx_bin_dir, "cfx5post.exe")
    template_cfx = template_cfx or _env_or_default("IMPELLER_TEMPLATE_CFX", r"F:\optimazition\Templates\BaseModel.cfx")
    template_cse = template_cse or _env_or_default("IMPELLER_TEMPLATE_CSE", r"F:\optimazition\Templates\Extract_Results.cse")
    
    gtm_file = os.path.join(working_dir, "Impeller_Mesh.gtm").replace("\\", "/")
    def_file = os.path.join(working_dir, "Impeller.def").replace("\\", "/")
    pre_script = os.path.join(working_dir, "Update_Mesh.pre")
    output_txt = os.path.join(working_dir, "CFX_Results.txt")

 # ==========================================
    # 0-A. 最高优先级：若结果文件已存在，直接读取返回
    #      与 DOE.py 的 5-A 检测形成双重保险：
    #      无论从哪里调用本函数，只要结果已完整提取过，
    #      就跳过全部求解与后处理步骤。
    # ==========================================
    if os.path.exists(output_txt):
        try:
            with open(output_txt, 'r') as f:
                data = f.read().strip().split(',')
            cfx_results = {
                'Efficiency':    float(data[0]),               # 无量纲，不乘叶片数
                'PressureRatio': float(data[1]),               # 无量纲，不乘叶片数
                'Power':         float(data[2]) * n_blades,   # 整机功率 = 单流道 × nBl
                'MassFlow':      float(data[3]) * n_blades,    # 整机流量 = 单流道 × nBl
                'totalpressureratio': float(data[4])      # 总压比
            }
            print(f"[{run_id}] 发现已提取的结果文件，直接返回，跳过全部计算。")
            return True, cfx_results, "Recovered from existing result"
        except Exception as e:
            # 结果文件损坏或格式异常，继续往下重新走完整流程
            print(f"[{run_id}] 结果文件存在但读取失败（{e}），将重新执行后处理。")
 

    # ==========================================
    # 0-B. 动态生成覆盖背压的 CCL 文件
    # ==========================================
    ccl_file = os.path.join(working_dir, "update_bc.ccl").replace("\\", "/")
    ccl_content = f"""
LIBRARY:
  CEL:
    EXPRESSIONS:
      MyBackPressure = {p_out} [Pa]
    END
  END
END
"""
    with open(ccl_file, "w", encoding="utf-8") as f:
        f.write(ccl_content.strip())
    
    # ==========================================
    # 0-C. 断点续算：检查是否已有 .res 求解结果
    #      注意：只有在 output_txt 不存在时才会走到这里，
    #      因此 existing_res 代表"求解完成但 Post 未提取"的状态。
    # ==========================================
    existing_res = sorted(
        glob.glob(os.path.join(working_dir, "*.res")),
        key=os.path.getmtime
    )
 
    if existing_res:
        # .res 存在 + output_txt 不存在 → 求解已完成，仅需重跑 Post
        res_file = existing_res[-1].replace("\\", "/")
        print(f"[{run_id}] 发现已有 .res 文件 {os.path.basename(res_file)}，"
              f"跳过求解，直接进入后处理。")
    else:
        # ==========================================
        # 1. CFX-Pre：动态生成脚本并替换网格
        # ==========================================
        print(f"[{run_id}] 正在合成物理边界条件 (CFX-Pre)...")
        pre_content = f"""
COMMAND FILE:
  CFX Pre Version = 25.1
END
>load filename={template_cfx.replace("\\", "/")}
>update
> gtmImport filename={gtm_file}, type=GTM, \
units=m, nameStrategy= Assembly
>update
>writeCaseFile filename={def_file}, operation=\
write def file
> update
>quit
"""
        with open(pre_script, "w", encoding="utf-8") as f:
            f.write(pre_content.strip())
            
        try:
            subprocess.run([cfx5pre_exe, "-batch", pre_script], 
                           cwd=working_dir, check=True, capture_output=True, text=True) 
        except Exception as e:
            return False, None, f"CFX-Pre 失败，无法生成 .def 文件: {e}"

        if not os.path.exists(def_file):
            return False, None, "CFX-Pre 运行结束但未找到 .def 文件"

        # ==========================================
        # 2. CFX-Solve：调用求解器
        # ==========================================
        print(f"[{run_id}] 正在运行 CFX 求解器...")
        solve_cmd = [
            cfx5solve_exe,
            "-def",      def_file,
            "-ccl",      ccl_file,
            "-double",
            "-par-local",
            "-part",     str(cores),
            "-batch"
        ]


    # 找到 .out 文件路径（CFX 自动生成，名称与 def 文件一致）
        def_basename = os.path.splitext(os.path.basename(def_file))[0]
        out_file = os.path.join(working_dir, f"{def_basename}_001.out")

        solve_proc = subprocess.Popen(
            solve_cmd,
            cwd=working_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW
    )

        # ── 堵塞监控线程 ──────────────────────────────────────────────
        blocked_flag = threading.Event()
        def kill_cfx_tree(pid: int, run_id: str):
            """
            先用 psutil 递归收集整棵进程树，
            再逐个强杀，确保 solver-mpi.exe 等脱离子树的进程也被清理。
            """
            killed = []
            try:
                root = psutil.Process(pid)
        # recursive=True 可以拿到所有后代，包括跨 Job Object 的情况
                children = root.children(recursive=True)
                targets = children + [root]
                for p in targets:
                    try:
                        print(f"[{run_id}] 正在杀死进程: {p.name()} (PID={p.pid})")
                        p.kill()
                        killed.append(p.pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except psutil.NoSuchProcess:
                pass

    # 兜底：按进程名强杀，应对 detached 的 solver-mpi.exe
            for proc_name in ["solver-mpi.exe", "cfx5solve.exe", "cfx5control.exe"]:
                for p in psutil.process_iter(['name', 'pid']):
                    if p.info['name'] and p.info['name'].lower() == proc_name.lower():
                        try:
                            p.kill()
                            print(f"[{run_id}] 按名称补杀: {proc_name} (PID={p.pid})")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

            return killed   # 触发后主线程知道要终止

        def monitor_out_file():
            """
            每30秒检查一次 .out 文件。
            若连续 CHECK_PATIENCE 次都同时出现 INLET 和 OUTLET 100% 堵塞，
            则设置 blocked_flag 通知主线程终止求解。
            """
            CHECK_INTERVAL  = 30   # 秒，检查间隔
            CHECK_PATIENCE  = 3    # 连续出现几次才判定为真正堵塞（避免误判启动阶段）
            WAIT_BEFORE_MON = 120  # 秒，求解开始后等多久才开始监控（避免初始迭代误判）

            inlet_pattern  = "wall has been placed at portion(s) of an INLET"
            outlet_pattern = "wall has been placed at portion(s) of an OUTLET"
            block_100      = "100.0% of the faces, 100.0% of the area"

        # 等待求解稳定后再开始监控
            time.sleep(WAIT_BEFORE_MON)

            consecutive = 0
            last_size   = 0

            while not blocked_flag.is_set() and solve_proc.poll() is None:
                time.sleep(CHECK_INTERVAL)

                if not os.path.exists(out_file):
                   continue

            # 只读取文件新增内容，避免每次全量扫描大文件
                current_size = os.path.getsize(out_file)
                if current_size == last_size:
                   continue

                try:
                    with open(out_file, 'r', errors='ignore') as f:
                    # 只读最新的 50KB，避免大文件读取太慢
                        f.seek(max(0, current_size - 51200))
                        recent_content = f.read()
                except Exception:
                    continue

                last_size = current_size

            # 判断是否同时出现 INLET 和 OUTLET 100% 堵塞
                has_inlet_block  = (inlet_pattern  in recent_content and
                                    block_100       in recent_content)
                has_outlet_block = (outlet_pattern in recent_content and
                                    block_100       in recent_content)

                if has_inlet_block and has_outlet_block:
                    consecutive += 1
                    print(f"[{run_id}] ⚠ 检测到进出口100%堵塞 "
                        f"（第 {consecutive}/{CHECK_PATIENCE} 次）")
                    if consecutive >= CHECK_PATIENCE:
                        print(f"[{run_id}] ✗ 确认持续堵塞，强制终止求解器。")
                        blocked_flag.set()
                else:
                # 堵塞消失，重置计数（求解可能在恢复）
                    if consecutive > 0:
                        print(f"[{run_id}] 堵塞消失，重置计数，继续监控...")
                    consecutive = 0

        monitor_thread = threading.Thread(target=monitor_out_file, daemon=True)
        monitor_thread.start()

    # ── 主线程等待求解器，同时响应监控信号 ─────────────────────────
        try:
            while True:
                ret = solve_proc.poll()

                if blocked_flag.is_set():
                    print(f"[{run_id}] 正在强制终止 CFX 进程树...")
                    kill_cfx_tree(solve_proc.pid, run_id)
                    time.sleep(3)
            # 确认 solver-mpi.exe 已经退出
    
                    still_alive = [p for p in psutil.process_iter(['name', 'pid'])
                                   if p.info['name'] and 'solver-mpi' in p.info['name'].lower()]
                    if still_alive:
                        print(f"[{run_id}] ⚠ 仍有 {len(still_alive)} 个 solver-mpi 进程残留，正在补杀...")
                        for p in still_alive:
                            try:
                                p.kill()
                                print(f"[{run_id}] 补杀成功: solver-mpi.exe (PID={p.pid})")
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                    return False, None, "进出口持续100%堵塞，提前终止"

                if ret is not None:
                    break  # 求解器自然退出

                time.sleep(5)

            if solve_proc.returncode != 0:
                return False, None, f"CFD 计算发散或崩溃。Exit Code: {solve_proc.returncode}"

        # ✅ 修复后
        except Exception as e:
            kill_cfx_tree(solve_proc.pid, run_id)
            return False, None, f"求解器异常: {e}"

        finally:
            blocked_flag.set()
            monitor_thread.join(timeout=10)

        new_res = sorted(
            glob.glob(os.path.join(working_dir, "*.res")),
            key=os.path.getmtime
        )
        if not new_res:
            return False, None, "求解结束但未生成任何 .res 结果文件"
        res_file = new_res[-1].replace("\\", "/")
        print(f"[{run_id}] 求解成功！生成结果文件: {os.path.basename(res_file)}")
    # ==========================================
    # 3. CFX-Post：运行宏提取数据
    # ==========================================
    print(f"[{run_id}] 正在提取气动性能参数...")
    post_cmd = [
        cfx5post_exe, 
        "-batch", template_cse, 
        "-res", res_file
    ]
    
    try:
        subprocess.run(post_cmd, cwd=working_dir, check=True, capture_output=True, text=True,encoding='utf-8', errors='ignore') # 移除 timeout
    except Exception as e:
        return False, None, f"CFX-Post 后处理失败: {e}"

    # ==========================================
    # 4. 读取结果与清理空间
    # ==========================================
    if os.path.exists(output_txt):
        try:
            with open(output_txt, 'r') as f:
                data = f.read().strip().split(',')
                # 支持 4 维输出：效率, 压比, 功率, 流量
                cfx_results = {
                    'Efficiency': float(data[0]),
                    'PressureRatio': float(data[1]),
                    'Power': float(data[2]) * n_blades,      
                    'MassFlow': float(data[3]) * n_blades,   # 整机流量 = 单流道 × nBl
                    'totalpressureratio': float(data[4])      # 总压比
                }
        
            for f_path in [gtm_file, def_file]:
                try:
                    if os.path.exists(f_path):
                        os.remove(f_path)
                except OSError:
                    pass
 
            return True, cfx_results, "Success"
 
        except Exception as e:
            return False, None, f"结果文件解析异常: {e}"
    else:
        return False, None, "CFX-Post 执行完毕但未生成结果文本文件"
