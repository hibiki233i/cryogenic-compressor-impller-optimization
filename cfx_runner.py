import subprocess
import os
import glob
import json
import re
import threading
import time
import signal
import psutil
import uuid
from pathlib import Path
from cfx_convergence import (inspect_out, latest_pair, restart_ccl, atomic_json, digest, reject_result, legacy_restart_budget)

from dataclasses import dataclass

CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEMPLATE_CSE = os.path.join(PROJECT_DIR, "cfx_post", "Extract_Results.cse")
RESULT_SCHEMA_VERSION = 2
RESULT_METADATA_FILENAME = "CFX_Results.meta.json"


@dataclass(frozen=True)
class BoundaryBlockageStatus:
    iteration: int
    inlet_faces_pct: float | None
    inlet_area_pct: float | None
    outlet_faces_pct: float | None
    outlet_area_pct: float | None

    @property
    def inlet_is_100_percent(self) -> bool:
        return self.inlet_faces_pct == 100.0 and self.inlet_area_pct == 100.0

    @property
    def outlet_is_100_percent(self) -> bool:
        return self.outlet_faces_pct == 100.0 and self.outlet_area_pct == 100.0

    @property
    def both_are_100_percent(self) -> bool:
        return self.inlet_is_100_percent and self.outlet_is_100_percent


class CfxBlockageParser:
    """Incrementally pair boundary warnings with ratios inside one iteration.

    A ratio is accepted only when it follows the INLET/OUTLET warning in the
    same warning block.  An alarm status is produced only after both boundaries
    have an explicit faces-and-area ratio for the same solver iteration.
    """

    ITERATION_RE = re.compile(
        r"^[^A-Za-z0-9]*(?:OUTER\s+LOOP\s+ITERATION|ITERATION)"
        r"\s*(?:=|:)?\s*(\d+)\b",
        re.IGNORECASE,
    )
    BOUNDARY_RE = re.compile(
        r"wall\s+has\s+been\s+placed\s+at\s+portion\(s\)\s+of\s+an\s+"
        r"(INLET|OUTLET)\b",
        re.IGNORECASE,
    )
    RATIO_RE = re.compile(
        r"([0-9]+(?:\.[0-9]+)?)%\s+of\s+the\s+faces\s*,\s*"
        r"([0-9]+(?:\.[0-9]+)?)%\s+of\s+the\s+area",
        re.IGNORECASE,
    )
    WARNING_START_RE = re.compile(r"\bWARNING\s*#", re.IGNORECASE)

    def __init__(self):
        self._carry = ""
        self._iteration: int | None = None
        self._ratios: dict[str, tuple[float, float]] = {}
        self._block_boundary: str | None = None
        self._emitted = False

    def feed(self, text: str) -> list[BoundaryBlockageStatus]:
        combined = self._carry + str(text or "")
        raw_lines = combined.splitlines(keepends=True)
        self._carry = ""
        if raw_lines and not raw_lines[-1].endswith(("\n", "\r")):
            self._carry = raw_lines.pop()

        statuses: list[BoundaryBlockageStatus] = []
        for raw_line in raw_lines:
            line = raw_line.rstrip("\r\n")
            iteration_match = self.ITERATION_RE.search(line)
            if iteration_match:
                next_iteration = int(iteration_match.group(1))
                if self._iteration != next_iteration:
                    prior = self._status_if_observed()
                    if prior is not None and not self._emitted:
                        statuses.append(prior)
                    self._iteration = next_iteration
                    self._ratios = {}
                    self._block_boundary = None
                    self._emitted = False

            # A new warning header closes the previous boundary message block.
            # Ratios in this new block must not be attached to the old INLET or
            # OUTLET marker.
            if self._block_boundary is not None and self.WARNING_START_RE.search(line):
                self._block_boundary = None

            boundary_match = self.BOUNDARY_RE.search(line)
            if boundary_match:
                self._block_boundary = boundary_match.group(1).lower()

            ratio_match = self.RATIO_RE.search(line)
            if ratio_match and self._block_boundary is not None:
                self._ratios[self._block_boundary] = (
                    float(ratio_match.group(1)),
                    float(ratio_match.group(2)),
                )
                self._block_boundary = None
                if {"inlet", "outlet"}.issubset(self._ratios) and not self._emitted:
                    status = self._status_if_observed()
                    if status is not None:
                        statuses.append(status)
                        self._emitted = True
        return statuses

    def _status_if_observed(self) -> BoundaryBlockageStatus | None:
        if self._iteration is None or not self._ratios:
            return None
        inlet = self._ratios.get("inlet", (None, None))
        outlet = self._ratios.get("outlet", (None, None))
        return BoundaryBlockageStatus(
            iteration=self._iteration,
            inlet_faces_pct=inlet[0],
            inlet_area_pct=inlet[1],
            outlet_faces_pct=outlet[0],
            outlet_area_pct=outlet[1],
        )


def read_new_log_content(path, offset: int) -> tuple[str, int]:
    """Read bytes appended after ``offset`` and return the new byte position."""
    if not path or not os.path.exists(path):
        return "", int(offset)
    current_size = os.path.getsize(path)
    start = int(offset)
    if current_size < start:
        start = 0
    if current_size == start:
        return "", start
    with open(path, "rb") as stream:
        stream.seek(start)
        payload = stream.read(current_size - start)
    return payload.decode("utf-8", errors="ignore"), current_size


def solver_exit_failure_message(out_file, exit_code):
    """Preserve fatal-overflow evidence from the CFX output file."""
    tail = ""
    try:
        if out_file and os.path.exists(out_file):
            with open(out_file, "r", encoding="utf-8", errors="ignore") as stream:
                stream.seek(max(0, os.path.getsize(out_file) - 512 * 1024))
                tail = stream.read().lower()
    except OSError:
        tail = ""

    fatal_overflow_tokens = (
        "fatal overflow",
        "floating point overflow",
        "overflow error",
    )
    if any(token in tail for token in fatal_overflow_tokens):
        return (
            "CFX FATAL OVERFLOW（按批处理策略标记为设计不可行）。"
            f"Exit Code: {exit_code}"
        )
    return f"CFD 计算发散或崩溃。Exit Code: {exit_code}"


def _result_metadata_path(result_txt):
    return os.path.join(os.path.dirname(os.path.abspath(result_txt)), RESULT_METADATA_FILENAME)


def is_current_cfx_result(result_txt, residual_threshold=1e-4):
    """Accept current pressure definition only with final converged OUT/RES evidence."""
    if not os.path.exists(result_txt):
        return False
    metadata_path = _result_metadata_path(result_txt)
    if not os.path.exists(metadata_path):
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, ValueError, TypeError):
        return False
    root = Path(result_txt).parent
    if (root / "CFX_INVALID.json").exists() or (root / "DOE_INVALID.json").exists():
        return False
    out, res = latest_pair(root)
    if out is None or res is None:
        return False
    evidence = inspect_out(out, residual_threshold)
    if not evidence["accepted"] or max(out.stat().st_mtime_ns, res.stat().st_mtime_ns) > Path(result_txt).stat().st_mtime_ns:
        return False
    proof = metadata.get("convergence", {})
    if proof and (proof.get("out_sha256") != digest(out) or proof.get("res_name") != res.name
                  or proof.get("res_size") != res.stat().st_size
                  or proof.get("res_mtime_ns") != res.stat().st_mtime_ns):
        return False
    return (
        int(metadata.get("schema_version", 0)) == RESULT_SCHEMA_VERSION
        and metadata.get("total_pressure_frame") == "stationary"
        and metadata.get("total_pressure_averaging") == "massFlowAve"
    )


def _write_result_metadata(result_txt, convergence=None):
    metadata = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "total_pressure_frame": "stationary",
        "total_pressure_averaging": "massFlowAve",
        "total_pressure_ratio": "outlet_total_pressure / inlet_total_pressure",
    }
    if convergence is not None:
        metadata["convergence"] = convergence
    with open(_result_metadata_path(result_txt), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _env_or_default(name, default):
    value = os.environ.get(name)
    return value if value else default


def _is_cancelled(cancel_event=None):
    return bool(cancel_event is not None and cancel_event.is_set())


def _kill_process_tree(pid: int):
    killed = []
    try:
        root = psutil.Process(pid)
        targets = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return killed
    for proc in targets:
        try:
            proc.kill()
            killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return killed


def _run_cancellable_process(cmd, cwd, cancel_event=None, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
        text=True,
        creationflags=CREATE_NO_WINDOW,
    )
    while proc.poll() is None:
        if _is_cancelled(cancel_event):
            _kill_process_tree(proc.pid)
            return -999
        time.sleep(0.5)
    return int(proc.returncode or 0)


class _SolverProcesses:
    """Retain process identities across reparenting; scope detached CFX to this run."""

    NAMES = {'solver-mpi.exe', 'cfx5solve.exe', 'cfx5control.exe',
             'solver.exe', 'solver-mpi', 'cfx5solve', 'cfx5control'}

    def __init__(self, pid, working_dir, started):
        self.pid = pid
        self.directory = Path(working_dir).resolve()
        self.started = started
        self.known = {}

    def refresh(self):
        candidates = []
        # Keep Process objects: psutil verifies creation time before kill(),
        # preventing a recycled PID from targeting an unrelated process.
        roots = list(self.known.values())
        if not roots:
            try:
                roots = [psutil.Process(self.pid)]
            except psutil.NoSuchProcess:
                pass
        for root in roots:
            try:
                if root.is_running():
                    candidates.extend([root, *root.children(recursive=True)])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        for proc in psutil.process_iter(['name', 'create_time']):
            try:
                if ((proc.info.get('name') or '').lower() in self.NAMES
                        and proc.info.get('create_time', 0) >= self.started
                        and Path(proc.cwd()).resolve().is_relative_to(self.directory)):
                    candidates.extend([proc, *proc.children(recursive=True)])
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
        for proc in candidates:
            try:
                self.known[(proc.pid, proc.create_time())] = proc
            except psutil.NoSuchProcess:
                pass

    def stop(self, reason):
        print(f'[CFX] {reason}，正在终止本次求解进程 (launcher PID={self.pid})', flush=True)
        alive = []
        for _ in range(3):
            self.refresh()
            targets = list(self.known.values())
            for proc in reversed(targets):
                try:
                    if proc.is_running():
                        print(f'[CFX] 终止 PID={proc.pid}', flush=True)
                        proc.kill()
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied:
                    print(f'[CFX] 无权终止 PID={proc.pid}', flush=True)
            _, alive = psutil.wait_procs(targets, timeout=3)
            if not alive:
                print('[CFX] 本次求解进程已确认退出', flush=True)
                return
        raise RuntimeError(f'CFX process cleanup failed; surviving PIDs: {[p.pid for p in alive]}')


def _run_monitored_solver(cmd, working_dir, out_file, cancel_event=None):
    """Monitor this invocation only; never kill other runs by executable name."""
    started = time.time()
    proc = subprocess.Popen(cmd, cwd=working_dir, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
    processes = _SolverProcesses(proc.pid, working_dir, started)
    parser = CfxBlockageParser()
    offset = 0
    consecutive = {'inlet': 0, 'outlet': 0}
    previous = None
    stopped = False
    try:
        while True:
            processes.refresh()
            if _is_cancelled(cancel_event):
                stopped = True
                processes.stop('收到取消请求')
                return False, "canceled"
            content, offset = read_new_log_content(out_file, offset)
            blocked = False
            for status in parser.feed(content):
                if previous is None or status.iteration != previous + 1:
                    consecutive = {'inlet': 0, 'outlet': 0}
                for side in consecutive:
                    consecutive[side] = (consecutive[side] + 1
                        if getattr(status, f'{side}_is_100_percent') else 0)
                previous = status.iteration
                blocked = blocked or max(consecutive.values()) >= 3
            # Include the last complete warning even before the next iteration begins.
            if blocked or (Path(out_file).exists() and inspect_out(out_file)["blocked"]):
                stopped = True
                processes.stop('进出口任一边界连续三次迭代 100% wall')
                return False, "进出口任一边界持续100%堵塞 (blockage)"
            code = proc.poll()
            if code is not None:
                if code != 0:
                    return False, solver_exit_failure_message(out_file, code)
                return True, "Success"
            time.sleep(5)
    except Exception:
        if not stopped:
            processes.stop('求解监控异常')
        raise


def _solve_to_convergence(working_dir, initial_def, ccl_file, solver, cores,
                          residual_threshold, max_extra_iterations, restart_chunk, cancel_event=None):
    root = Path(working_dir)
    state_path = root / "cfx_convergence_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else legacy_restart_budget(root)
    latest_out, latest_res = latest_pair(root)
    for name in ('DOE_INVALID.json','CFX_INVALID.json'):
        marker = root/name
        if not marker.exists():
            continue
        invalid = json.loads(marker.read_text(encoding='utf-8'))
        superseded = (latest_out is not None and latest_res is not None
                      and latest_out.stat().st_mtime_ns > marker.stat().st_mtime_ns
                      and inspect_out(latest_out,residual_threshold)['accepted'])
        if not superseded:
            reason = str(invalid.get('reason',''))
            if '100_percent_wall' in reason or 'blockage' in reason:
                return None, None, 'Previously confirmed 100%堵塞 (blockage); a new successful rerun is required'
            if 'residual_unconverged' in reason:
                return None, None, reason
    if state.get("pending") and latest_out is not None and latest_res is not None and inspect_out(latest_out,residual_threshold)["accepted"]:
        state.pop("pending")
        atomic_json(state_path,state)
    if state.get("pending"):
        pending = root / state["pending"]
        if not pending.exists() or not pending.with_suffix('.res').exists() or not inspect_out(pending)["finished"]:
            return None, None, "CFX convergence restart interrupted; pending attempt requires review (budget retained)"
        state.pop("pending")
        atomic_json(state_path, state)
    out, res = latest_pair(root)
    if out is None:
        if list(root.glob('*.res')):
            return None, None, "CFX residual_unavailable: RES exists without matching OUT"
        name = "CFX_initial_" + uuid.uuid4().hex[:12]
        out = root / (name + '.out')
        cmd = [solver, '-def', str(initial_def), '-ccl', str(ccl_file), '-fullname', name,
               '-double', '-par-local', '-part', str(cores), '-batch']
        ok, message = _run_monitored_solver(cmd, str(root), str(out), cancel_event)
        if not ok:
            if 'blockage' in message:
                reject_result(root, message, {'out':str(out)})
            return None, None, message
        res = out.with_suffix('.res')
    while True:
        if _is_cancelled(cancel_event):
            return None, None, "canceled"
        if out is None or res is None or not out.exists() or not res.exists():
            return None, None, "CFX residual_unavailable: latest OUT has no matching RES"
        evidence = inspect_out(out, residual_threshold)
        state['latest_evidence'] = evidence
        atomic_json(state_path, state)
        if evidence['blocked']:
            message = '进出口任一边界持续100%堵塞 (blockage)'
            reject_result(root, message, evidence)
            return None, None, message
        if evidence['fatal']:
            return None, None, solver_exit_failure_message(str(out), 1)
        if not evidence['complete'] or not evidence['finished']:
            return None, None, 'CFX residual_unavailable: incomplete final RMS table or unfinished solver'
        if evidence['accepted']:
            evidence.update(out_sha256=digest(out), res_name=res.name,
                            res_size=res.stat().st_size,res_mtime_ns=res.stat().st_mtime_ns,
                            extra_iterations_reserved=state['extra_iterations_reserved'],policy_version=1)
            return str(res), evidence, 'Success'
        remaining = max_extra_iterations - state['extra_iterations_reserved']
        if remaining <= 0:
            message = f"CFX residual_unconverged: extra iteration budget {max_extra_iterations} exhausted; point discarded"
            reject_result(root, message, evidence)
            return None, None, message
        count = min(restart_chunk, remaining)
        name = 'CFX_resume_' + uuid.uuid4().hex[:12]
        next_out = root / (name + '.out')
        override = root / (name + '.ccl')
        override.write_text(restart_ccl(out, count, residual_threshold), encoding='utf-8')
        # Reserve before launching so cancellation/recovery cannot grant another 1500.
        state['extra_iterations_reserved'] += count
        state['attempts'].append({'source_res':res.name,'out':next_out.name,'iterations':count})
        state['pending'] = next_out.name
        atomic_json(state_path, state)
        print(f"[CFX] RMS max={evidence['max_rms']:.3g}; restart from {res.name}, +{count} iterations")
        cmd = [solver, '-def', str(res), '-ccl', str(override), '-fullname', name,
               '-double', '-par-local', '-part', str(cores), '-batch']
        ok, message = _run_monitored_solver(cmd, str(root), str(next_out), cancel_event)
        if not ok:
            if 'blockage' in message:
                reject_result(root, message, {'out':str(next_out)})
            return None, None, message
        state.pop('pending', None)
        atomic_json(state_path, state)
        out, res = next_out, next_out.with_suffix('.res')


def run_cfx_pipeline(
    working_dir,
    run_id,
    p_out,
    cores=8,
    n_blades=1,
    cfx_bin_dir=None,
    template_cfx=None,
    template_cse=None,
    cancel_event=None,
    residual_threshold=1e-4,
    max_extra_iterations=1500,
    restart_chunk=500,
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
    template_cse = template_cse or _env_or_default("IMPELLER_TEMPLATE_CSE", DEFAULT_TEMPLATE_CSE)
    
    gtm_file = os.path.join(working_dir, "Impeller_Mesh.gtm").replace("\\", "/")
    def_file = os.path.join(working_dir, "Impeller.def").replace("\\", "/")
    pre_script = os.path.join(working_dir, "Update_Mesh.pre")
    output_txt = os.path.join(working_dir, "CFX_Results.txt")

    # =============================================================================
    # 0-A. 若结果文件已存在，则直接读取返回
    # 与 DOE.py 中的已完成检测配合，避免重复求解。
    # =============================================================================
    if not (0 < residual_threshold < 1) or not (0 <= max_extra_iterations <= 1500) or restart_chunk <= 0:
        raise ValueError("Invalid CFX convergence policy")
    if _is_cancelled(cancel_event):
        return False, None, "canceled"
    if is_current_cfx_result(output_txt, residual_threshold):
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
    elif os.path.exists(output_txt):
        print(
            f"[{run_id}] 发现旧版结果文件（缺少 stationary-frame v{RESULT_SCHEMA_VERSION} "
            "元数据），将使用现有 .res 重新执行 CFX-Post。"
        )
 

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
    #      已有 RES 必须与最新 OUT 同名，并通过残差验收后才进入 Post。
    #      不达标时从 RES 续算，不依赖已清理的 DEF。
    # ==========================================
    existing_out, existing_res = latest_pair(working_dir)
    if existing_out is None and not glob.glob(os.path.join(working_dir, '*.res')):
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
            
        pre_ret = _run_cancellable_process(
            [cfx5pre_exe, "-batch", pre_script],
            cwd=working_dir,
            cancel_event=cancel_event,
        )
        if pre_ret == -999:
            return False, None, "canceled"
        if pre_ret != 0:
            return False, None, f"CFX-Pre 失败，无法生成 .def 文件。Exit Code: {pre_ret}"

        if not os.path.exists(def_file):
            return False, None, "CFX-Pre 运行结束但未找到 .def 文件"


    try:
        res_file, convergence, message = _solve_to_convergence(
            working_dir, def_file, ccl_file, cfx5solve_exe, cores,
            residual_threshold, max_extra_iterations, restart_chunk, cancel_event)
    except (OSError, ValueError) as exc:
        return False, None, f"CFX convergence check failed: {exc}"
    if res_file is None:
        return False, None, message
    # A successful Post must create a fresh file; stale values cannot masquerade as success.
    if os.path.exists(output_txt):
        archive = Path(working_dir) / 'replaced_results' / uuid.uuid4().hex
        archive.mkdir(parents=True)
        Path(output_txt).replace(archive / 'CFX_Results.txt')
        old_meta = Path(_result_metadata_path(output_txt))
        if old_meta.exists():
            old_meta.replace(archive / old_meta.name)
    # ==========================================
    # 3. CFX-Post：运行宏提取数据
    # ==========================================
    print(f"[{run_id}] 正在提取气动性能参数...")
    post_cmd = [
        cfx5post_exe, 
        "-batch", template_cse, 
        "-res", res_file
    ]
    
    post_ret = _run_cancellable_process(
        post_cmd,
        cwd=working_dir,
        cancel_event=cancel_event,
    )
    if post_ret == -999:
        return False, None, "canceled"
    if post_ret != 0:
        return False, None, f"CFX-Post 后处理失败。Exit Code: {post_ret}"

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
            if not all(__import__('math').isfinite(v) for v in cfx_results.values()):
                raise ValueError("Nonfinite CFX performance result")
            _write_result_metadata(output_txt, convergence)
            (Path(working_dir) / 'CFX_INVALID.json').unlink(missing_ok=True)
            (Path(working_dir) / 'DOE_INVALID.json').unlink(missing_ok=True)
        
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
