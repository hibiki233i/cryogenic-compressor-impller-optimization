from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from scipy.stats import qmc

from ..config import AppConfig
from ..models import TaskResult, TaskUpdate
from cfx_runner import run_cfx_pipeline
from design_variables import ensure_training_csv, load_variable_specs, lower_bounds, training_csv_columns, upper_bounds, variable_names

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _emit(progress_callback, message: str, progress=None, metrics=None):
    if progress_callback:
        progress_callback(TaskUpdate(status="running", message=message, progress=progress, metrics=metrics or {}))


def _is_cancelled(cancel_event=None) -> bool:
    return bool(cancel_event is not None and cancel_event.is_set())


def _kill_process_tree(pid: int) -> None:
    try:
        root = psutil.Process(pid)
        targets = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return
    for proc in targets:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


class RunnerAPI:
    def __init__(self, config: AppConfig):
        self.config = config.resolved()
        self.variable_specs = load_variable_specs(self.config.workspace.design_variables_json)
        self.variable_names = variable_names(self.variable_specs)
        self.l_bounds = lower_bounds(self.variable_specs)
        self.u_bounds = upper_bounds(self.variable_specs)
        self.min_discard_flow_g_s = float(self.config.runtime.default_discard_flow_g_s)
        self.boundary_flow_g_s = float(self.config.runtime.default_boundary_flow_g_s)

    def validate_environment(self) -> TaskResult:
        cfg = self.config
        checks = {
            "powershell_exe": cfg.solver.powershell_exe,
            "geometry_script": cfg.solver.geometry_script_path,
            "cfturbo_exe": cfg.solver.cfturbo_exe,
            "turbogrid_exe": cfg.solver.turbogrid_exe,
            "template_cfx": cfg.solver.template_cfx,
            "template_cse": cfg.solver.template_cse,
            "base_cft": cfg.solver.base_cft,
            "cft_batch_template": cfg.solver.cft_batch_template,
            "turbogrid_template": cfg.solver.turbogrid_template,
        }
        cfx_execs = [
            cfg.solver.cfx_bin_dir / "cfx5pre.exe",
            cfg.solver.cfx_bin_dir / "cfx5solve.exe",
            cfg.solver.cfx_bin_dir / "cfx5post.exe",
        ]
        for idx, exe in enumerate(cfx_execs, start=1):
            checks[f"cfx_exe_{idx}"] = exe

        missing = {name: str(path) for name, path in checks.items() if not Path(path).exists()}
        if missing:
            return TaskResult(status="failed", message="Environment validation failed.", metrics={"missing": missing})

        created_training_csv = False
        training_csv = cfg.workspace.training_csv
        try:
            if not training_csv.exists():
                ensure_training_csv(training_csv, self.variable_specs)
                created_training_csv = True
            ensure_training_csv(training_csv, self.variable_specs)
        except Exception as exc:
            return TaskResult(
                status="failed",
                message="Environment validation failed while preparing training CSV.",
                metrics={"training_csv": str(training_csv), "error": str(exc)},
            )

        checked = {k: str(v) for k, v in checks.items()}
        checked["training_csv"] = str(training_csv)
        checked["design_variables_json"] = str(cfg.workspace.design_variables_json)
        return TaskResult(
            status="succeeded",
            message="Environment validation passed.",
            metrics={"checked": checked, "created_training_csv": created_training_csv},
        )

    def recover_runs(self) -> TaskResult:
        recovery = self._recover_doe_progress()
        return TaskResult(
            status="succeeded",
            message="DOE run recovery scan completed.",
            metrics={
                "run_dirs": recovery["run_dirs"],
                "completed_runs": len(recovery["rows"]),
                "partial_runs": recovery["partial_runs"],
                "next_index": recovery["next_index"],
            },
        )

    def generate_lhs_samples(self, count: int, seed: int = 42) -> list[dict]:
        sampler = qmc.LatinHypercube(d=len(self.variable_names), seed=seed)
        sample_real = qmc.scale(sampler.random(n=count), self.l_bounds, self.u_bounds)
        return [dict(zip(self.variable_names, row)) for row in sample_real]

    def _columns(self) -> list[str]:
        return training_csv_columns(self.variable_specs)

    def _load_extra_samples(self) -> list[dict]:
        path = self.config.workspace.extra_samples_json
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return []

    def _save_extra_samples(self, payload: list[dict]) -> None:
        self.config.workspace.extra_samples_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _new_extra_sample(self) -> dict:
        norm = qmc.LatinHypercube(d=len(self.variable_names)).random(1)
        sample_real = qmc.scale(norm, self.l_bounds, self.u_bounds)[0]
        sample = dict(zip(self.variable_names, sample_real))
        extras = self._load_extra_samples()
        extras.append(sample)
        self._save_extra_samples(extras)
        return sample

    def _result_row_from_sample(self, sample: dict, raw_values: list[str]) -> dict | None:
        row = dict(sample)
        nbl = int(round(float(row["nBl"])))
        mass_flow = float(raw_values[3]) * nbl
        if mass_flow < self.min_discard_flow_g_s:
            return None
        row["nBl"] = nbl
        row["Efficiency"] = float(raw_values[0])
        row["PressureRatio"] = float(raw_values[1])
        row["Power"] = float(raw_values[2]) * nbl
        row["MassFlow"] = mass_flow
        row["totalpressureratio"] = float(raw_values[4])
        row["is_boundary"] = 1 if mass_flow < self.boundary_flow_g_s else 0
        return row

    def _read_result_file(self, result_txt: Path, sample: dict) -> dict | None:
        data = result_txt.read_text(encoding="utf-8").strip().split(",")
        return self._result_row_from_sample(sample, data)

    def _sample_for_index(self, index: int, base_samples: list[dict], extra_samples: list[dict]) -> dict | None:
        if index < len(base_samples):
            return dict(base_samples[index])
        extra_index = index - len(base_samples)
        if 0 <= extra_index < len(extra_samples):
            return dict(extra_samples[extra_index])
        return None

    def _recover_doe_progress(self) -> dict:
        runs_dir = self.config.workspace.doe_runs_dir
        runs_dir.mkdir(parents=True, exist_ok=True)
        base_samples = self.generate_lhs_samples(self.config.runtime.doe_initial_samples)
        extra_samples = self._load_extra_samples()
        rows: list[dict] = []
        partial_runs = 0
        max_run_idx = -1
        run_dirs = sorted([p for p in runs_dir.glob("Run_*") if p.is_dir()], key=lambda p: int(p.name.split("_")[1]))
        for run_dir in run_dirs:
            idx = int(run_dir.name.split("_")[1])
            max_run_idx = max(max_run_idx, idx)
            sample = self._sample_for_index(idx, base_samples, extra_samples)
            if sample is None:
                continue
            result_txt = run_dir / "CFX_Results.txt"
            if result_txt.exists():
                try:
                    row = self._read_result_file(result_txt, sample)
                    if row is not None:
                        rows.append(row)
                except Exception:
                    pass
            elif list(run_dir.glob("*.res")):
                partial_runs += 1
        df = pd.DataFrame(rows, columns=self._columns()) if rows else pd.DataFrame(columns=self._columns())
        df.to_csv(self.config.workspace.training_csv, index=False)
        return {
            "rows": rows,
            "run_dirs": len(run_dirs),
            "partial_runs": partial_runs,
            "next_index": max_run_idx + 1,
            "base_samples": base_samples,
            "extra_samples": extra_samples,
        }

    def _append_training_row(self, row: dict) -> None:
        training_csv = self.config.workspace.training_csv
        frame = pd.DataFrame([row], columns=self._columns())
        if training_csv.exists():
            frame.to_csv(training_csv, mode="a", header=False, index=False)
        else:
            frame.to_csv(training_csv, index=False)

    def build_geometry_command(self, working_dir: Path, sample: dict) -> list[str]:
        cmd = [
            str(self.config.solver.powershell_exe),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.config.solver.geometry_script_path),
            "-WorkingDir",
            str(working_dir),
            "-CFturboExe",
            str(self.config.solver.cfturbo_exe),
            "-TurboGridExe",
            str(self.config.solver.turbogrid_exe),
            "-CftBatchTemplate",
            str(self.config.solver.cft_batch_template),
            "-BaseCft",
            str(self.config.solver.base_cft),
            "-TurboGridTemplate",
            str(self.config.solver.turbogrid_template),
        ]
        for name in self.variable_names:
            if name == "P_out":
                continue
            value = int(round(float(sample[name]))) if name == "nBl" else sample[name]
            cmd.extend([f"-{name}", str(value)])
        cmd.extend(["-mFlow", str(self.config.runtime.mass_flow), "-N_rpm", str(self.config.runtime.rpm), "-alpha0", str(self.config.runtime.alpha0)])
        return cmd

    def run_cfx_case(self, working_dir: Path, run_id: str, p_out: float, n_blades: int, cancel_event=None) -> TaskResult:
        success, payload, message = run_cfx_pipeline(
            str(working_dir),
            run_id,
            p_out=p_out,
            cores=self.config.runtime.cfx_cores,
            n_blades=n_blades,
            cfx_bin_dir=str(self.config.solver.cfx_bin_dir),
            template_cfx=str(self.config.solver.template_cfx),
            template_cse=str(self.config.solver.template_cse),
            cancel_event=cancel_event,
        )
        status = "succeeded" if success else ("canceled" if message == "canceled" else "failed")
        return TaskResult(status=status, message=message, metrics=payload or {}, artifacts={"working_dir": str(working_dir)})

    def _run_streamed_command(self, cmd: list[str], working_dir: Path, progress_callback=None, cancel_event=None) -> tuple[str, str, int]:
        proc = subprocess.Popen(
            cmd,
            cwd=str(working_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
            creationflags=CREATE_NO_WINDOW,
        )
        lines: list[str] = []
        line_queue: queue.Queue[str | None] = queue.Queue()

        def read_output():
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line_queue.put(line.rstrip())
            finally:
                line_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        reader_done = False

        while proc.poll() is None:
            while True:
                try:
                    line = line_queue.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    reader_done = True
                    continue
                if line:
                    lines.append(line)
                    _emit(progress_callback, line)
            if _is_cancelled(cancel_event):
                _kill_process_tree(proc.pid)
                reader.join(timeout=2)
                return "\n".join(lines), "", -999
            time.sleep(0.2)

        reader.join(timeout=2)
        while not reader_done:
            try:
                line = line_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                reader_done = True
            elif line:
                lines.append(line)
                _emit(progress_callback, line)
        return "\n".join(lines), "", int(proc.returncode or 0)

    def run_geometry_generation(self, working_dir: Path, sample: dict, run_id: str | None = None, progress_callback=None, cancel_event=None) -> TaskResult:
        working_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_geometry_command(working_dir, sample)
        prefix = f"{run_id}: " if run_id else ""
        stdout, stderr, returncode = self._run_streamed_command(cmd, working_dir, progress_callback=progress_callback, cancel_event=cancel_event)
        if returncode == -999:
            return TaskResult(
                status="canceled",
                message=f"{prefix}geometry generation canceled.",
                metrics={"stdout": stdout, "stderr": stderr},
                artifacts={"working_dir": str(working_dir)},
            )
        if returncode != 0:
            return TaskResult(
                status="failed",
                message=f"{prefix}geometry generation failed with exit code {returncode}.",
                metrics={"stdout": stdout, "stderr": stderr},
                artifacts={"working_dir": str(working_dir)},
            )
        return TaskResult(
            status="succeeded",
            message=f"{prefix}geometry and mesh generated.",
            metrics={"stdout": stdout, "stderr": stderr},
            artifacts={"working_dir": str(working_dir)},
        )

    def run_doe_sample(self, index: int, sample: dict, progress_callback=None, cancel_event=None) -> TaskResult:
        run_id = f"Run_{index:03d}"
        working_dir = self.config.workspace.doe_runs_dir / run_id
        working_dir.mkdir(parents=True, exist_ok=True)
        if _is_cancelled(cancel_event):
            return TaskResult(status="canceled", message=f"{run_id}: canceled before start.", artifacts={"working_dir": str(working_dir)})
        result_txt = working_dir / "CFX_Results.txt"
        if result_txt.exists():
            row = self._read_result_file(result_txt, sample)
            if row is None:
                return TaskResult(status="failed", message=f"{run_id}: existing result diverged and was discarded.", artifacts={"working_dir": str(working_dir)})
            return TaskResult(status="succeeded", message=f"{run_id}: reused existing result.", metrics=row, artifacts={"working_dir": str(working_dir)})
        _emit(progress_callback, f"{run_id}: generating geometry and mesh...")
        geometry_result = self.run_geometry_generation(working_dir, sample, run_id=run_id, progress_callback=progress_callback, cancel_event=cancel_event)
        if geometry_result.status != "succeeded":
            return geometry_result
        if _is_cancelled(cancel_event):
            return TaskResult(status="canceled", message=f"{run_id}: canceled before CFX.", artifacts={"working_dir": str(working_dir)})
        _emit(progress_callback, f"{run_id}: geometry done, launching CFX...")
        cfx_result = self.run_cfx_case(working_dir, run_id, float(sample["P_out"]), int(round(float(sample["nBl"]))), cancel_event=cancel_event)
        if cfx_result.status != "succeeded":
            return cfx_result
        row = {
            **sample,
            "nBl": int(round(float(sample["nBl"]))),
            "Efficiency": float(cfx_result.metrics["Efficiency"]),
            "PressureRatio": float(cfx_result.metrics["PressureRatio"]),
            "Power": float(cfx_result.metrics["Power"]),
            "MassFlow": float(cfx_result.metrics["MassFlow"]),
            "totalpressureratio": float(cfx_result.metrics["totalpressureratio"]),
            "is_boundary": 1 if float(cfx_result.metrics["MassFlow"]) < self.boundary_flow_g_s else 0,
        }
        if float(row["MassFlow"]) < self.min_discard_flow_g_s:
            return TaskResult(status="failed", message=f"{run_id}: divergent MassFlow, discarded.", metrics=row, artifacts={"working_dir": str(working_dir)})
        self._append_training_row(row)
        return TaskResult(status="succeeded", message=f"{run_id}: DOE sample completed.", metrics=row, artifacts={"working_dir": str(working_dir)})

    def run_doe_batch(self, progress_callback=None, cancel_event=None) -> TaskResult:
        self.config.workspace.doe_runs_dir.mkdir(parents=True, exist_ok=True)
        recovery = self._recover_doe_progress()
        samples = recovery["base_samples"]
        extra_samples = recovery["extra_samples"]
        current_index = int(recovery["next_index"])
        target = int(self.config.runtime.doe_target_samples)
        completed = len(recovery["rows"])
        extra_ptr = max(0, current_index - len(samples))
        while completed < target:
            if _is_cancelled(cancel_event):
                return TaskResult(status="canceled", message="DOE batch canceled.", metrics={"completed_runs": completed, "target_runs": target})
            if current_index < len(samples):
                sample = dict(samples[current_index])
            elif extra_ptr < len(extra_samples):
                sample = dict(extra_samples[extra_ptr])
                extra_ptr += 1
            else:
                sample = self._new_extra_sample()
                extra_samples.append(sample)
                extra_ptr += 1
            result = self.run_doe_sample(current_index, sample, progress_callback=progress_callback, cancel_event=cancel_event)
            current_index += 1
            if result.status == "canceled":
                return TaskResult(status="canceled", message=result.message, metrics={"completed_runs": completed, "target_runs": target}, artifacts=result.artifacts)
            if result.status == "succeeded":
                completed += 1
                _emit(progress_callback, result.message, progress=completed / max(1, target), metrics={"completed_runs": completed, "target_runs": target})
            else:
                _emit(progress_callback, result.message, progress=completed / max(1, target))
        return TaskResult(status="succeeded", message="DOE batch finished.", metrics={"completed_runs": completed, "target_runs": target})
