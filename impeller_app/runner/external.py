from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from ..config import AppConfig
from ..models import TaskResult, TaskUpdate
from cfx_runner import run_cfx_pipeline


DOEVariableNames = [
    "d1s",
    "dH",
    "beta1hb",
    "beta1sb",
    "d2",
    "b2",
    "beta2hb",
    "beta2sb",
    "Lz",
    "t",
    "TipClear",
    "nBl",
    "rake_te_s",
    "P_out",
]
L_BOUNDS = np.array([0.34, 0.044, 70.0, 20.0, 0.45, 0.044, 40.0, 40.0, 0.185, 0.0015, 0.0007, 9.0, -25.0, 8.0])
U_BOUNDS = np.array([0.42, 0.056, 84.0, 35.0, 0.53, 0.056, 54.0, 55.0, 0.215, 0.0035, 0.0015, 12.0, -15.0, 13.0])
MIN_NORMAL = 3.6
MIN_DISCARD = 0.0001


def _emit(progress_callback, message: str, progress=None, metrics=None):
    if progress_callback:
        progress_callback(TaskUpdate(status="running", message=message, progress=progress, metrics=metrics or {}))


class RunnerAPI:
    def __init__(self, config: AppConfig):
        self.config = config.resolved()

    def validate_environment(self) -> TaskResult:
        cfg = self.config
        checks = {
            "powershell_exe": cfg.solver.powershell_exe,
            "geometry_script": cfg.solver.geometry_script_path,
            "template_cfx": cfg.solver.template_cfx,
            "template_cse": cfg.solver.template_cse,
            "training_csv": cfg.workspace.training_csv,
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
        return TaskResult(status="succeeded", message="Environment validation passed.", metrics={"checked": {k: str(v) for k, v in checks.items()}})

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
        sampler = qmc.LatinHypercube(d=len(DOEVariableNames), seed=seed)
        sample_real = qmc.scale(sampler.random(n=count), L_BOUNDS, U_BOUNDS)
        return [dict(zip(DOEVariableNames, row)) for row in sample_real]

    def _columns(self) -> list[str]:
        return DOEVariableNames + ["Efficiency", "PressureRatio", "Power", "MassFlow", "totalpressureratio", "is_boundary"]

    def _load_extra_samples(self) -> list[dict]:
        path = self.config.workspace.extra_samples_json
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return []

    def _save_extra_samples(self, payload: list[dict]) -> None:
        self.config.workspace.extra_samples_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _new_extra_sample(self) -> dict:
        norm = qmc.LatinHypercube(d=len(DOEVariableNames)).random(1)
        sample_real = qmc.scale(norm, L_BOUNDS, U_BOUNDS)[0]
        sample = dict(zip(DOEVariableNames, sample_real))
        extras = self._load_extra_samples()
        extras.append(sample)
        self._save_extra_samples(extras)
        return sample

    def _result_row_from_sample(self, sample: dict, raw_values: list[str]) -> dict | None:
        row = dict(sample)
        nbl = int(round(float(row["nBl"])))
        mass_flow = float(raw_values[3]) * nbl
        if mass_flow < MIN_DISCARD:
            return None
        row["nBl"] = nbl
        row["Efficiency"] = float(raw_values[0])
        row["PressureRatio"] = float(raw_values[1])
        row["Power"] = float(raw_values[2]) * nbl
        row["MassFlow"] = mass_flow
        row["totalpressureratio"] = float(raw_values[4])
        row["is_boundary"] = 1 if mass_flow < MIN_NORMAL else 0
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
        ]
        for name in DOEVariableNames:
            if name == "P_out":
                continue
            value = int(round(float(sample[name]))) if name == "nBl" else sample[name]
            cmd.extend([f"-{name}", str(value)])
        cmd.extend(["-mFlow", str(self.config.runtime.mass_flow), "-N_rpm", str(self.config.runtime.rpm), "-alpha0", str(self.config.runtime.alpha0)])
        return cmd

    def run_cfx_case(self, working_dir: Path, run_id: str, p_out: float, n_blades: int) -> TaskResult:
        success, payload, message = run_cfx_pipeline(
            str(working_dir),
            run_id,
            p_out=p_out,
            cores=self.config.runtime.cfx_cores,
            n_blades=n_blades,
            cfx_bin_dir=str(self.config.solver.cfx_bin_dir),
            template_cfx=str(self.config.solver.template_cfx),
            template_cse=str(self.config.solver.template_cse),
        )
        return TaskResult(status="succeeded" if success else "failed", message=message, metrics=payload or {}, artifacts={"working_dir": str(working_dir)})

    def run_doe_sample(self, index: int, sample: dict, progress_callback=None) -> TaskResult:
        run_id = f"Run_{index:03d}"
        working_dir = self.config.workspace.doe_runs_dir / run_id
        working_dir.mkdir(parents=True, exist_ok=True)
        result_txt = working_dir / "CFX_Results.txt"
        if result_txt.exists():
            row = self._read_result_file(result_txt, sample)
            if row is None:
                return TaskResult(status="failed", message=f"{run_id}: existing result diverged and was discarded.", artifacts={"working_dir": str(working_dir)})
            return TaskResult(status="succeeded", message=f"{run_id}: reused existing result.", metrics=row, artifacts={"working_dir": str(working_dir)})
        cmd = self.build_geometry_command(working_dir, sample)
        _emit(progress_callback, f"{run_id}: generating geometry and mesh...")
        ps_result = subprocess.run(cmd, capture_output=True, text=True)
        if ps_result.returncode != 0:
            return TaskResult(
                status="failed",
                message=f"{run_id}: geometry generation failed with exit code {ps_result.returncode}.",
                metrics={"stdout": ps_result.stdout, "stderr": ps_result.stderr},
                artifacts={"working_dir": str(working_dir)},
            )
        _emit(progress_callback, f"{run_id}: geometry done, launching CFX...")
        cfx_result = self.run_cfx_case(working_dir, run_id, float(sample["P_out"]), int(round(float(sample["nBl"]))))
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
            "is_boundary": 1 if float(cfx_result.metrics["MassFlow"]) < MIN_NORMAL else 0,
        }
        if float(row["MassFlow"]) < MIN_DISCARD:
            return TaskResult(status="failed", message=f"{run_id}: divergent MassFlow, discarded.", metrics=row, artifacts={"working_dir": str(working_dir)})
        self._append_training_row(row)
        return TaskResult(status="succeeded", message=f"{run_id}: DOE sample completed.", metrics=row, artifacts={"working_dir": str(working_dir)})

    def run_doe_batch(self, progress_callback=None) -> TaskResult:
        self.config.workspace.doe_runs_dir.mkdir(parents=True, exist_ok=True)
        recovery = self._recover_doe_progress()
        samples = recovery["base_samples"]
        extra_samples = recovery["extra_samples"]
        current_index = int(recovery["next_index"])
        target = int(self.config.runtime.doe_target_samples)
        completed = len(recovery["rows"])
        extra_ptr = max(0, current_index - len(samples))
        while completed < target:
            if current_index < len(samples):
                sample = dict(samples[current_index])
            elif extra_ptr < len(extra_samples):
                sample = dict(extra_samples[extra_ptr])
                extra_ptr += 1
            else:
                sample = self._new_extra_sample()
                extra_samples.append(sample)
                extra_ptr += 1
            result = self.run_doe_sample(current_index, sample, progress_callback=progress_callback)
            current_index += 1
            if result.status == "succeeded":
                completed += 1
                _emit(progress_callback, result.message, progress=completed / max(1, target), metrics={"completed_runs": completed, "target_runs": target})
            else:
                _emit(progress_callback, result.message, progress=completed / max(1, target))
        return TaskResult(status="succeeded", message="DOE batch finished.", metrics={"completed_runs": completed, "target_runs": target})
