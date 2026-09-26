from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ..config import AppConfig
from ..legacy import active_learning_module
from ..models import TaskResult, TaskUpdate


class _NeverCancelledError(Exception):
    pass


def _emit(progress_callback, status: str, message: str, progress=None, metrics=None, artifacts=None):
    if progress_callback:
        progress_callback(
            TaskUpdate(
                status=status,
                message=message,
                progress=progress,
                metrics=metrics or {},
                artifacts=artifacts or {},
            )
        )


class _LineEmitter(io.TextIOBase):
    def __init__(self, callback):
        self._callback = callback
        self._buffer = ""

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                _emit(self._callback, "running", line)
        return len(text)

    def flush(self):
        if self._buffer.strip():
            _emit(self._callback, "running", self._buffer.strip())
            self._buffer = ""


class ActiveLearningService:
    def __init__(self, config: AppConfig):
        self.config = config.resolved()
        self._legacy = None

    @property
    def legacy(self):
        if self._legacy is None:
            os.environ["IMPELLER_DESIGN_VARIABLES_PATH"] = str(self.config.workspace.design_variables_json)
            self._legacy = active_learning_module()
            self._legacy.configure_runtime(**self.config.legacy_overrides())
        return self._legacy

    def resume_from_checkpoint(self) -> TaskResult:
        meta_path = self.config.workspace.checkpoint_meta_json
        pool_path = self.config.workspace.pool_checkpoint_csv
        completed_iters = 0
        effective_resume_iter = 0
        in_progress_iter = None
        if meta_path.exists():
            checkpoint_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            completed_iters = int(checkpoint_meta.get("completed_iters", 0))
            in_progress_iter = checkpoint_meta.get("in_progress_iter")
            if in_progress_iter is not None:
                in_progress_iter = int(in_progress_iter)
            effective_resume_iter = max(completed_iters, (in_progress_iter - 1) if in_progress_iter is not None else 0)
        metrics = {
            "completed_iters": completed_iters,
            "effective_resume_iter": effective_resume_iter,
            "in_progress_iter": in_progress_iter,
            "pool_size": 0,
            "pool_checkpoint_path": str(pool_path),
            "pool_checkpoint_exists": pool_path.exists(),
            "checkpoint_meta_path": str(meta_path),
            "checkpoint_meta_exists": meta_path.exists(),
        }
        if pool_path.exists():
            metrics["pool_size"] = sum(1 for _ in pool_path.open("r", encoding="utf-8")) - 1
        if meta_path.exists():
            metrics["checkpoint_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        return TaskResult(
            status="succeeded",
            message=f"Recovered checkpoint at iteration {completed_iters}.",
            metrics=metrics,
            artifacts={"checkpoint_meta": str(meta_path)},
        )

    def train_surrogate(self, progress_callback=None) -> TaskResult:
        try:
            _emit(progress_callback, "running", "Loading training data...")
            legacy = self.legacy
            df = (legacy._load_nsga2_training_dataframe(True) if self.config.workspace.pool_checkpoint_csv.exists()
                  else legacy.load_and_clean_data(str(self.config.workspace.training_csv)))
            x_pool, x_test, y_pool, y_test, w_pool, w_test = legacy.split_with_fixed_testset(
                df,
                test_csv=str(self.config.workspace.project_root / "fixed_test_set.csv"),
            )
            x_pool = legacy.snap_discrete_vars(x_pool)
            x_test = legacy.snap_discrete_vars(x_test)

            scaler_x = legacy.make_categorical_nbl_scaler(
                legacy.VAR_NAMES, legacy.L_BOUNDS, legacy.U_BOUNDS
            )
            scaler_y = MinMaxScaler()
            y_pool_surr = y_pool[:, legacy.SURROGATE_OUTPUT_IDX]
            y_test_surr = y_test[:, legacy.SURROGATE_OUTPUT_IDX]

            train_idx, val_idx, _ = legacy.split_training_validation_indices(
                x_pool, w_pool,
            )
            scaler_x.fit(x_pool[train_idx])
            scaler_y.fit(y_pool_surr[train_idx])
            x_pool_norm = scaler_x.transform(x_pool)
            y_pool_norm = scaler_y.transform(y_pool_surr)
            x_tr, x_val = x_pool_norm[train_idx], x_pool_norm[val_idx]
            y_tr, y_val = y_pool_norm[train_idx], y_pool_norm[val_idx]
            nbl_idx = legacy.VAR_NAMES.index("nBl")
            train_weights, weight_diagnostics = legacy.compute_density_nbl_sample_weights(
                x_tr,
                x_pool[train_idx, nbl_idx],
                k_neighbors=legacy.CFG.density_k_neighbors,
                min_weight=legacy.CFG.sample_weight_min,
                max_weight=legacy.CFG.sample_weight_max,
            )

            _emit(progress_callback, "running", "Training surrogate network...")
            model, history = legacy.train_regressor(
                x_tr,
                y_tr,
                (y_pool_surr[train_idx, 2] < legacy.BOUNDARY_FLOW_G_S).astype(float),
                x_val,
                y_val,
                (y_pool_surr[val_idx, 2] < legacy.BOUNDARY_FLOW_G_S).astype(float),
                save_path=str(self.config.workspace.best_regressor_pth),
                sample_weights_train=train_weights,
                random_seed=1000,
            )
            joblib.dump(scaler_x, self.config.workspace.scaler_x_pkl)
            joblib.dump(scaler_y, self.config.workspace.scaler_y_pkl)
            import pandas as pd
            from .artifacts import write_model_manifest, stage_info
            frame = pd.DataFrame(x_pool, columns=legacy.VAR_NAMES)
            for j, name in enumerate(legacy.SURROGATE_OUTPUT_NAMES):
                frame[name] = y_pool_surr[:, j]
            write_model_manifest(self.config.workspace.best_regressor_pth,
                self.config.workspace.scaler_x_pkl, self.config.workspace.scaler_y_pkl,
                frame, legacy.VAR_NAMES, legacy.SURROGATE_OUTPUT_NAMES,
                stage_info(self.config.workspace.checkpoint_meta_json)["stage_id"])


            y_pred = legacy.deterministic_predict(model, scaler_x.transform(x_test), scaler_y)
            fixed_metrics = legacy.regression_metrics(
                y_test_surr,
                y_pred,
                prefix="fixed_test_",
            )
            _emit(progress_callback, "running", f"Running {legacy.CFG.cv_folds}-fold validation...")
            cv_rows, cv_summary = legacy.run_kfold_surrogate_validation(
                X_raw=x_pool,
                Y_surr=y_pool_surr,
                W_boundary=w_pool,
                al_iter=0,
                cfg=legacy.CFG,
            )
            if cv_rows:
                legacy.upsert_csv_records(
                    str(self.config.workspace.cv_fold_metrics_csv),
                    cv_rows,
                    key_columns=["iter", "fold"],
                )
            legacy.upsert_csv_records(
                str(self.config.workspace.surrogate_metrics_csv),
                {
                    "iter": 0,
                    "mode": "standalone_training",
                    "train_pool_samples_before_cfd": int(len(x_pool)),
                    "fixed_test_samples": int(len(x_test)),
                    "training_epochs": int(len(history)),
                    "recorded_at_utc": legacy.utc_timestamp(),
                    **fixed_metrics,
                    **cv_summary,
                    **weight_diagnostics,
                },
                key_columns=["iter"],
            )
            metrics = {
                "train_samples": int(len(x_pool)),
                "test_samples": int(len(x_test)),
                "epochs": int(len(history)),
                "mse_eff": fixed_metrics["fixed_test_mse_eff"],
                "mse_pr": fixed_metrics["fixed_test_mse_pr"],
                "mse_mf": fixed_metrics["fixed_test_mse_mf"],
                **fixed_metrics,
                **cv_summary,
            }
            return TaskResult(
                status="succeeded",
                message="Surrogate training completed.",
                metrics=metrics,
                artifacts={
                    "model": str(self.config.workspace.best_regressor_pth),
                    "scaler_x": str(self.config.workspace.scaler_x_pkl),
                    "scaler_y": str(self.config.workspace.scaler_y_pkl),
                    "validation_metrics": str(self.config.workspace.surrogate_metrics_csv),
                    "cv_fold_metrics": str(self.config.workspace.cv_fold_metrics_csv),
                },
            )
        except ValueError as exc:
            return TaskResult(
                status="failed",
                message=str(exc),
                artifacts={
                    "training_csv": str(self.config.workspace.training_csv),
                    "fixed_test_set": str(self.config.workspace.project_root / "fixed_test_set.csv"),
                },
            )

    def run_active_learning_iteration(
        self,
        additional_iters: int = 1,
        progress_callback=None,
        cancel_event=None,
    ) -> TaskResult:
        if cancel_event is not None and cancel_event.is_set():
            return TaskResult(
                status="canceled",
                message="Active learning canceled before start.",
            )
        legacy = self.legacy
        cancellation_error = getattr(
            legacy, "ActiveLearningCancelled", _NeverCancelledError
        )
        try:
            checkpoint = int(legacy.get_resume_iter())
            target = checkpoint + max(1, int(additional_iters))
            _emit(progress_callback, "running", f"Starting active learning until iteration {target}...")
            emitter = _LineEmitter(progress_callback)
            with contextlib.redirect_stdout(emitter), contextlib.redirect_stderr(emitter):
                run_summary = legacy.main_multiobjective_active_learning(
                    max_al_iters=target,
                    cancel_event=cancel_event,
                )
            hv_csv = self.config.workspace.hv_history_csv
            completed_iters = int(legacy.get_resume_iter())
            stopped_early = False
            stop_reason = ""
            if isinstance(run_summary, dict):
                completed_iters = int(
                    run_summary.get("completed_iters", completed_iters)
                )
                stopped_early = bool(run_summary.get("stopped_early", False))
                stop_reason = str(run_summary.get("stop_reason", ""))
            metrics = {
                "completed_iters": completed_iters,
                "target_iters": target,
                "stopped_early": stopped_early,
            }
            if isinstance(run_summary, dict):
                for key in (
                    "rolling_true_hv_gain",
                    "hv_stagnation_triggered",
                    "hv_stagnation_window",
                    "min_true_hv_gain",
                    "hv_stagnation_action",
                    "model_reliability",
                    "local_online_max_nrmse",
                    "local_online_max_abs_bias_norm",
                    "local_flow_brier",
                    "local_flow_ece",
                ):
                    if key in run_summary:
                        metrics[key] = run_summary[key]
            if hv_csv.exists():
                metrics["hv_history_csv"] = str(hv_csv)
            return TaskResult(
                status="succeeded",
                message=(
                    f"主动学习已在第 {completed_iters} 轮暂停审查：{stop_reason}"
                    if stopped_early
                    else f"主动学习已完成至第 {completed_iters} 轮。"
                ),
                metrics=metrics,
                artifacts={
                    "hv_history": str(hv_csv),
                    "hv_plot": str(self.config.workspace.hv_plot_png),
                    "pool_checkpoint": str(self.config.workspace.pool_checkpoint_csv),
                    "validation_metrics": str(self.config.workspace.surrogate_metrics_csv),
                    "cv_fold_metrics": str(self.config.workspace.cv_fold_metrics_csv),
                    "fixed_test_predictions": str(self.config.workspace.fixed_test_predictions_csv),
                    "query_validation": str(self.config.workspace.al_query_validation_csv),
                    "failure_records": str(self.config.workspace.failure_records_csv),
                },
            )
        except cancellation_error as exc:
            _emit(progress_callback, "canceled", str(exc))
            return TaskResult(
                status="canceled",
                message=str(exc),
                artifacts={
                    "pool_checkpoint": str(self.config.workspace.pool_checkpoint_csv),
                    "query_validation": str(self.config.workspace.al_query_validation_csv),
                },
            )
        except ValueError as exc:
            return TaskResult(
                status="failed",
                message=str(exc),
                artifacts={
                    "training_csv": str(self.config.workspace.training_csv),
                    "fixed_test_set": str(self.config.workspace.project_root / "fixed_test_set.csv"),
                    "pool_checkpoint": str(self.config.workspace.pool_checkpoint_csv),
                },
            )

    def run_nsga2_only(self, progress_callback=None, use_pool_checkpoint: bool = False) -> TaskResult:
        try:
            _emit(progress_callback, "running", ("Starting NSGA-II from DOE + AL..." if use_pool_checkpoint else "Starting DOE-only NSGA-II baseline (AL samples excluded)..."))
            emitter = _LineEmitter(progress_callback)
            with contextlib.redirect_stdout(emitter), contextlib.redirect_stderr(emitter):
                summary = self.legacy.run_nsga2_only_from_lhs(
                    output_csv=str(self.config.workspace.nsga2_surrogate_pareto_csv),
                    summary_json=str(self.config.workspace.nsga2_surrogate_summary_json),
                    use_pool_checkpoint=use_pool_checkpoint,
                )
            metrics = {
                "train_samples": int(summary.get("train_samples", 0)),
                "test_samples": int(summary.get("test_samples", 0)),
                "front_size": int(summary.get("front_size", 0)),
                "surrogate_hv": float(summary.get("surrogate_hv", 0.0)),
                "mse_eff": float(summary.get("mse_eff", 0.0)),
                "mse_pr": float(summary.get("mse_pr", 0.0)),
                "mse_mf": float(summary.get("mse_mf", 0.0)),
            }
            return TaskResult(
                status="succeeded",
                message="NSGA-II surrogate optimization completed.",
                metrics=metrics,
                artifacts={
                    "surrogate_pareto_csv": str(self.config.workspace.nsga2_surrogate_pareto_csv),
                    "summary_json": str(self.config.workspace.nsga2_surrogate_summary_json),
                    "model": str(summary.get("model", self.config.workspace.best_regressor_pth)),
                    "scaler_x": str(summary.get("scaler_x", self.config.workspace.scaler_x_pkl)),
                    "scaler_y": str(summary.get("scaler_y", self.config.workspace.scaler_y_pkl)),
                },
            )
        except ValueError as exc:
            return TaskResult(
                status="failed",
                message=str(exc),
                artifacts={
                    "training_csv": str(self.config.workspace.training_csv),
                    "surrogate_pareto_csv": str(self.config.workspace.nsga2_surrogate_pareto_csv),
                },
            )
