from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ..config import AppConfig
from ..legacy import active_learning_module
from ..models import TaskResult, TaskUpdate


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
        if meta_path.exists():
            checkpoint_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            completed_iters = int(checkpoint_meta.get("completed_iters", 0))
        metrics = {"completed_iters": completed_iters, "pool_size": 0}
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
        _emit(progress_callback, "running", "Loading training data...")
        legacy = self.legacy
        df = legacy.load_and_clean_data(str(self.config.workspace.training_csv))
        x_pool, x_test, y_pool, y_test, w_pool, w_test = legacy.split_with_fixed_testset(
            df,
            test_csv=str(self.config.workspace.project_root / "fixed_test_set.csv"),
        )
        x_pool = legacy.snap_discrete_vars(x_pool)
        x_test = legacy.snap_discrete_vars(x_test)

        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        x_pool_norm = scaler_x.fit_transform(x_pool)
        y_pool_surr = y_pool[:, legacy.SURROGATE_OUTPUT_IDX]
        y_pool_norm = scaler_y.fit_transform(y_pool_surr)
        y_test_surr = y_test[:, legacy.SURROGATE_OUTPUT_IDX]

        stratify_labels = w_pool if len(set(w_pool)) > 1 else None
        x_tr, x_val, y_tr, y_val, w_tr, w_val = train_test_split(
            x_pool_norm,
            y_pool_norm,
            w_pool,
            test_size=0.2,
            random_state=42,
            stratify=stratify_labels,
        )

        _emit(progress_callback, "running", "Training surrogate network...")
        model, history = legacy.train_regressor(x_tr, y_tr, w_tr, x_val, y_val, w_val, save_path=str(self.config.workspace.best_regressor_pth))
        joblib.dump(scaler_x, self.config.workspace.scaler_x_pkl)
        joblib.dump(scaler_y, self.config.workspace.scaler_y_pkl)

        y_pred = legacy.deterministic_predict(model, scaler_x.transform(x_test), scaler_y)
        metrics = {
            "train_samples": int(len(x_pool)),
            "test_samples": int(len(x_test)),
            "epochs": int(len(history)),
            "mse_eff": float(mean_squared_error(y_test_surr[:, 0], y_pred[:, 0])),
            "mse_pr": float(mean_squared_error(y_test_surr[:, 1], y_pred[:, 1])),
            "mse_mf": float(mean_squared_error(y_test_surr[:, 2], y_pred[:, 2])),
        }
        return TaskResult(
            status="succeeded",
            message="Surrogate training completed.",
            metrics=metrics,
            artifacts={
                "model": str(self.config.workspace.best_regressor_pth),
                "scaler_x": str(self.config.workspace.scaler_x_pkl),
                "scaler_y": str(self.config.workspace.scaler_y_pkl),
            },
        )

    def run_active_learning_iteration(self, additional_iters: int = 1, progress_callback=None) -> TaskResult:
        checkpoint = int(self.legacy.get_resume_iter())
        target = checkpoint + max(1, int(additional_iters))
        _emit(progress_callback, "running", f"Starting active learning until iteration {target}...")
        emitter = _LineEmitter(progress_callback)
        with contextlib.redirect_stdout(emitter), contextlib.redirect_stderr(emitter):
            self.legacy.main_multiobjective_active_learning(max_al_iters=target)
        hv_csv = self.config.workspace.hv_history_csv
        metrics = {"completed_iters": target}
        if hv_csv.exists():
            metrics["hv_history_csv"] = str(hv_csv)
        return TaskResult(
            status="succeeded",
            message=f"Active learning completed through iteration {target}.",
            metrics=metrics,
            artifacts={
                "hv_history": str(hv_csv),
                "hv_plot": str(self.config.workspace.hv_plot_png),
                "pool_checkpoint": str(self.config.workspace.pool_checkpoint_csv),
            },
        )
