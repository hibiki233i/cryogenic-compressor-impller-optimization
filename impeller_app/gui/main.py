from __future__ import annotations

import sys
import traceback
from pathlib import Path

from ..config import AppConfig, RuntimeSettings, SolverPaths, WorkspacePaths
from ..core import ActiveLearningService, ParetoService
from ..models import TaskResult, TaskUpdate
from ..runner import RunnerAPI

try:
    from PySide6.QtCore import QObject, Signal, Qt
    from PySide6.QtWidgets import (
        QApplication,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PySide6 is required to launch the desktop GUI. Install requirements-gui.txt first.") from exc


class Worker(QObject):
    update = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def start(self):
        import threading

        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            result = self._fn(self.update.emit)
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class PathField(QWidget):
    def __init__(self, text: str):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(text)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        layout.addWidget(self.edit)
        layout.addWidget(browse)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select file")
        if path:
            self.edit.setText(path)

    def text(self) -> str:
        return self.edit.text().strip()


class DirectoryField(PathField):
    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if path:
            self.edit.setText(path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Impeller Optimization Workbench")
        self.resize(1320, 860)
        self.config = AppConfig().resolved()
        self._workers = []

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        title = QLabel("Impeller Active Learning / NSGA-II Desktop Console")
        title.setStyleSheet("font-size: 22px; font-weight: 700; padding: 6px 0;")
        layout.addWidget(title)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Task logs and structured updates will appear here...")
        self.log.setMaximumBlockCount(5000)
        layout.addWidget(self.log, 1)

        self._build_environment_tab()
        self._build_doe_tab()
        self._build_active_learning_tab()
        self._build_pareto_tab()
        self._build_export_tab()

    def _build_environment_tab(self):
        page = QWidget()
        wrapper = QVBoxLayout(page)

        solver_box = QGroupBox("Environment Configuration")
        solver_form = QFormLayout(solver_box)
        self.project_root = DirectoryField(str(self.config.workspace.project_root))
        self.powershell = PathField(str(self.config.solver.powershell_exe))
        self.geometry_script = PathField(str(self.config.solver.geometry_script_path))
        self.cfx_bin_dir = DirectoryField(str(self.config.solver.cfx_bin_dir))
        self.template_cfx = PathField(str(self.config.solver.template_cfx))
        self.template_cse = PathField(str(self.config.solver.template_cse))
        self.training_csv = PathField(str(self.config.workspace.training_csv))
        solver_form.addRow("Project Root", self.project_root)
        solver_form.addRow("PowerShell", self.powershell)
        solver_form.addRow("Geometry Script", self.geometry_script)
        solver_form.addRow("CFX Bin Dir", self.cfx_bin_dir)
        solver_form.addRow("BaseModel.cfx", self.template_cfx)
        solver_form.addRow("Extract_Results.cse", self.template_cse)
        solver_form.addRow("Training CSV", self.training_csv)
        wrapper.addWidget(solver_box)

        validate = QPushButton("Validate Environment")
        validate.clicked.connect(self._validate_environment)
        wrapper.addWidget(validate, alignment=Qt.AlignmentFlag.AlignLeft)
        self.tabs.addTab(page, "Environment")

    def _build_doe_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("DOE Sampling and Execution")
        form = QFormLayout(box)
        self.doe_initial_samples = QSpinBox()
        self.doe_initial_samples.setRange(1, 10000)
        self.doe_initial_samples.setValue(self.config.runtime.doe_initial_samples)
        self.doe_target_samples = QSpinBox()
        self.doe_target_samples.setRange(1, 10000)
        self.doe_target_samples.setValue(self.config.runtime.doe_target_samples)
        self.doe_runs_dir = DirectoryField(str(self.config.workspace.doe_runs_dir))
        form.addRow("Initial LHS Samples", self.doe_initial_samples)
        form.addRow("Target Samples", self.doe_target_samples)
        form.addRow("DOE Runs Dir", self.doe_runs_dir)
        layout.addWidget(box)

        row = QHBoxLayout()
        recover_btn = QPushButton("Recover Runs")
        recover_btn.clicked.connect(self._recover_doe_runs)
        run_btn = QPushButton("Start DOE")
        run_btn.clicked.connect(self._start_doe)
        row.addWidget(recover_btn)
        row.addWidget(run_btn)
        row.addStretch(1)
        layout.addLayout(row)
        self.tabs.addTab(page, "DOE")

    def _build_active_learning_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("Active Learning Optimization")
        form = QFormLayout(box)
        self.al_runs_dir = DirectoryField(str(self.config.workspace.active_learning_runs_dir))
        self.al_iters = QSpinBox()
        self.al_iters.setRange(1, 100)
        self.al_iters.setValue(self.config.runtime.active_learning_additional_iters)
        self.cfx_cores = QSpinBox()
        self.cfx_cores.setRange(1, 128)
        self.cfx_cores.setValue(self.config.runtime.cfx_cores)
        form.addRow("ActiveLearning Runs Dir", self.al_runs_dir)
        form.addRow("Additional Iterations", self.al_iters)
        form.addRow("CFX Cores", self.cfx_cores)
        layout.addWidget(box)

        row = QHBoxLayout()
        resume_btn = QPushButton("Resume Checkpoint")
        resume_btn.clicked.connect(self._resume_checkpoint)
        train_btn = QPushButton("Train Surrogate")
        train_btn.clicked.connect(self._train_surrogate)
        run_btn = QPushButton("Run Active Learning")
        run_btn.clicked.connect(self._run_active_learning)
        row.addWidget(resume_btn)
        row.addWidget(train_btn)
        row.addWidget(run_btn)
        row.addStretch(1)
        layout.addLayout(row)
        self.tabs.addTab(page, "Active Learning")

    def _build_pareto_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("Pareto Front Query and Inverse Design")
        form = QFormLayout(box)
        self.geom_safe = QDoubleSpinBox()
        self.geom_safe.setDecimals(2)
        self.geom_safe.setRange(0.0, 1.0)
        self.geom_safe.setSingleStep(0.05)
        self.geom_safe.setValue(self.config.runtime.pareto_geom_safe_threshold)
        self.front_index = QSpinBox()
        self.front_index.setRange(-1, 9999)
        self.front_index.setValue(-1)
        self.curve_frac = QDoubleSpinBox()
        self.curve_frac.setDecimals(2)
        self.curve_frac.setRange(0.0, 1.0)
        self.curve_frac.setValue(0.5)
        self.target_eff = QDoubleSpinBox()
        self.target_eff.setDecimals(4)
        self.target_eff.setRange(0.0, 5.0)
        self.target_eff.setValue(0.0)
        self.target_pr = QDoubleSpinBox()
        self.target_pr.setDecimals(4)
        self.target_pr.setRange(0.0, 10.0)
        self.target_pr.setValue(0.0)
        form.addRow("Geom Safe Threshold", self.geom_safe)
        form.addRow("Front Index (-1 disables)", self.front_index)
        form.addRow("Curve Fraction", self.curve_frac)
        form.addRow("Target Efficiency (0 disables)", self.target_eff)
        form.addRow("Target Pressure Ratio (0 disables)", self.target_pr)
        layout.addWidget(box)

        row = QHBoxLayout()
        build_btn = QPushButton("Compute Pareto Front")
        build_btn.clicked.connect(self._compute_pareto)
        query_btn = QPushButton("Run Query")
        query_btn.clicked.connect(self._query_pareto)
        row.addWidget(build_btn)
        row.addWidget(query_btn)
        row.addStretch(1)
        layout.addLayout(row)
        self.tabs.addTab(page, "Pareto")

    def _build_export_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("Case Export and Artifact Review")
        form = QFormLayout(box)
        self.export_top_n = QSpinBox()
        self.export_top_n.setRange(1, 100)
        self.export_top_n.setValue(3)
        self.export_dir = DirectoryField(str(self.config.workspace.pareto_export_dir))
        self.base_cft = PathField("")
        self.batch_template = PathField("")
        form.addRow("Top N Cases", self.export_top_n)
        form.addRow("Export Dir", self.export_dir)
        form.addRow("Base .cft (optional)", self.base_cft)
        form.addRow("Batch Template (optional)", self.batch_template)
        layout.addWidget(box)

        export_btn = QPushButton("Export Cases")
        export_btn.clicked.connect(self._export_cases)
        layout.addWidget(export_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        self.tabs.addTab(page, "Export")

    def _current_config(self) -> AppConfig:
        root = Path(self.project_root.text() or ".")
        workspace = WorkspacePaths(
            project_root=root,
            doe_runs_dir=Path(self.doe_runs_dir.text() or "Runs"),
            active_learning_runs_dir=Path(self.al_runs_dir.text() or "ActiveLearning_Runs"),
            training_csv=Path(self.training_csv.text() or "Compressor_Training_Data.csv"),
            extra_samples_json=Path("extra_samples.json"),
            pool_checkpoint_csv=Path("al_training_pool_checkpoint.csv"),
            checkpoint_meta_json=Path("al_checkpoint_meta.json"),
            failed_points_npy=Path("failed_points.npy"),
            hv_history_csv=Path("hv_history.csv"),
            hv_plot_png=Path("hv_convergence.png"),
            scaler_x_pkl=Path("scaler_X.pkl"),
            scaler_y_pkl=Path("scaler_Y.pkl"),
            best_regressor_pth=Path("best_regressor.pth"),
            geom_warn_clf_pkl=Path("geometry_warning_clf.pkl"),
            pareto_front_csv=Path("pareto_front_points.csv"),
            pareto_plot_png=Path("pareto_front.png"),
            pareto_selection_json=Path("pareto_selected_point.json"),
            pareto_engineering_csv=Path("pareto_engineering_ranked.csv"),
            pareto_engineering_json=Path("pareto_engineering_report.json"),
            pareto_export_dir=Path(self.export_dir.text() or "pareto_cft_cases"),
        )
        solver = SolverPaths(
            powershell_exe=Path(self.powershell.text()),
            geometry_script_path=Path(self.geometry_script.text()),
            cfx_bin_dir=Path(self.cfx_bin_dir.text()),
            template_cfx=Path(self.template_cfx.text()),
            template_cse=Path(self.template_cse.text()),
        )
        runtime = RuntimeSettings(
            cfx_cores=self.cfx_cores.value(),
            rpm=self.config.runtime.rpm,
            mass_flow=self.config.runtime.mass_flow,
            alpha0=self.config.runtime.alpha0,
            doe_initial_samples=self.doe_initial_samples.value(),
            doe_target_samples=self.doe_target_samples.value(),
            active_learning_additional_iters=self.al_iters.value(),
            pareto_geom_safe_threshold=self.geom_safe.value(),
        )
        return AppConfig(solver=solver, workspace=workspace, runtime=runtime).resolved()

    def _run_worker(self, fn):
        worker = Worker(fn)
        worker.update.connect(self._handle_update)
        worker.finished.connect(self._handle_result)
        worker.failed.connect(self._handle_failure)
        self._workers.append(worker)
        worker.start()

    def _handle_update(self, payload):
        if isinstance(payload, TaskUpdate):
            line = f"[{payload.status}] {payload.message}"
            if payload.metrics:
                line += f" | {payload.metrics}"
        else:
            line = str(payload)
        self.log.appendPlainText(line)

    def _handle_result(self, result):
        if isinstance(result, TaskResult):
            self.log.appendPlainText(f"[{result.status}] {result.message}")
            if result.metrics:
                self.log.appendPlainText(str(result.metrics))
            if result.artifacts:
                self.log.appendPlainText(str(result.artifacts))
            if result.status == "failed":
                QMessageBox.warning(self, "Task Failed", result.message)
        else:
            self.log.appendPlainText(str(result))

    def _handle_failure(self, text):
        self.log.appendPlainText(text)
        QMessageBox.critical(self, "Unhandled Error", text)

    def _validate_environment(self):
        config = self._current_config()
        result = RunnerAPI(config).validate_environment()
        self._handle_result(result)

    def _recover_doe_runs(self):
        result = RunnerAPI(self._current_config()).recover_runs()
        self._handle_result(result)

    def _start_doe(self):
        config = self._current_config()
        self._run_worker(lambda callback: RunnerAPI(config).run_doe_batch(progress_callback=callback))

    def _resume_checkpoint(self):
        result = ActiveLearningService(self._current_config()).resume_from_checkpoint()
        self._handle_result(result)

    def _train_surrogate(self):
        config = self._current_config()
        self._run_worker(lambda callback: ActiveLearningService(config).train_surrogate(progress_callback=callback))

    def _run_active_learning(self):
        config = self._current_config()
        self._run_worker(lambda callback: ActiveLearningService(config).run_active_learning_iteration(config.runtime.active_learning_additional_iters, progress_callback=callback))

    def _compute_pareto(self):
        result = ParetoService(self._current_config()).compute_pareto_front()
        self._handle_result(result)

    def _query_pareto(self):
        selection = {}
        if self.front_index.value() >= 0:
            selection["front_index"] = self.front_index.value()
        elif self.target_eff.value() > 0 and self.target_pr.value() > 0:
            selection["target_eff"] = self.target_eff.value()
            selection["target_pr"] = self.target_pr.value()
        else:
            selection["curve_frac"] = self.curve_frac.value()
        result = ParetoService(self._current_config()).query_front(selection)
        self._handle_result(result)

    def _export_cases(self):
        config = self._current_config()
        base_cft = self.base_cft.text() or None
        batch_template = self.batch_template.text() or None
        result = ParetoService(config).export_cases(
            top_n=self.export_top_n.value(),
            force=True,
            base_cft=base_cft,
            cft_batch_template=batch_template,
        )
        self._handle_result(result)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
