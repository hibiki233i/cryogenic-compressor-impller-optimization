from __future__ import annotations

import sys
import traceback
from pathlib import Path

from design_variables import load_variable_specs, save_variable_specs
from ..config import AppConfig, RuntimeSettings, SolverPaths, WorkspacePaths
from ..core import ActiveLearningService, ParetoService
from ..models import TaskResult, TaskUpdate
from ..runner import RunnerAPI

try:
    from PySide6.QtCore import QObject, Signal, Qt
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
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


APP_VERSION = "version1.0"


TEXTS = {
    "en": {
        "window_title": f"Cryogenic Centrifugal Compressor Geometry Parameter Optimization {APP_VERSION}",
        "app_title": f"Geometry Parameter Active Learning / NSGA-2 Optimization {APP_VERSION}",
        "log_placeholder": "Task logs and structured updates will appear here...",
        "language": "Language",
        "language_zh": "Chinese",
        "language_en": "English",
        "browse": "Browse",
        "select_file": "Select file",
        "select_folder": "Select folder",
        "tab_environment": "Environment",
        "tab_doe": "DOE",
        "tab_active_learning": "Active Learning",
        "tab_pareto": "Pareto",
        "tab_export": "Export",
        "environment_group": "Environment Configuration",
        "project_root": "Project Root",
        "powershell": "PowerShell",
        "geometry_script": "Geometry Script",
        "cfx_bin_dir": "CFX Bin Dir",
        "template_cfx": "BaseModel.cfx",
        "template_cse": "Extract_Results.cse",
        "base_cft": "Base .cft",
        "batch_template": "CFturbo Batch Template",
        "training_csv": "Training CSV",
        "cfx_cores": "CFX Cores",
        "validate_environment": "Validate Environment",
        "doe_group": "DOE Sampling and Execution",
        "doe_initial_samples": "Initial LHS Samples",
        "doe_target_samples": "Target Samples",
        "doe_runs_dir": "DOE Runs Dir",
        "variable_ranges_group": "Design Variable Ranges",
        "variable_name": "Variable",
        "lower_bound": "Lower",
        "upper_bound": "Upper",
        "recover_runs": "Recover Runs",
        "start_doe": "Start DOE",
        "active_learning_group": "Active Learning Optimization",
        "al_runs_dir": "ActiveLearning Runs Dir",
        "al_iters": "Additional Iterations",
        "resume_checkpoint": "Resume Checkpoint",
        "train_surrogate": "Train Surrogate",
        "run_active_learning": "Run Active Learning",
        "pareto_group": "Pareto Front Query and Inverse Design",
        "geom_safe": "Geom Safe Threshold",
        "front_index": "Front Index (-1 disables)",
        "curve_frac": "Curve Fraction",
        "target_eff": "Target Efficiency (0 disables)",
        "target_pr": "Target Pressure Ratio (0 disables)",
        "compute_pareto": "Compute Pareto Front",
        "run_query": "Run Query",
        "export_group": "Case Export and Artifact Review",
        "export_top_n": "Top N Cases",
        "export_dir": "Export Dir",
        "export_cases": "Export Cases",
        "task_failed": "Task Failed",
        "unhandled_error": "Unhandled Error",
        "invalid_ranges": "Invalid variable ranges",
    },
    "zh": {
        "window_title": f"低温离心压缩机几何参数优化 {APP_VERSION}",
        "app_title": f"几何参数主动学习/NSGA-2优化 {APP_VERSION}",
        "log_placeholder": "任务日志和结构化更新会显示在这里……",
        "language": "语言",
        "language_zh": "中文",
        "language_en": "英文",
        "browse": "浏览",
        "select_file": "选择文件",
        "select_folder": "选择文件夹",
        "tab_environment": "环境",
        "tab_doe": "DOE",
        "tab_active_learning": "主动学习",
        "tab_pareto": "帕累托",
        "tab_export": "导出",
        "environment_group": "环境配置",
        "project_root": "项目根目录",
        "powershell": "PowerShell",
        "geometry_script": "几何脚本",
        "cfx_bin_dir": "CFX 可执行目录",
        "template_cfx": "BaseModel.cfx",
        "template_cse": "Extract_Results.cse",
        "base_cft": "基础 .cft",
        "batch_template": "CFturbo 批处理模板",
        "training_csv": "训练数据 CSV",
        "cfx_cores": "CFX 核心数",
        "validate_environment": "校验环境",
        "doe_group": "DOE 采样与执行",
        "doe_initial_samples": "初始 LHS 样本数",
        "doe_target_samples": "目标样本数",
        "doe_runs_dir": "DOE 运行目录",
        "variable_ranges_group": "设计变量范围",
        "variable_name": "变量",
        "lower_bound": "下界",
        "upper_bound": "上界",
        "recover_runs": "恢复运行结果",
        "start_doe": "启动 DOE",
        "active_learning_group": "主动学习优化",
        "al_runs_dir": "主动学习运行目录",
        "al_iters": "额外迭代次数",
        "resume_checkpoint": "恢复检查点",
        "train_surrogate": "训练代理模型",
        "run_active_learning": "运行主动学习",
        "pareto_group": "帕累托前沿查询与逆向设计",
        "geom_safe": "几何安全阈值",
        "front_index": "前沿索引（-1 表示禁用）",
        "curve_frac": "曲线分数",
        "target_eff": "目标效率（0 表示禁用）",
        "target_pr": "目标压比（0 表示禁用）",
        "compute_pareto": "计算帕累托前沿",
        "run_query": "执行查询",
        "export_group": "案例导出与产物查看",
        "export_top_n": "导出前 N 个案例",
        "export_dir": "导出目录",
        "export_cases": "导出案例",
        "task_failed": "任务失败",
        "unhandled_error": "未处理异常",
        "invalid_ranges": "变量范围无效",
    },
}


def translate(language: str, key: str) -> str:
    return TEXTS.get(language, {}).get(key) or TEXTS["en"].get(key, key)


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
    def __init__(self, text: str, dialog_title_key: str = "select_file"):
        super().__init__()
        self._language = "zh"
        self._dialog_title_key = dialog_title_key
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(text)
        self.browse_button = QPushButton()
        self.browse_button.clicked.connect(self._browse)
        layout.addWidget(self.edit)
        layout.addWidget(self.browse_button)
        self.set_language(self._language)

    def set_language(self, language: str):
        self._language = language
        self.browse_button.setText(translate(language, "browse"))

    def _dialog_title(self) -> str:
        return translate(self._language, self._dialog_title_key)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, self._dialog_title())
        if path:
            self.edit.setText(path)

    def text(self) -> str:
        return self.edit.text().strip()


class DirectoryField(PathField):
    def __init__(self, text: str):
        super().__init__(text, dialog_title_key="select_folder")

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, self._dialog_title())
        if path:
            self.edit.setText(path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_path = None
        self.config = AppConfig.load().resolved()
        self.variable_specs = load_variable_specs(self.config.workspace.design_variables_json)
        self._workers = []
        self._language = "zh"
        self._form_labels: dict[str, QLabel] = {}
        self._tab_indexes: dict[str, int] = {}
        self._translatable_fields: list[PathField] = []
        self._range_spinboxes: dict[str, tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 22px; font-weight: 700; padding: 6px 0;")
        layout.addWidget(self.title_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        layout.addWidget(self.log, 1)

        self._build_environment_tab()
        self._build_doe_tab()
        self._build_active_learning_tab()
        self._build_pareto_tab()
        self._build_export_tab()
        self._apply_language()

    def tr(self, key: str) -> str:
        return translate(self._language, key)

    def _register_field(self, field: PathField) -> PathField:
        self._translatable_fields.append(field)
        return field

    def _add_form_row(self, form: QFormLayout, key: str, field: QWidget):
        label = QLabel()
        self._form_labels[key] = label
        form.addRow(label, field)

    def _build_environment_tab(self):
        page = QWidget()
        wrapper = QVBoxLayout(page)

        self.environment_group = QGroupBox()
        solver_form = QFormLayout(self.environment_group)
        self.language_selector = QComboBox()
        self.language_selector.addItem("")
        self.language_selector.addItem("")
        self.language_selector.currentIndexChanged.connect(self._on_language_changed)
        self.project_root = self._register_field(DirectoryField(str(self.config.workspace.project_root)))
        self.powershell = self._register_field(PathField(str(self.config.solver.powershell_exe)))
        self.geometry_script = self._register_field(PathField(str(self.config.solver.geometry_script_path)))
        self.cfx_bin_dir = self._register_field(DirectoryField(str(self.config.solver.cfx_bin_dir)))
        self.template_cfx = self._register_field(PathField(str(self.config.solver.template_cfx)))
        self.template_cse = self._register_field(PathField(str(self.config.solver.template_cse)))
        self.base_cft = self._register_field(PathField(str(self.config.solver.base_cft)))
        self.batch_template = self._register_field(PathField(str(self.config.solver.cft_batch_template)))
        self.training_csv = self._register_field(PathField(str(self.config.workspace.training_csv)))
        self.cfx_cores = QSpinBox()
        self.cfx_cores.setRange(1, 128)
        self.cfx_cores.setValue(self.config.runtime.cfx_cores)

        self._add_form_row(solver_form, "language", self.language_selector)
        self._add_form_row(solver_form, "project_root", self.project_root)
        self._add_form_row(solver_form, "powershell", self.powershell)
        self._add_form_row(solver_form, "geometry_script", self.geometry_script)
        self._add_form_row(solver_form, "cfx_bin_dir", self.cfx_bin_dir)
        self._add_form_row(solver_form, "template_cfx", self.template_cfx)
        self._add_form_row(solver_form, "template_cse", self.template_cse)
        self._add_form_row(solver_form, "base_cft", self.base_cft)
        self._add_form_row(solver_form, "batch_template", self.batch_template)
        self._add_form_row(solver_form, "training_csv", self.training_csv)
        self._add_form_row(solver_form, "cfx_cores", self.cfx_cores)
        wrapper.addWidget(self.environment_group)

        self.validate_button = QPushButton()
        self.validate_button.clicked.connect(self._validate_environment)
        wrapper.addWidget(self.validate_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self._tab_indexes["tab_environment"] = self.tabs.addTab(page, "")

    def _build_doe_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.doe_group = QGroupBox()
        form = QFormLayout(self.doe_group)
        self.doe_initial_samples = QSpinBox()
        self.doe_initial_samples.setRange(1, 10000)
        self.doe_initial_samples.setValue(self.config.runtime.doe_initial_samples)
        self.doe_target_samples = QSpinBox()
        self.doe_target_samples.setRange(1, 10000)
        self.doe_target_samples.setValue(self.config.runtime.doe_target_samples)
        self.doe_runs_dir = self._register_field(DirectoryField(str(self.config.workspace.doe_runs_dir)))
        self._add_form_row(form, "doe_initial_samples", self.doe_initial_samples)
        self._add_form_row(form, "doe_target_samples", self.doe_target_samples)
        self._add_form_row(form, "doe_runs_dir", self.doe_runs_dir)
        layout.addWidget(self.doe_group)

        self.variable_ranges_group = QGroupBox()
        range_layout = QGridLayout(self.variable_ranges_group)
        self.range_header_name = QLabel()
        self.range_header_min = QLabel()
        self.range_header_max = QLabel()
        range_layout.addWidget(self.range_header_name, 0, 0)
        range_layout.addWidget(self.range_header_min, 0, 1)
        range_layout.addWidget(self.range_header_max, 0, 2)

        for row_idx, spec in enumerate(self.variable_specs, start=1):
            name_label = QLabel(spec["name"])
            min_spin = QDoubleSpinBox()
            max_spin = QDoubleSpinBox()
            decimals = int(spec.get("decimals", 4))
            min_spin.setDecimals(decimals)
            max_spin.setDecimals(decimals)
            min_spin.setRange(-1_000_000.0, 1_000_000.0)
            max_spin.setRange(-1_000_000.0, 1_000_000.0)
            if spec.get("is_integer"):
                min_spin.setSingleStep(1.0)
                max_spin.setSingleStep(1.0)
            min_spin.setValue(float(spec["lower"]))
            max_spin.setValue(float(spec["upper"]))
            range_layout.addWidget(name_label, row_idx, 0)
            range_layout.addWidget(min_spin, row_idx, 1)
            range_layout.addWidget(max_spin, row_idx, 2)
            self._range_spinboxes[spec["name"]] = (min_spin, max_spin)
        layout.addWidget(self.variable_ranges_group)

        row = QHBoxLayout()
        self.recover_button = QPushButton()
        self.recover_button.clicked.connect(self._recover_doe_runs)
        self.start_doe_button = QPushButton()
        self.start_doe_button.clicked.connect(self._start_doe)
        row.addWidget(self.recover_button)
        row.addWidget(self.start_doe_button)
        row.addStretch(1)
        layout.addLayout(row)
        self._tab_indexes["tab_doe"] = self.tabs.addTab(page, "")

    def _build_active_learning_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.active_learning_group = QGroupBox()
        form = QFormLayout(self.active_learning_group)
        self.al_runs_dir = self._register_field(DirectoryField(str(self.config.workspace.active_learning_runs_dir)))
        self.al_iters = QSpinBox()
        self.al_iters.setRange(1, 100)
        self.al_iters.setValue(self.config.runtime.active_learning_additional_iters)
        self._add_form_row(form, "al_runs_dir", self.al_runs_dir)
        self._add_form_row(form, "al_iters", self.al_iters)
        layout.addWidget(self.active_learning_group)

        row = QHBoxLayout()
        self.resume_button = QPushButton()
        self.resume_button.clicked.connect(self._resume_checkpoint)
        self.train_button = QPushButton()
        self.train_button.clicked.connect(self._train_surrogate)
        self.run_active_learning_button = QPushButton()
        self.run_active_learning_button.clicked.connect(self._run_active_learning)
        row.addWidget(self.resume_button)
        row.addWidget(self.train_button)
        row.addWidget(self.run_active_learning_button)
        row.addStretch(1)
        layout.addLayout(row)
        self._tab_indexes["tab_active_learning"] = self.tabs.addTab(page, "")

    def _build_pareto_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.pareto_group = QGroupBox()
        form = QFormLayout(self.pareto_group)
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
        self._add_form_row(form, "geom_safe", self.geom_safe)
        self._add_form_row(form, "front_index", self.front_index)
        self._add_form_row(form, "curve_frac", self.curve_frac)
        self._add_form_row(form, "target_eff", self.target_eff)
        self._add_form_row(form, "target_pr", self.target_pr)
        layout.addWidget(self.pareto_group)

        row = QHBoxLayout()
        self.compute_pareto_button = QPushButton()
        self.compute_pareto_button.clicked.connect(self._compute_pareto)
        self.query_button = QPushButton()
        self.query_button.clicked.connect(self._query_pareto)
        row.addWidget(self.compute_pareto_button)
        row.addWidget(self.query_button)
        row.addStretch(1)
        layout.addLayout(row)
        self._tab_indexes["tab_pareto"] = self.tabs.addTab(page, "")

    def _build_export_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.export_group = QGroupBox()
        form = QFormLayout(self.export_group)
        self.export_top_n = QSpinBox()
        self.export_top_n.setRange(1, 100)
        self.export_top_n.setValue(3)
        self.export_dir = self._register_field(DirectoryField(str(self.config.workspace.pareto_export_dir)))
        self._add_form_row(form, "export_top_n", self.export_top_n)
        self._add_form_row(form, "export_dir", self.export_dir)
        layout.addWidget(self.export_group)

        self.export_button = QPushButton()
        self.export_button.clicked.connect(self._export_cases)
        layout.addWidget(self.export_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self._tab_indexes["tab_export"] = self.tabs.addTab(page, "")

    def _apply_language(self):
        self.setWindowTitle(self.tr("window_title"))
        self.title_label.setText(self.tr("app_title"))
        self.log.setPlaceholderText(self.tr("log_placeholder"))

        self.language_selector.blockSignals(True)
        self.language_selector.setItemText(0, self.tr("language_zh"))
        self.language_selector.setItemText(1, self.tr("language_en"))
        self.language_selector.setCurrentIndex(0 if self._language == "zh" else 1)
        self.language_selector.blockSignals(False)

        for key, label in self._form_labels.items():
            label.setText(self.tr(key))

        for field in self._translatable_fields:
            field.set_language(self._language)

        self.environment_group.setTitle(self.tr("environment_group"))
        self.doe_group.setTitle(self.tr("doe_group"))
        self.variable_ranges_group.setTitle(self.tr("variable_ranges_group"))
        self.active_learning_group.setTitle(self.tr("active_learning_group"))
        self.pareto_group.setTitle(self.tr("pareto_group"))
        self.export_group.setTitle(self.tr("export_group"))

        self.range_header_name.setText(self.tr("variable_name"))
        self.range_header_min.setText(self.tr("lower_bound"))
        self.range_header_max.setText(self.tr("upper_bound"))

        self.validate_button.setText(self.tr("validate_environment"))
        self.recover_button.setText(self.tr("recover_runs"))
        self.start_doe_button.setText(self.tr("start_doe"))
        self.resume_button.setText(self.tr("resume_checkpoint"))
        self.train_button.setText(self.tr("train_surrogate"))
        self.run_active_learning_button.setText(self.tr("run_active_learning"))
        self.compute_pareto_button.setText(self.tr("compute_pareto"))
        self.query_button.setText(self.tr("run_query"))
        self.export_button.setText(self.tr("export_cases"))

        for key, index in self._tab_indexes.items():
            self.tabs.setTabText(index, self.tr(key))

    def _on_language_changed(self, index: int):
        self._language = "zh" if index == 0 else "en"
        self._apply_language()

    def _serialize_variable_specs(self) -> list[dict]:
        serialized = []
        for spec in self.variable_specs:
            min_spin, max_spin = self._range_spinboxes[spec["name"]]
            lower = float(min_spin.value())
            upper = float(max_spin.value())
            if spec.get("is_integer"):
                lower = float(round(lower))
                upper = float(round(upper))
            if lower >= upper:
                raise ValueError(f"{spec['name']}: {lower} >= {upper}")
            serialized.append(
                {
                    **spec,
                    "lower": lower,
                    "upper": upper,
                }
            )
        return serialized

    def _current_config(self) -> AppConfig:
        serialized_specs = self._serialize_variable_specs()
        root = Path(self.project_root.text() or ".")
        workspace = WorkspacePaths(
            project_root=root,
            doe_runs_dir=Path(self.doe_runs_dir.text() or "Runs"),
            active_learning_runs_dir=Path(self.al_runs_dir.text() or "ActiveLearning_Runs"),
            training_csv=Path(self.training_csv.text() or "Compressor_Training_Data.csv"),
            design_variables_json=Path("design_variables.json"),
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
            base_cft=Path(self.base_cft.text()),
            cft_batch_template=Path(self.batch_template.text()),
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
        return AppConfig(solver=solver, workspace=workspace, runtime=runtime)

    def _persist_current_config(self) -> AppConfig:
        self.config = self._current_config()
        resolved = self.config.resolved()
        serialized_specs = self._serialize_variable_specs()
        save_variable_specs(serialized_specs, resolved.workspace.design_variables_json)
        self.variable_specs = load_variable_specs(resolved.workspace.design_variables_json)
        self.config_path = self.config.save()
        return resolved

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
                QMessageBox.warning(self, self.tr("task_failed"), result.message)
        else:
            self.log.appendPlainText(str(result))

    def _handle_failure(self, text):
        self.log.appendPlainText(text)
        QMessageBox.critical(self, self.tr("unhandled_error"), text)

    def _with_config(self, fn):
        try:
            return fn(self._persist_current_config())
        except ValueError as exc:
            QMessageBox.warning(self, self.tr("invalid_ranges"), str(exc))
            return None

    def _validate_environment(self):
        result = self._with_config(lambda config: RunnerAPI(config).validate_environment())
        if result is not None:
            self._handle_result(result)

    def _recover_doe_runs(self):
        result = self._with_config(lambda config: RunnerAPI(config).recover_runs())
        if result is not None:
            self._handle_result(result)

    def _start_doe(self):
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return
        self._run_worker(lambda callback: RunnerAPI(config).run_doe_batch(progress_callback=callback))

    def _resume_checkpoint(self):
        result = self._with_config(lambda config: ActiveLearningService(config).resume_from_checkpoint())
        if result is not None:
            self._handle_result(result)

    def _train_surrogate(self):
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return
        self._run_worker(lambda callback: ActiveLearningService(config).train_surrogate(progress_callback=callback))

    def _run_active_learning(self):
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return
        self._run_worker(
            lambda callback: ActiveLearningService(config).run_active_learning_iteration(
                config.runtime.active_learning_additional_iters,
                progress_callback=callback,
            )
        )

    def _compute_pareto(self):
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return
        self._run_worker(
            lambda callback: (
                callback("Computing Pareto front..."),
                ParetoService(config).compute_pareto_front(),
            )[1]
        )

    def _query_pareto(self):
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return

        def run(callback):
            selection = {}
            if self.front_index.value() >= 0:
                selection["front_index"] = self.front_index.value()
            elif self.target_eff.value() > 0 and self.target_pr.value() > 0:
                selection["target_eff"] = self.target_eff.value()
                selection["target_pr"] = self.target_pr.value()
            else:
                selection["curve_frac"] = self.curve_frac.value()
            callback("Computing Pareto front and running query...")
            return ParetoService(config).query_front(selection)

        self._run_worker(run)

    def _export_cases(self):
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return
        self._run_worker(
            lambda callback: ParetoService(config).export_cases(
                top_n=self.export_top_n.value(),
                force=True,
                base_cft=self.base_cft.text() or None,
                cft_batch_template=self.batch_template.text() or None,
                progress_callback=callback,
            )
        )

    def closeEvent(self, event):
        try:
            self._persist_current_config()
        except Exception:
            self.log.appendPlainText(traceback.format_exc())
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
