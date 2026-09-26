from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path

os.environ.setdefault("QT_LOGGING_RULES", "qt.text.font.db=false")

from design_variables import load_variable_specs, save_variable_specs
from ..config import AppConfig, RuntimeSettings, SolverPaths, WorkspacePaths
from ..core import ActiveLearningService, ParetoService, SobolService
from ..models import TaskResult, TaskUpdate
from ..runner import RunnerAPI

try:
    from PySide6.QtCore import QObject, Signal, Qt, QUrl
    from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QStackedWidget,
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
        "log_title": "Task activity",
        "show_log": "Show activity",
        "hide_log": "Hide activity",
        "status_ready": "Ready to run",
        "status_running": "Task running",
        "status_stopping": "Stopping task...",
        "status_done": "Task finished",
        "status_failed": "Task failed",
        "app_subtitle": "Configure → Sample → Learn → Explore → Export",
        "environment_intro": "Set project paths and check external tools before launching a run.",
        "doe_intro": "Choose a sample count and geometry bounds. Engineering thresholds are available below.",
        "active_learning_intro": "Train, compare, and continue optimization from the current workspace.",
        "pareto_intro": "Compute a front, then select a point by index, target, or curve position.",
        "sobol_intro": "Measure sensitivity at a fixed blade count using the current surrogate.",
        "export_intro": "Export selected cases to a reviewable directory.",
        "show_advanced": "Show engineering thresholds",
        "hide_advanced": "Hide engineering thresholds",
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
        "solver_group": "Solver and templates",
        "project_root": "Project Root",
        "powershell": "PowerShell",
        "geometry_script": "Geometry Script",
        "cfturbo_exe": "CFturbo Executable",
        "turbogrid_exe": "TurboGrid Executable",
        "cfx_bin_dir": "CFX Bin Dir",
        "template_cfx": "BaseModel.cfx",
        "template_cse": "Extract_Results.cse",
        "base_cft": "Base .cft",
        "batch_template": "CFturbo Batch Template",
        "turbogrid_template": "TurboGrid State Template",
        "training_csv": "Training CSV",
        "cfx_cores": "CFX Cores",
        "validate_environment": "Validate Environment",
        "doe_group": "DOE Sampling and Execution",
        "doe_initial_samples": "Initial LHS Samples",
        "doe_target_samples": "Target Samples",
        "doe_runs_dir": "DOE Runs Dir",
        "engineering_defaults_group": "Default Engineering Parameters",
        "default_invalid_flow_g_s": "Invalid Flow (g/s)",
        "default_discard_flow_g_s": "Discard Flow (g/s)",
        "default_boundary_flow_g_s": "Boundary Flow (g/s)",
        "default_min_efficiency": "Minimum Efficiency",
        "default_min_power": "Minimum Power",
        "optimization_outlet_static_pressure_pa": "DOE / Active Learning Fixed Outlet Static Pressure (Pa)",
        "operating_point_pressure_tolerance_pa": "Observed-Data Pressure Band (Pa)",
        "default_min_d2_d1s_gap": "Minimum d2-d1s Gap",
        "default_max_le_sweep_diff": "Maximum LE Sweep Diff",
        "default_max_exit_angle_diff": "Maximum Exit Angle Diff",
        "default_min_rake_te_s_nbl_9": "Minimum rake_te_s (nBl=9)",
        "default_min_rake_te_s_nbl_10": "Minimum rake_te_s (nBl=10)",
        "default_min_rake_te_s_nbl_11": "Minimum rake_te_s (nBl=11)",
        "default_min_rake_te_s_nbl_12": "Minimum rake_te_s (nBl=12)",
        "variable_ranges_group": "Geometry Variable Ranges",
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
        "run_nsga2_only": "Run DOE-only NSGA-II baseline",
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
        "tab_sobol": "Sobol",
        "sobol_group": "Sobol Sensitivity Analysis",
        "sobol_fixed_nbl": "Fixed nBl",
        "sobol_base_n": "Base Samples (N)",
        "sobol_use_al_samples": "Include active-learning samples",
        "sobol_tag": "Output Tag",
        "run_sobol": "Run Sobol Analysis",
        "stop_task": "Stop Current Task",
        "task_failed": "Task Failed",
        "unhandled_error": "Unhandled Error",
        "invalid_ranges": "Invalid variable ranges",
    },
    "zh": {
        "window_title": f"低温离心压缩机几何参数优化 {APP_VERSION}",
        "app_title": f"几何参数主动学习/NSGA-2优化 {APP_VERSION}",
        "log_placeholder": "任务日志和结构化更新会显示在这里……",
        "log_title": "任务动态",
        "show_log": "展开动态",
        "hide_log": "收起动态",
        "status_ready": "等待任务",
        "status_running": "任务运行中",
        "status_stopping": "正在停止任务……",
        "status_done": "任务已完成",
        "status_failed": "任务失败",
        "app_subtitle": "配置环境 → DOE 采样 → 主动学习 → 结果分析 → 导出",
        "environment_intro": "先设置工程路径并校验外部工具，再启动计算任务。",
        "doe_intro": "设置样本量和几何变量范围；工程阈值可在下方展开。",
        "active_learning_intro": "在当前工作区训练、对照并继续优化。",
        "pareto_intro": "计算前沿后，可按索引、目标值或曲线位置选点。",
        "sobol_intro": "基于当前代理模型，在固定叶片数下分析灵敏度。",
        "export_intro": "将选定案例导出到便于检查的目录。",
        "show_advanced": "展开工程阈值",
        "hide_advanced": "收起工程阈值",
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
        "solver_group": "求解器与模板",
        "project_root": "项目根目录",
        "powershell": "PowerShell",
        "geometry_script": "几何脚本",
        "cfturbo_exe": "CFturbo 可执行文件",
        "turbogrid_exe": "TurboGrid 可执行文件",
        "cfx_bin_dir": "CFX 可执行目录",
        "template_cfx": "BaseModel.cfx",
        "template_cse": "Extract_Results.cse",
        "base_cft": "基础 .cft",
        "batch_template": "CFturbo 批处理模板",
        "turbogrid_template": "TurboGrid 状态模板",
        "training_csv": "训练数据 CSV",
        "cfx_cores": "CFX 核心数",
        "validate_environment": "校验环境",
        "doe_group": "DOE 采样与执行",
        "doe_initial_samples": "初始 LHS 样本数",
        "doe_target_samples": "目标样本数",
        "doe_runs_dir": "DOE 运行目录",
        "engineering_defaults_group": "默认工程参数",
        "default_invalid_flow_g_s": "无效流量阈值 (g/s)",
        "default_discard_flow_g_s": "丢弃流量阈值 (g/s)",
        "default_boundary_flow_g_s": "边界流量阈值 (g/s)",
        "default_min_efficiency": "最低效率",
        "default_min_power": "最低功率",
        "optimization_outlet_static_pressure_pa": "DOE / 主动学习固定出口静压 (Pa)",
        "operating_point_pressure_tolerance_pa": "观测数据工况压力带宽 (Pa)",
        "default_min_d2_d1s_gap": "最小 d2-d1s 差值",
        "default_max_le_sweep_diff": "最大前缘角差",
        "default_max_exit_angle_diff": "最大出口角差",
        "default_min_rake_te_s_nbl_9": "最小 rake_te_s (nBl=9)",
        "default_min_rake_te_s_nbl_10": "最小 rake_te_s (nBl=10)",
        "default_min_rake_te_s_nbl_11": "最小 rake_te_s (nBl=11)",
        "default_min_rake_te_s_nbl_12": "最小 rake_te_s (nBl=12)",
        "variable_ranges_group": "几何变量范围",
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
        "run_nsga2_only": "仅 DOE 的 NSGA-II 基准",
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
        "tab_sobol": "Sobol 分析",
        "sobol_group": "Sobol 灵敏度分析",
        "sobol_fixed_nbl": "固定叶片数 (nBl)",
        "sobol_base_n": "基础样本数 (N)",
        "sobol_use_al_samples": "纳入后续主动学习样本",
        "sobol_tag": "输出标签",
        "run_sobol": "运行 Sobol 分析",
        "stop_task": "停止当前任务",
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
        self.cancel_event = threading.Event()
        self.done = False

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.cancel_event.set()

    def _run(self):
        try:
            result = self._fn(self.update.emit, self.cancel_event)
            self.done = True
            self.finished.emit(result)
        except Exception:
            self.done = True
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
        self._configure_font()
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
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)
        self.resize(1260, 850)
        self.setMinimumSize(860, 600)
        self.setStyleSheet(self._style_sheet())

        self.title_label = QLabel()
        self.title_label.setObjectName("pageTitle")
        layout.addWidget(self.title_label)
        self.subtitle_label = QLabel()
        self.subtitle_label.setObjectName("subtitle")
        layout.addWidget(self.subtitle_label)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        workflow = QWidget()
        workflow_layout = QHBoxLayout(workflow)
        workflow_layout.setContentsMargins(0, 0, 0, 0)
        workflow_layout.setSpacing(0)
        self.navigation = QListWidget()
        self.navigation.setObjectName("workflowNavigation")
        self.navigation.setFixedWidth(172)
        self.navigation.setSpacing(4)
        self.pages = QStackedWidget()
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        workflow_layout.addWidget(self.navigation)
        workflow_layout.addWidget(self.pages, 1)
        self.splitter.addWidget(workflow)

        activity = QWidget()
        activity_layout = QVBoxLayout(activity)
        activity_layout.setContentsMargins(0, 0, 0, 0)
        activity_layout.setSpacing(8)
        activity_header = QHBoxLayout()
        self.activity_label = QLabel()
        self.activity_label.setObjectName("sectionTitle")
        activity_header.addWidget(self.activity_label)
        activity_header.addStretch(1)
        self.log_toggle = QPushButton()
        self.log_toggle.setObjectName("quietButton")
        self.log_toggle.clicked.connect(self._toggle_log)
        activity_header.addWidget(self.log_toggle)
        activity_layout.addLayout(activity_header)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        activity_layout.addWidget(self.log)
        self.splitter.addWidget(activity)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        layout.addWidget(self.splitter, 1)

        task_row = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setObjectName("statusLabel")
        task_row.addWidget(self.status_label)
        self.stop_button = QPushButton()
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self._stop_current_tasks)
        self.stop_button.setEnabled(False)
        task_row.addStretch(1)
        task_row.addWidget(self.stop_button)
        layout.addLayout(task_row)

        self._build_environment_tab()
        self._build_doe_tab()
        self._build_active_learning_tab()
        self._build_pareto_tab()
        self._build_sobol_tab()
        self._build_export_tab()
        self._apply_language()
        self._set_status("status_ready")
        self.splitter.setSizes([620, 150])

    @staticmethod
    def _style_sheet() -> str:
        return """
            QMainWindow { background: #f5f7fa; }
            QWidget { color: #203149; font-size: 13px; }
            QLabel#pageTitle { font-size: 22px; font-weight: 700; padding: 2px 0 8px; }
            QLabel#subtitle { color: #60738b; padding-bottom: 8px; }
            QLabel#pageIntro { color: #60738b; font-size: 14px; padding: 5px 0 11px; }
            QLabel#sectionTitle { font-size: 13px; font-weight: 700; color: #52647d; }
            QLabel#statusLabel { color: #52647d; font-weight: 600; }
            QListWidget#workflowNavigation { background: #eaf0f6; border: none;
                border-radius: 9px 0 0 9px; padding: 12px 7px; outline: none; }
            QListWidget#workflowNavigation::item { color: #52647d; padding: 13px 12px;
                border-radius: 6px; }
            QListWidget#workflowNavigation::item:selected { background: #ffffff;
                color: #145caa; font-weight: 700; }
            QStackedWidget { border: 1px solid #dce4ed; background: #ffffff;
                border-radius: 0 9px 9px 0; }
            QScrollArea, QScrollArea > QWidget > QWidget { background: #ffffff; border: none; }
            QGroupBox { background: #ffffff; border: 1px solid #dce4ed;
                        border-radius: 9px; margin-top: 16px; padding: 18px 14px 12px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px;
                               color: #284a73; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
                background: #ffffff; border: 1px solid #cbd6e2; border-radius: 6px;
                padding: 6px 8px; selection-background-color: #176cc2; }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
            QPlainTextEdit:focus { border: 1px solid #176cc2; }
            QPlainTextEdit { font-family: Consolas, monospace; }
            QPushButton { background: #176cc2; color: #ffffff; border: none;
                          border-radius: 6px; padding: 8px 14px; font-weight: 600; }
            QPushButton:hover { background: #115baf; }
            QPushButton:disabled { background: #dce4ed; color: #7a899a; }
            QPushButton#quietButton { background: transparent; color: #176cc2; }
            QPushButton#quietButton:hover { background: #eaf0f6; }
            QPushButton#stopButton { background: #fff0ed; color: #a72d23; }
            QPushButton#stopButton:hover { background: #ffe0da; }
            QSplitter::handle { background: #e2e9f0; height: 5px; }
        """

    def _add_page(self, page: QWidget, key: str):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(page)
        self._tab_indexes[key] = self.pages.addWidget(scroll)
        self.navigation.addItem("")
        if self.navigation.currentRow() < 0:
            self.navigation.setCurrentRow(0)

    @staticmethod
    def _configure_font():
        if sys.platform != "win32":
            return
        font_file = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
        if font_file.is_file():
            QFontDatabase.addApplicationFont(str(font_file))
            QApplication.instance().setFont(QFont("Microsoft YaHei UI", 10))

    def _page_intro(self, layout: QVBoxLayout, key: str) -> QLabel:
        label = QLabel()
        label.setObjectName("pageIntro")
        label.setWordWrap(True)
        layout.addWidget(label)
        self._form_labels[key] = label
        return label

    def _toggle_log(self):
        self.log.setHidden(not self.log.isHidden())
        self.log_toggle.setText(self.tr("show_log") if self.log.isHidden() else self.tr("hide_log"))

    def _toggle_advanced(self):
        self.engineering_defaults_group.setHidden(not self.engineering_defaults_group.isHidden())
        self.advanced_button.setText(
            self.tr("show_advanced") if self.engineering_defaults_group.isHidden()
            else self.tr("hide_advanced")
        )

    def _set_status(self, key: str):
        self._status_key = key
        self.status_label.setText(self.tr(key))

    def tr(self, key: str) -> str:
        return translate(self._language, key)

    def _register_field(self, field: PathField) -> PathField:
        self._translatable_fields.append(field)
        return field

    def _add_form_row(self, form: QFormLayout, key: str, field: QWidget):
        label = QLabel()
        self._form_labels[key] = label
        form.addRow(label, field)

    def _double_spin(self, value: float, decimals: int, minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _build_environment_tab(self):
        page = QWidget()
        wrapper = QVBoxLayout(page)
        self._page_intro(wrapper, "environment_intro")

        self.environment_group = QGroupBox()
        workspace_form = QFormLayout(self.environment_group)
        self.solver_group = QGroupBox()
        solver_form = QFormLayout(self.solver_group)
        self.language_selector = QComboBox()
        self.language_selector.addItem("")
        self.language_selector.addItem("")
        self.language_selector.currentIndexChanged.connect(self._on_language_changed)
        self.project_root = self._register_field(DirectoryField(str(self.config.workspace.project_root)))
        self.powershell = self._register_field(PathField(str(self.config.solver.powershell_exe)))
        self.geometry_script = self._register_field(PathField(str(self.config.solver.geometry_script_path)))
        self.cfturbo_exe = self._register_field(PathField(str(self.config.solver.cfturbo_exe)))
        self.turbogrid_exe = self._register_field(PathField(str(self.config.solver.turbogrid_exe)))
        self.cfx_bin_dir = self._register_field(DirectoryField(str(self.config.solver.cfx_bin_dir)))
        self.template_cfx = self._register_field(PathField(str(self.config.solver.template_cfx)))
        self.template_cse = self._register_field(PathField(str(self.config.solver.template_cse)))
        self.base_cft = self._register_field(PathField(str(self.config.solver.base_cft)))
        self.batch_template = self._register_field(PathField(str(self.config.solver.cft_batch_template)))
        self.turbogrid_template = self._register_field(PathField(str(self.config.solver.turbogrid_template)))
        self.training_csv = self._register_field(PathField(str(self.config.workspace.training_csv)))
        self.cfx_cores = QSpinBox()
        self.cfx_cores.setRange(1, 128)
        self.cfx_cores.setValue(self.config.runtime.cfx_cores)

        self._add_form_row(workspace_form, "language", self.language_selector)
        self._add_form_row(workspace_form, "project_root", self.project_root)
        self._add_form_row(workspace_form, "training_csv", self.training_csv)
        self._add_form_row(workspace_form, "cfx_cores", self.cfx_cores)
        self._add_form_row(solver_form, "powershell", self.powershell)
        self._add_form_row(solver_form, "geometry_script", self.geometry_script)
        self._add_form_row(solver_form, "cfturbo_exe", self.cfturbo_exe)
        self._add_form_row(solver_form, "turbogrid_exe", self.turbogrid_exe)
        self._add_form_row(solver_form, "cfx_bin_dir", self.cfx_bin_dir)
        self._add_form_row(solver_form, "template_cfx", self.template_cfx)
        self._add_form_row(solver_form, "template_cse", self.template_cse)
        self._add_form_row(solver_form, "base_cft", self.base_cft)
        self._add_form_row(solver_form, "batch_template", self.batch_template)
        self._add_form_row(solver_form, "turbogrid_template", self.turbogrid_template)
        wrapper.addWidget(self.environment_group)
        wrapper.addWidget(self.solver_group)

        self.validate_button = QPushButton()
        self.validate_button.clicked.connect(self._validate_environment)
        wrapper.addWidget(self.validate_button, alignment=Qt.AlignmentFlag.AlignLeft)
        wrapper.addStretch(1)
        self._add_page(page, "tab_environment")

    def _build_doe_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self._page_intro(layout, "doe_intro")
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

        row = QHBoxLayout()
        self.recover_button = QPushButton()
        self.recover_button.clicked.connect(self._recover_doe_runs)
        self.start_doe_button = QPushButton()
        self.start_doe_button.clicked.connect(self._start_doe)
        row.addWidget(self.recover_button)
        row.addWidget(self.start_doe_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.engineering_defaults_group = QGroupBox()
        defaults_form = QFormLayout(self.engineering_defaults_group)
        runtime = self.config.runtime
        self.default_invalid_flow_g_s = self._double_spin(runtime.default_invalid_flow_g_s, 4, 0.0, 1_000_000.0, 0.1)
        self.default_discard_flow_g_s = self._double_spin(runtime.default_discard_flow_g_s, 6, 0.0, 1_000_000.0, 0.0001)
        self.default_boundary_flow_g_s = self._double_spin(runtime.default_boundary_flow_g_s, 4, 0.0, 1_000_000.0, 0.1)
        self.default_min_efficiency = self._double_spin(runtime.default_min_efficiency, 4, 0.0, 1.0, 0.01)
        self.default_min_power = self._double_spin(runtime.default_min_power, 4, 0.0, 1_000_000.0, 1.0)
        self.optimization_outlet_static_pressure_pa = self._double_spin(runtime.optimization_outlet_static_pressure_pa, 4, 0.0, 1_000_000.0, 0.1)
        self.operating_point_pressure_tolerance_pa = self._double_spin(runtime.operating_point_pressure_tolerance_pa, 4, 0.0, 1_000_000.0, 0.05)
        self.default_min_d2_d1s_gap = self._double_spin(runtime.default_min_d2_d1s_gap, 6, -1_000.0, 1_000.0, 0.001)
        self.default_max_le_sweep_diff = self._double_spin(runtime.default_max_le_sweep_diff, 3, 0.0, 180.0, 1.0)
        self.default_max_exit_angle_diff = self._double_spin(runtime.default_max_exit_angle_diff, 3, 0.0, 180.0, 1.0)
        self.default_min_rake_te_s_nbl_9 = self._double_spin(runtime.default_min_rake_te_s_nbl_9, 3, -180.0, 180.0, 1.0)
        self.default_min_rake_te_s_nbl_10 = self._double_spin(runtime.default_min_rake_te_s_nbl_10, 3, -180.0, 180.0, 1.0)
        self.default_min_rake_te_s_nbl_11 = self._double_spin(runtime.default_min_rake_te_s_nbl_11, 3, -180.0, 180.0, 1.0)
        self.default_min_rake_te_s_nbl_12 = self._double_spin(runtime.default_min_rake_te_s_nbl_12, 3, -180.0, 180.0, 1.0)
        self._add_form_row(form, "optimization_outlet_static_pressure_pa", self.optimization_outlet_static_pressure_pa)
        self._add_form_row(defaults_form, "default_invalid_flow_g_s", self.default_invalid_flow_g_s)
        self._add_form_row(defaults_form, "default_discard_flow_g_s", self.default_discard_flow_g_s)
        self._add_form_row(defaults_form, "default_boundary_flow_g_s", self.default_boundary_flow_g_s)
        self._add_form_row(defaults_form, "default_min_efficiency", self.default_min_efficiency)
        self._add_form_row(defaults_form, "default_min_power", self.default_min_power)
        self._add_form_row(defaults_form, "operating_point_pressure_tolerance_pa", self.operating_point_pressure_tolerance_pa)
        self._add_form_row(defaults_form, "default_min_d2_d1s_gap", self.default_min_d2_d1s_gap)
        self._add_form_row(defaults_form, "default_max_le_sweep_diff", self.default_max_le_sweep_diff)
        self._add_form_row(defaults_form, "default_max_exit_angle_diff", self.default_max_exit_angle_diff)
        self._add_form_row(defaults_form, "default_min_rake_te_s_nbl_9", self.default_min_rake_te_s_nbl_9)
        self._add_form_row(defaults_form, "default_min_rake_te_s_nbl_10", self.default_min_rake_te_s_nbl_10)
        self._add_form_row(defaults_form, "default_min_rake_te_s_nbl_11", self.default_min_rake_te_s_nbl_11)
        self._add_form_row(defaults_form, "default_min_rake_te_s_nbl_12", self.default_min_rake_te_s_nbl_12)
        self.advanced_button = QPushButton()
        self.advanced_button.setObjectName("quietButton")
        self.advanced_button.clicked.connect(self._toggle_advanced)
        layout.addWidget(self.advanced_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.engineering_defaults_group)
        self.engineering_defaults_group.setVisible(False)

        self.variable_ranges_group = QGroupBox()
        range_layout = QGridLayout(self.variable_ranges_group)
        self.range_header_name = QLabel()
        self.range_header_min = QLabel()
        self.range_header_max = QLabel()
        range_layout.addWidget(self.range_header_name, 0, 0)
        range_layout.addWidget(self.range_header_min, 0, 1)
        range_layout.addWidget(self.range_header_max, 0, 2)

        geometry_specs = [
            spec for spec in self.variable_specs if spec.get("role") == "geometry"
        ]
        for row_idx, spec in enumerate(geometry_specs, start=1):
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

        layout.addStretch(1)
        self._add_page(page, "tab_doe")

    def _build_active_learning_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self._page_intro(layout, "active_learning_intro")
        self.active_learning_group = QGroupBox()
        form = QFormLayout(self.active_learning_group)
        self.al_runs_dir = self._register_field(DirectoryField(str(self.config.workspace.active_learning_runs_dir)))
        self.al_iters = QSpinBox()
        self.al_iters.setRange(1, 100)
        self.al_iters.setValue(self.config.runtime.active_learning_additional_iters)
        self._add_form_row(form, "al_runs_dir", self.al_runs_dir)
        self._add_form_row(form, "al_iters", self.al_iters)
        layout.addWidget(self.active_learning_group)

        row = QGridLayout()
        self.resume_button = QPushButton()
        self.resume_button.clicked.connect(self._resume_checkpoint)
        self.train_button = QPushButton()
        self.train_button.clicked.connect(self._train_surrogate)
        self.run_nsga2_button = QPushButton()
        self.run_nsga2_button.clicked.connect(self._run_nsga2_only)
        self.run_active_learning_button = QPushButton()
        self.run_active_learning_button.clicked.connect(self._run_active_learning)
        row.addWidget(self.resume_button, 0, 0)
        row.addWidget(self.train_button, 0, 1)
        row.addWidget(self.run_nsga2_button, 1, 0)
        row.addWidget(self.run_active_learning_button, 1, 1)
        layout.addLayout(row)
        layout.addStretch(1)
        self._add_page(page, "tab_active_learning")

    def _build_pareto_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self._page_intro(layout, "pareto_intro")
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
        layout.addStretch(1)
        self._add_page(page, "tab_pareto")

    def _build_sobol_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self._page_intro(layout, "sobol_intro")
        self.sobol_group = QGroupBox()
        form = QFormLayout(self.sobol_group)
        self.sobol_fixed_nbl = QSpinBox()
        nbl_spec = next((spec for spec in self.variable_specs if spec["name"] == "nBl"), {"lower": 9, "upper": 12})
        self.sobol_fixed_nbl.setRange(int(round(float(nbl_spec["lower"]))), int(round(float(nbl_spec["upper"]))))
        self.sobol_fixed_nbl.setValue(self.config.runtime.sobol_fixed_nbl)
        self.sobol_base_n = QSpinBox()
        self.sobol_base_n.setRange(128, 100000)
        self.sobol_base_n.setSingleStep(128)
        self.sobol_base_n.setValue(self.config.runtime.sobol_base_n)
        self.sobol_use_al_samples = QCheckBox()
        self.sobol_use_al_samples.setChecked(self.config.runtime.sobol_use_al_samples)
        self.sobol_tag = QLineEdit(self.config.runtime.sobol_tag)
        self._add_form_row(form, "sobol_fixed_nbl", self.sobol_fixed_nbl)
        self._add_form_row(form, "sobol_base_n", self.sobol_base_n)
        self._add_form_row(form, "sobol_use_al_samples", self.sobol_use_al_samples)
        self._add_form_row(form, "sobol_tag", self.sobol_tag)
        layout.addWidget(self.sobol_group)

        self.run_sobol_button = QPushButton()
        self.run_sobol_button.clicked.connect(self._run_sobol)
        layout.addWidget(self.run_sobol_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        self._add_page(page, "tab_sobol")

    def _build_export_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self._page_intro(layout, "export_intro")
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
        self._add_page(page, "tab_export")

    def _apply_language(self):
        self.setWindowTitle(self.tr("window_title"))
        self.title_label.setText(self.tr("app_title"))
        self.subtitle_label.setText(self.tr("app_subtitle"))
        self.log.setPlaceholderText(self.tr("log_placeholder"))
        self.activity_label.setText(self.tr("log_title"))
        self.log_toggle.setText(self.tr("show_log") if self.log.isHidden() else self.tr("hide_log"))
        self.advanced_button.setText(
            self.tr("show_advanced") if self.engineering_defaults_group.isHidden()
            else self.tr("hide_advanced")
        )
        if hasattr(self, "_status_key"):
            self.status_label.setText(self.tr(self._status_key))

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
        self.solver_group.setTitle(self.tr("solver_group"))
        self.doe_group.setTitle(self.tr("doe_group"))
        self.engineering_defaults_group.setTitle(self.tr("engineering_defaults_group"))
        self.variable_ranges_group.setTitle(self.tr("variable_ranges_group"))
        self.active_learning_group.setTitle(self.tr("active_learning_group"))
        self.pareto_group.setTitle(self.tr("pareto_group"))
        self.sobol_group.setTitle(self.tr("sobol_group"))
        self.export_group.setTitle(self.tr("export_group"))

        self.range_header_name.setText(self.tr("variable_name"))
        self.range_header_min.setText(self.tr("lower_bound"))
        self.range_header_max.setText(self.tr("upper_bound"))

        self.validate_button.setText(self.tr("validate_environment"))
        self.recover_button.setText(self.tr("recover_runs"))
        self.start_doe_button.setText(self.tr("start_doe"))
        self.resume_button.setText(self.tr("resume_checkpoint"))
        self.train_button.setText(self.tr("train_surrogate"))
        self.run_nsga2_button.setText(self.tr("run_nsga2_only"))
        self.run_active_learning_button.setText(self.tr("run_active_learning"))
        self.compute_pareto_button.setText(self.tr("compute_pareto"))
        self.query_button.setText(self.tr("run_query"))
        self.run_sobol_button.setText(self.tr("run_sobol"))
        self.export_button.setText(self.tr("export_cases"))
        self.stop_button.setText(self.tr("stop_task"))

        for key, index in self._tab_indexes.items():
            self.navigation.item(index).setText(self.tr(key))

    def _on_language_changed(self, index: int):
        self._language = "zh" if index == 0 else "en"
        self._apply_language()

    def _serialize_variable_specs(self) -> list[dict]:
        serialized = []
        for spec in self.variable_specs:
            if spec.get("role") != "geometry":
                # Operating conditions such as P_out are configured as one
                # fixed runtime value, not edited as DOE design ranges.
                serialized.append(dict(spec))
                continue
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
            nsga2_surrogate_pareto_csv=Path("nsga2_surrogate_pareto.csv"),
            nsga2_surrogate_summary_json=Path("nsga2_surrogate_summary.json"),
        )
        solver = SolverPaths(
            powershell_exe=Path(self.powershell.text()),
            geometry_script_path=Path(self.geometry_script.text()),
            cfturbo_exe=Path(self.cfturbo_exe.text()),
            turbogrid_exe=Path(self.turbogrid_exe.text()),
            cfx_bin_dir=Path(self.cfx_bin_dir.text()),
            template_cfx=Path(self.template_cfx.text()),
            template_cse=Path(self.template_cse.text()),
            base_cft=Path(self.base_cft.text()),
            cft_batch_template=Path(self.batch_template.text()),
            turbogrid_template=Path(self.turbogrid_template.text()),
        )
        runtime = RuntimeSettings(
            cfx_cores=self.cfx_cores.value(),
            cfx_residual_threshold=self.config.runtime.cfx_residual_threshold,
            cfx_max_extra_iterations=self.config.runtime.cfx_max_extra_iterations,
            cfx_restart_chunk=self.config.runtime.cfx_restart_chunk,
            rpm=self.config.runtime.rpm,
            mass_flow=self.config.runtime.mass_flow,
            alpha0=self.config.runtime.alpha0,
            optimization_outlet_static_pressure_pa=self.optimization_outlet_static_pressure_pa.value(),
            operating_point_pressure_tolerance_pa=self.operating_point_pressure_tolerance_pa.value(),
            default_invalid_flow_g_s=self.default_invalid_flow_g_s.value(),
            default_discard_flow_g_s=self.default_discard_flow_g_s.value(),
            default_boundary_flow_g_s=self.default_boundary_flow_g_s.value(),
            default_min_efficiency=self.default_min_efficiency.value(),
            default_min_power=self.default_min_power.value(),
            default_min_d2_d1s_gap=self.default_min_d2_d1s_gap.value(),
            default_max_le_sweep_diff=self.default_max_le_sweep_diff.value(),
            default_max_exit_angle_diff=self.default_max_exit_angle_diff.value(),
            default_min_rake_te_s_nbl_9=self.default_min_rake_te_s_nbl_9.value(),
            default_min_rake_te_s_nbl_10=self.default_min_rake_te_s_nbl_10.value(),
            default_min_rake_te_s_nbl_11=self.default_min_rake_te_s_nbl_11.value(),
            default_min_rake_te_s_nbl_12=self.default_min_rake_te_s_nbl_12.value(),
            doe_initial_samples=self.doe_initial_samples.value(),
            doe_target_samples=self.doe_target_samples.value(),
            active_learning_additional_iters=self.al_iters.value(),
            pareto_geom_safe_threshold=self.geom_safe.value(),
            sobol_fixed_nbl=self.sobol_fixed_nbl.value(),
            sobol_base_n=self.sobol_base_n.value(),
            sobol_use_al_samples=self.sobol_use_al_samples.isChecked(),
            sobol_tag=self.sobol_tag.text().strip(),
        )
        if runtime.default_discard_flow_g_s > runtime.default_boundary_flow_g_s:
            raise ValueError("default_discard_flow_g_s must not exceed default_boundary_flow_g_s")
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
        self.stop_button.setEnabled(True)
        self._set_status("status_running")
        self.log.setVisible(True)
        worker.start()

    def _cleanup_workers(self):
        self._workers = [worker for worker in self._workers if not worker.done]
        self.stop_button.setEnabled(bool(self._workers))

    def _stop_current_tasks(self):
        for worker in self._workers:
            if not worker.done:
                worker.stop()
        self.log.appendPlainText("[running] Stop requested; waiting for external processes to terminate...")
        self.stop_button.setEnabled(False)
        self._set_status("status_stopping")

    def _handle_update(self, payload):
        if isinstance(payload, TaskUpdate):
            line = f"[{payload.status}] {payload.message}"
            if payload.metrics:
                line += f" | {payload.metrics}"
        else:
            line = str(payload)
        self.log.appendPlainText(line)

    def _handle_result(self, result):
        self._cleanup_workers()
        if isinstance(result, TaskResult):
            if hasattr(self, "_set_status"):
                self._set_status("status_failed" if result.status == "failed" else "status_done")
            self.log.appendPlainText(f"[{result.status}] {result.message}")
            if result.metrics:
                self.log.appendPlainText(str(result.metrics))
            if result.artifacts:
                self.log.appendPlainText(str(result.artifacts))
            if result.status == "succeeded" and result.artifacts.get("hv_plot"):
                plot = Path(result.artifacts["hv_plot"])
                if plot.is_file():
                    if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(plot.resolve()))):
                        self.log.appendPlainText(f"HV 图片已保存，请手动打开：{plot}")
                else:
                    self.log.appendPlainText(f"HV 图片未生成，请检查日志：{plot}")
            if result.status == "failed":
                QMessageBox.warning(self, self.tr("task_failed"), result.message)
        else:
            self.log.appendPlainText(str(result))

    def _handle_failure(self, text):
        self._cleanup_workers()
        self._set_status("status_failed")
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
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return
        self._run_worker(lambda callback, cancel_event: RunnerAPI(config).recover_runs(progress_callback=callback, cancel_event=cancel_event))

    def _start_doe(self):
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return
        self._run_worker(lambda callback, cancel_event: RunnerAPI(config).run_doe_batch(progress_callback=callback, cancel_event=cancel_event))

    def _resume_checkpoint(self):
        result = self._with_config(lambda config: ActiveLearningService(config).resume_from_checkpoint())
        if result is not None:
            self._handle_result(result)

    def _train_surrogate(self):
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return
        self._run_worker(lambda callback, cancel_event: ActiveLearningService(config).train_surrogate(progress_callback=callback))

    def _run_nsga2_only(self):
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return
        self._run_worker(lambda callback, cancel_event: ActiveLearningService(config).run_nsga2_only(progress_callback=callback))

    def _run_active_learning(self):
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return
        self._run_worker(
            lambda callback, cancel_event: ActiveLearningService(config).run_active_learning_iteration(
                config.runtime.active_learning_additional_iters,
                progress_callback=callback,
                cancel_event=cancel_event,
            )
        )

    def _compute_pareto(self):
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return
        self._run_worker(
            lambda callback, cancel_event: (
                callback("Computing Pareto front..."),
                ParetoService(config).compute_pareto_front(),
            )[1]
        )

    def _query_pareto(self):
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return

        def run(callback, cancel_event):
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

    def _run_sobol(self):
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return
        self._run_worker(lambda callback, cancel_event: SobolService(config).run_analysis(progress_callback=callback))

    def _export_cases(self):
        config = self._with_config(lambda cfg: cfg)
        if config is None:
            return
        self._run_worker(
            lambda callback, cancel_event: ParetoService(config).export_cases(
                top_n=self.export_top_n.value(),
                force=True,
                base_cft=self.base_cft.text() or None,
                cft_batch_template=self.batch_template.text() or None,
                progress_callback=callback,
            )
        )

    def closeEvent(self, event):
        for worker in self._workers:
            if not worker.done:
                worker.stop()
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
