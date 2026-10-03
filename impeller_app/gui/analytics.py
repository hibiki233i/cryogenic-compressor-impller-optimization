"""PySide6 analytical workbench with observation and audit provenance kept apart."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal, QTimer, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QBoxLayout, QComboBox, QLabel, QPushButton, QTabWidget

from ..core.analysis import AnalysisReader, correlations, blade_groups, filter_observations, finite_rows, prediction_metrics, TARGETS
from .review_widgets import AsyncReader, DataTable
from .review_charts import ReviewChart, BarChart
from .theme import PALETTE


TEXT = {
    "zh": {
        "intro": "在 GUI 内分析真实样本和已保存的预测记录。图表支持滚轮缩放、框选放大、双击复位；点击在线预测点可查看算例。",
        "refresh": "刷新分析", "reading": "正在读取…", "variables": "变量与性能", "prediction": "代理预测质量", "cv": "交叉验证", "audit": "迭代审计",
        "doe": "DOE 样本", "pool": "主动学习训练池", "all": "全部边界标志", "normal": "非边界样本", "boundary": "边界样本", "blades": "全部叶片数",
        "query": "在线 CFD 前预测", "fixed": "固定测试集预测", "latest": "最新一轮", "rounds": "全部轮次", "legacy": "未标注协议（历史）",
        "correlation": "Pearson 相关性 · 仅连续几何变量", "scatter": "变量—性能散点", "groups": "按叶片数分组", "observations": "筛选后的样本",
        "parity": "预测与真实值", "error": "预测误差", "uncertainty": "预测标准差（原始单位）", "true": "真实值", "pred": "预测值", "iteration": "迭代",
        "note": "相关性不代表因果或 Sobol 灵敏度；nBl 按类别分组，P_out 仅作为固定工况过滤。包含边界样本时不能直接视为工程可行解。图表最多显示 5,000 点，统计与导出使用完整筛选数据。",
        "pairs_note": "使用已保存的历史预测，不重跑模型。仅对与 schema v2 观测数据逐行核对且阶段匹配的记录计算误差；误差 = 预测 − 真实。在线预测和固定测试集分开评估，空值不补零。",
        "cv_note": "按文件中明确记录的 CV 协议分开查看。旧记录没有协议或阶段标记时仅作为历史审计，不与新协议合并，不代表当前模型或当前工况的重新评估。",
        "audit_note": "候选/验证原始记录，仅供审计。submitted 不等于正在运行，failed 不等于物理不可行；失败或来源不明的真实值不会进入上方预测质量统计。",
        "queries": "候选记录", "validation": "验证汇总", "sources": "候选来源分布", "source": "数据来源", "empty": "暂无符合条件的数据", "sample_n": "样本数", "verified_n": "核对通过记录", "raw_n": "原始记录", "stage": "阶段", "details": "数据诊断", "summary": "点击表格行可查看全部字段",
    },
    "en": {
        "intro": "Explore observations and saved predictions inside the GUI. Wheel/rectangle to zoom, double-click to reset; click an online prediction to inspect its case.",
        "refresh": "Refresh analysis", "reading": "Reading…", "variables": "Variables & performance", "prediction": "Surrogate quality", "cv": "Cross-validation", "audit": "Iteration audit",
        "doe": "DOE observations", "pool": "AL training pool", "all": "All boundary flags", "normal": "Non-boundary", "boundary": "Boundary", "blades": "All blade counts",
        "query": "Pre-CFD online predictions", "fixed": "Fixed-test predictions", "latest": "Latest iteration", "rounds": "All iterations", "legacy": "Unspecified protocol (historical)",
        "correlation": "Pearson r · continuous geometry only", "scatter": "Variable / performance", "groups": "Blade-count groups", "observations": "Filtered observations",
        "parity": "Predicted vs observed", "error": "Prediction error", "uncertainty": "Prediction std (original units)", "true": "Observed", "pred": "Predicted", "iteration": "Iteration",
        "note": "Correlation is not causation or Sobol sensitivity. nBl is categorical; P_out is a fixed operating-condition filter. Boundary samples are not necessarily engineering-feasible. Charts show at most 5,000 points; statistics and exports use all filtered records.",
        "pairs_note": "Saved historical predictions; no model reruns. Errors use only stage-matched records checked row by row against schema-v2 observations. Error = prediction minus observation. Online and fixed-test evaluations remain separate; missing values are not zero-filled.",
        "cv_note": "CV protocols are shown separately as recorded. Untagged protocols/stages are historical audits, not merged with newer protocols and not re-evaluations of the current model or operating condition.",
        "audit_note": "Raw candidate/validation audit. Submitted does not prove an active process; failure does not imply physical infeasibility. Failed or unverified observations do not enter the prediction-quality statistics.",
        "queries": "Candidate records", "validation": "Validation summary", "sources": "Candidate sources", "source": "Data source", "empty": "No eligible data", "sample_n": "Samples", "verified_n": "Verified records", "raw_n": "Raw records", "stage": "Stage", "details": "Data diagnostics", "summary": "Select a table row to inspect every field",
    },
}


class AnalyticsPage(QWidget):
    case_requested = Signal(str, str)

    def __init__(self, config):
        super().__init__()
        self.config = config.resolved()
        self.snapshot = None
        self.language = "zh"
        self.auto_refresh = True
        self.loader = AsyncReader(self)
        self.loader.ready.connect(self._received)
        self.loader.error.connect(self._error)
        self.loader.busy.connect(self._busy)
        self.timer = QTimer(self)
        self.timer.setInterval(15000)
        self.timer.timeout.connect(lambda: self.refresh() if self.isVisible() and self.loader._job is None else None)
        root = QVBoxLayout(self)
        self.intro = QLabel()
        self.intro.setWordWrap(True)
        root.addWidget(self.intro)
        header = QHBoxLayout()
        self.objective = QComboBox()
        for column in TARGETS.values():
            self.objective.addItem(column)
        self.objective.addItem("Power")
        self.refresh_button = QPushButton()
        self.refresh_button.clicked.connect(self.refresh)
        header.addWidget(self.objective)
        header.addStretch()
        header.addWidget(self.refresh_button)
        root.addLayout(header)
        self.context = QLabel()
        self.context.setWordWrap(True)
        self.context.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.context)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)
        self.tables = []
        variables, v = self._tab()
        row = QHBoxLayout()
        self.data_source = self._combo([("doe", "doe"), ("pool", "pool")])
        self.boundary = self._combo([("all", "all"), ("normal", "0"), ("boundary", "1")])
        self.blades = QComboBox()
        self.blades.addItem("", None)
        self.variable = QComboBox()
        for widget in (self.data_source, self.boundary, self.blades, self.variable):
            row.addWidget(widget)
        v.addLayout(row)
        self.variable_note = self._label(v)
        self.variable_charts = QBoxLayout(QBoxLayout.LeftToRight)
        self.variable_chart, self.correlation_chart = ReviewChart(), BarChart()
        self.variable_charts.addWidget(self.variable_chart, 1)
        self.variable_charts.addWidget(self.correlation_chart, 1)
        v.addLayout(self.variable_charts)
        self.variable_tables = QTabWidget()
        self.observations, self.correlation_table, self.group_table = self._table(), self._table(), self._table()
        for table in (self.observations, self.correlation_table, self.group_table):
            self.variable_tables.addTab(table, "")
        v.addWidget(self.variable_tables)
        self.tabs.addTab(variables, "")

        prediction, p = self._tab()
        row = QHBoxLayout()
        self.prediction_source = self._combo([("query", "query"), ("fixed", "fixed")])
        self.round = QComboBox()
        self.round.addItem("", "latest")
        self.round.addItem("", "all")
        row.addWidget(self.prediction_source)
        row.addWidget(self.round)
        p.addLayout(row)
        self.prediction_note = self._label(p)
        self.metrics = self._label(p)
        self.diagnostic_tabs = QTabWidget()
        self.parity, self.error_chart, self.std_chart = ReviewChart(), ReviewChart(), ReviewChart()
        for chart in (self.parity, self.error_chart, self.std_chart):
            self.diagnostic_tabs.addTab(chart, "")
            chart.point_selected.connect(self._request_case)
        p.addWidget(self.diagnostic_tabs)
        self.pair_table = self._table()
        self.pair_table.activated.connect(lambda row: self._request_case(str(row.get("run_id", ""))))
        p.addWidget(self.pair_table)
        self.tabs.addTab(prediction, "")

        cv, c = self._tab()
        self.cv_note = self._label(c)
        row = QHBoxLayout()
        self.protocol = QComboBox()
        self.metric_selector = QComboBox()
        self.metric_selector.addItems(["rmse", "mae", "r2"])
        row.addWidget(self.protocol)
        row.addWidget(self.metric_selector)
        c.addLayout(row)
        self.cv_chart = ReviewChart()
        c.addWidget(self.cv_chart)
        self.cv_table = self._table()
        c.addWidget(self.cv_table)
        self.tabs.addTab(cv, "")

        audit, a = self._tab()
        self.audit_note = self._label(a)
        self.audit_source = self._combo([("queries", "query"), ("validation", "validation")])
        a.addWidget(self.audit_source)
        self.source_chart = BarChart()
        a.addWidget(self.source_chart)
        self.audit_table = self._table()
        self.audit_table.activated.connect(lambda row: self._request_case(str(row.get("run_id", "")), force=True))
        a.addWidget(self.audit_table)
        self.tabs.addTab(audit, "")
        self.diagnostics = self._label(root)
        self.diagnostics.setTextFormat(Qt.PlainText)
        for combo in (self.objective, self.data_source, self.boundary, self.blades, self.variable, self.prediction_source, self.round, self.protocol, self.metric_selector, self.audit_source):
            combo.currentIndexChanged.connect(self._render)
        self.tabs.currentChanged.connect(self._render)
        self.set_language("zh")

    def tr(self, key):
        return TEXT[self.language][key]

    def _tab(self):
        widget = QWidget()
        return widget, QVBoxLayout(widget)

    def _label(self, layout):
        label = QLabel()
        label.setTextFormat(Qt.PlainText)
        label.setWordWrap(True)
        layout.addWidget(label)
        return label

    def _combo(self, items):
        combo = QComboBox()
        for text, data in items:
            combo.addItem(self.tr(text), data)
        return combo

    def _table(self):
        table = DataTable()
        self.tables.append(table)
        return table

    def refresh(self):
        config = self.config
        self.loader.submit(lambda: AnalysisReader(config).read())

    def _busy(self, busy):
        self.refresh_button.setEnabled(not busy)
        self.refresh_button.setText(self.tr("reading" if busy else "refresh"))

    def set_config(self, config):
        self.config = config.resolved()
        self.loader.invalidate()
        self.snapshot = None
        self._render()
        if self.isVisible():
            self.refresh()

    def _received(self, snapshot):
        self.snapshot = snapshot
        self._sync(self.variable, [(name, name) for name in snapshot.geometry_names])
        counts = sorted({int(v) for frame in snapshot.frames.values() if "nBl" in frame for v in frame.nBl.unique()})
        self._sync(self.blades, [(self.tr("blades"), None)] + [(f"nBl = {v}", v) for v in counts])
        cv = snapshot.audits.get("cv", pd.DataFrame())
        protocols = sorted(cv.get("cv_protocol", pd.Series("legacy", index=cv.index)).fillna("legacy").astype(str).unique())
        self._sync(self.protocol, [(self.tr("legacy") if name == "legacy" else name, name) for name in protocols])
        iterations = sorted({int(v) for data in snapshot.pairs.values() if "iter" in data for v in data["iter"].unique()})
        self._sync(self.round, [(self.tr("latest"), "latest"), (self.tr("rounds"), "all")] + [(str(v), v) for v in iterations])
        for table in self.tables:
            table.protected_paths = set(snapshot.sources.values())
        self._render()

    @staticmethod
    def _sync(combo, entries):
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for label, value in entries:
            combo.addItem(label, value)
        found = combo.findData(current)
        if found >= 0:
            combo.setCurrentIndex(found)
        combo.blockSignals(False)

    def _error(self, message):
        self.snapshot = None
        self._render()
        self.diagnostics.setText(message)

    def set_language(self, language):
        self.language = language
        self.intro.setText(self.tr("intro"))
        self.refresh_button.setText(self.tr("refresh"))
        for index, key in enumerate(("variables", "prediction", "cv", "audit")):
            self.tabs.setTabText(index, self.tr(key))
        for index, key in enumerate(("observations", "correlation", "groups")):
            self.variable_tables.setTabText(index, self.tr(key))
        for index, key in enumerate(("parity", "error", "uncertainty")):
            self.diagnostic_tabs.setTabText(index, self.tr(key))
        for combo, keys in ((self.data_source, ("doe", "pool")), (self.boundary, ("all", "normal", "boundary")), (self.prediction_source, ("query", "fixed")), (self.audit_source, ("queries", "validation"))):
            for index, key in enumerate(keys):
                combo.setItemText(index, self.tr(key))
        self.variable_note.setText(self.tr("note"))
        self.prediction_note.setText(self.tr("pairs_note"))
        self.cv_note.setText(self.tr("cv_note"))
        self.audit_note.setText(self.tr("audit_note"))
        for table in self.tables:
            table.set_language(language)
        self.blades.setItemText(0, self.tr("blades"))
        self.round.setItemText(0, self.tr("latest"))
        self.round.setItemText(1, self.tr("rounds"))
        if self.snapshot:
            self._received(self.snapshot)
        else:
            self._render()

    def _render(self):
        if not hasattr(self, "audit_table"):
            return
        snapshot = self.snapshot
        frame = snapshot.frames.get(self.data_source.currentData(), pd.DataFrame()) if snapshot else pd.DataFrame()
        data = filter_observations(frame, self.boundary.currentData(), self.blades.currentData())
        target, name = self.objective.currentText(), self.variable.currentText()
        self.observations.set_frame(data)
        corr = correlations(data, target, snapshot.geometry_names) if snapshot else pd.DataFrame(columns=["variable", "r", "n"])
        self.correlation_table.set_frame(corr)
        self.group_table.set_frame(blade_groups(data, target))
        self.correlation_chart.render_bars(self.tr("correlation"), corr.variable.tolist(), corr.r.tolist(), "r", symmetric=True)
        scatter = finite_rows(data, [name, target])
        points = self._points(scatter, name, target)
        self.variable_chart.scatter(self.tr("scatter") + f" · n={len(scatter)}", name, target, points)
        self._render_predictions()
        self._render_cv()
        audit = snapshot.audits.get(self.audit_source.currentData(), pd.DataFrame()) if snapshot else pd.DataFrame()
        self.audit_table.set_frame(audit)
        counts = audit.get("candidate_source", pd.Series(dtype=str)).fillna("unknown").value_counts()
        self.source_chart.render_bars(self.tr("sources"), counts.index.tolist(), counts.tolist())
        ws = self.config.workspace
        path = ws.training_csv if self.data_source.currentData() == "doe" else ws.pool_checkpoint_csv
        tab = self.tabs.currentIndex()
        if tab == 1:
            path = ws.al_query_validation_csv if self.prediction_source.currentData() == "query" else ws.fixed_test_predictions_csv
        elif tab == 2:
            path = ws.cv_fold_metrics_csv
        elif tab == 3:
            path = ws.al_query_validation_csv if self.audit_source.currentData() == "query" else ws.surrogate_metrics_csv
        context = f"{self.tr('source')}: {path}\n{self.tr('stage')}: {snapshot.stage if snapshot else '—'}"
        if tab == 0:
            context += f" · {self.tr('sample_n')}: {len(data)} · P_out = {self.config.runtime.optimization_outlet_static_pressure_pa:g} ± {self.config.runtime.operating_point_pressure_tolerance_pa:g} Pa"
        self.context.setText(context)
        self.diagnostics.setText((self.tr("details") + ":\n" + "\n".join(snapshot.issues)) if snapshot and snapshot.issues else "")
        self.diagnostics.setVisible(bool(self.diagnostics.text()))

    @staticmethod
    def _points(frame, x, y):
        if frame.empty or x not in frame or y not in frame:
            return []
        if len(frame) > 5000:
            frame = frame.iloc[np.linspace(0, len(frame) - 1, 5000, dtype=int)]
        return list(frame[[x, y]].itertuples(index=False, name=None))

    def _render_predictions(self):
        kind = self.prediction_source.currentData()
        data = self.snapshot.pairs.get(kind, pd.DataFrame()).copy() if self.snapshot else pd.DataFrame()
        if not data.empty:
            selected = self.round.currentData()
            if selected == "latest":
                data = data.loc[data["iter"] == data["iter"].max()]
            elif selected != "all":
                data = data.loc[data["iter"] == selected]
        suffix = next((key for key, value in TARGETS.items() if value == self.objective.currentText()), None)
        # Power has observed values but is not a predicted surrogate target.
        if suffix is None:
            data = data.iloc[:0]
            suffix = "eff"
        self.pair_table.set_frame(data)
        metrics = prediction_metrics(data, suffix)
        raw = len(self.snapshot.audits.get(kind, [])) if self.snapshot else 0
        self.metrics.setText(f"{self.tr('verified_n')}: {metrics['n']} / {self.tr('raw_n')}: {raw} · " + " · ".join(f"{key.upper()} = {value:.6g}" if value is not None else f"{key.upper()} = —" for key, value in metrics.items() if key != "n"))
        plot = data.iloc[np.linspace(0, len(data) - 1, min(5000, len(data)), dtype=int)].copy() if len(data) else data.copy()
        labels = plot.get("run_id", pd.Series("", index=plot.index)).astype(str).tolist() if kind == "query" else []
        self.parity.scatter(self.tr("parity"), self.tr("true"), self.tr("pred"), self._points(plot, f"true_{suffix}", f"pred_{suffix}"), labels, diagonal=True)
        if len(plot):
            plot["error"] = plot[f"pred_{suffix}"] - plot[f"true_{suffix}"]
        self.error_chart.scatter(self.tr("error") + " · pred − true", self.tr("iteration"), self.objective.currentText(), self._points(plot, "iter", "error"), labels)
        std = finite_rows(plot, ["iter", f"uncertainty_real_{suffix}"])
        if not std.empty:
            std = std.loc[std[f"uncertainty_real_{suffix}"] >= 0]
        self.std_chart.scatter(self.tr("uncertainty"), self.tr("iteration"), "std", self._points(std, "iter", f"uncertainty_real_{suffix}"), std.get("run_id", pd.Series("", index=std.index)).astype(str).tolist())

    def _render_cv(self):
        frame = self.snapshot.audits.get("cv", pd.DataFrame()).copy() if self.snapshot else pd.DataFrame()
        if not frame.empty:
            tags = frame.get("cv_protocol", pd.Series("legacy", index=frame.index)).fillna("legacy").astype(str)
            frame = frame.loc[tags.eq(self.protocol.currentData())]
        self.cv_table.set_frame(frame)
        suffix = next((key for key, value in TARGETS.items() if value == self.objective.currentText()), None)
        metric = f"{self.metric_selector.currentText()}_{suffix}"
        data = finite_rows(frame, ["iter", "fold", metric])
        series = []
        if not data.empty:
            # Keep distinct folds and any explicitly recorded stages separate.
            data["stage_id"] = data.get("stage_id", pd.Series("untagged", index=data.index)).fillna("untagged")
            for index, ((stage, fold), group) in enumerate(data.groupby(["stage_id", "fold"])):
                group = group.sort_values("iter")
                series.append((f"{stage} / fold {fold:g}", [PALETTE["accent"], PALETTE["blue"], PALETTE["warning"]][index % 3], self._points(group, "iter", metric)))
        self.cv_chart.render_series(self.tr("cv") + " · " + metric, self.tr("iteration"), self.metric_selector.currentText().upper(), series)

    def _request_case(self, run_id, force=False):
        if run_id and run_id != "nan" and (force or self.prediction_source.currentData() == "query"):
            self.case_requested.emit("active_learning", run_id)

    def showEvent(self, event):
        super().showEvent(event)
        if self.auto_refresh:
            self.timer.start()
        self.refresh()

    def resizeEvent(self, event):
        self.variable_charts.setDirection(QBoxLayout.LeftToRight if self.width() >= 950 else QBoxLayout.TopToBottom)
        super().resizeEvent(event)

    def set_auto_refresh(self, enabled):
        self.auto_refresh = enabled
        if enabled and self.isVisible():
            self.timer.start()
        else:
            self.timer.stop()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def stop(self):
        self.timer.stop()
        self.loader.close()
