"""Live results page. Background reads; all Qt updates stay on the GUI thread."""
from __future__ import annotations

from datetime import datetime
import math
import threading

from PySide6.QtCore import QObject, Signal, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QBoxLayout, QGridLayout, QLabel,
    QComboBox, QPushButton, QCheckBox, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QGroupBox,
)

from ..core.live_results import LiveResultsReader, ResultsSnapshot, ResultIssue, DISPLAY_COLUMNS
from .theme import PALETTE


TEXT = {
    "zh": {
        "intro": "每 3 秒读取已落盘结果。CFD 性能在算例完成并写入结果表后更新；HV 在迭代提交后更新。",
        "auto": "自动刷新 · 3 秒", "refresh": "立即刷新", "doe": "DOE 结果", "pool": "主动学习训练池",
        "count": "有效记录", "eff": "最高效率", "pr": "最高总压比", "excluded": "未纳入记录",
        "scatter": "已记录 CFD 样本 · 效率 / 总压比", "hv": "主动学习 · HV 历史",
        "eff_axis": "效率", "pr_axis": "总压比（total-to-total）", "iteration": "迭代",
        "true": "真实 CFD HV", "predicted": "代理预测 HV", "samples": "CFD 样本",
        "empty": "暂无可展示的数据", "missing": "文件或来源元数据尚未生成",
        "unavailable": "数据暂不可用：读取中、格式不完整或来源校验未通过",
        "source": "来源", "modified": "文件更新", "checked": "最近检查", "stage": "阶段",
        "note": "统计包含边界样本，不等同于工程可行前沿。过滤非有限值、无效边界标志和固定 P_out 压力带外记录；同一设计保留最后一条。散点最多显示 5,000 条，表格显示文件末尾 200 条（倒序，非完成时间排序）。",
        "waiting": "等待任务输出", "task": "当前任务输出", "progress_unknown": "进度未提供",
        "succeeded": "已完成", "failed": "失败", "canceled": "已取消", "running": "运行中",
        "pressure": "监视固定出口静压", "dedup": "去重", "reading": "正在读取结果…",
        "table": "最近写入的记录", "boundary": "边界标志", "flow": "流量 (g/s)", "power": "功率 (W)",
    },
    "en": {
        "intro": "Reads saved results every 3 seconds. CFD performance updates after a case is saved; HV updates after an iteration is committed.",
        "auto": "Auto refresh · 3 s", "refresh": "Refresh now", "doe": "DOE results", "pool": "Active-learning pool",
        "count": "Valid records", "eff": "Highest efficiency", "pr": "Highest total PR", "excluded": "Excluded records",
        "scatter": "Recorded CFD samples · Efficiency / Total PR", "hv": "Active learning · HV history",
        "eff_axis": "Efficiency", "pr_axis": "Total-to-total pressure ratio", "iteration": "Iteration",
        "true": "Verified CFD HV", "predicted": "Surrogate HV", "samples": "CFD samples",
        "empty": "No verified data to display", "missing": "File or provenance metadata not available yet",
        "unavailable": "Data unavailable: incomplete write, read error, or provenance validation failed",
        "source": "Source", "modified": "File updated", "checked": "Last checked", "stage": "Stage",
        "note": "Includes boundary samples; this is not an engineering-feasible front. Nonfinite values, invalid boundary flags and records outside the fixed P_out band are excluded. Duplicate designs keep the last row. Charts show up to 5,000 points; the table shows the last 200 file rows in reverse order, not completion-time order.",
        "waiting": "Waiting for task output", "task": "Current task output", "progress_unknown": "Progress not provided",
        "succeeded": "Finished", "failed": "Failed", "canceled": "Canceled", "running": "Running",
        "pressure": "Monitored fixed outlet pressure", "dedup": "Duplicates", "reading": "Reading results…",
        "table": "Recently written records", "boundary": "Boundary flag", "flow": "Mass flow (g/s)", "power": "Power (W)",
    },
}


class SnapshotJob(QObject):
    finished = Signal(object)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def start(self):
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            snapshot = LiveResultsReader(self.config).read()
        except Exception as exc:
            snapshot = ResultsSnapshot(issues=[ResultIssue("monitor", "unavailable", str(exc))])
        self.finished.emit(snapshot)


class ResultsChart(QChartView):
    def __init__(self):
        chart = QChart()
        super().__init__(chart)
        self.setBackgroundBrush(QColor(PALETTE["bg"]))
        chart.setBackgroundRoundness(6)
        chart.setBackgroundBrush(QColor(PALETTE["panel"]))
        chart.setTitleBrush(QColor(PALETTE["text"]))
        chart.legend().setLabelColor(QColor(PALETTE["muted"]))
        chart.legend().setAlignment(Qt.AlignBottom)
        chart.setAnimationOptions(QChart.NoAnimation)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMinimumSize(260, 290)

    def render_series(self, title, x_label, y_label, series_data, scatter=False):
        chart = self.chart()
        chart.removeAllSeries()
        for axis in chart.axes():
            chart.removeAxis(axis)
            axis.deleteLater()
        chart.setTitle(title)
        for name, color, points in series_data:
            if not points:
                continue
            series = QScatterSeries() if scatter else QLineSeries()
            series.setName(name)
            series.setColor(QColor(color))
            if scatter:
                series.setMarkerSize(6)
                series.setBorderColor(QColor(color))
            else:
                series.setPen(QPen(QColor(color), 2))
                if color == PALETTE["blue"]:
                    series.setPen(QPen(QColor(color), 2, Qt.DashLine))
                series.setPointsVisible(True)
            for x, y in points:
                series.append(float(x), float(y))
            chart.addSeries(series)
        chart.createDefaultAxes()
        for axis in chart.axes():
            axis.setLabelsColor(QColor(PALETTE["muted"]))
            axis.setTitleBrush(QColor(PALETTE["muted"]))
            axis.setGridLinePen(QPen(QColor(PALETTE["border"])))
            axis.setTitleText(x_label if axis.orientation() == Qt.Horizontal else y_label)
            axis.setLabelFormat("%.3g")
        chart.legend().setVisible(bool(chart.series()))


class LiveResultsPage(QWidget):
    def __init__(self, config):
        super().__init__()
        self._language = "zh"
        self._config = config.resolved()
        self._generation = 0
        self._job = None
        self._job_generation = 0
        self._stopped = False
        self._snapshot = None
        self._event = None
        self._final = False
        root = QVBoxLayout(self)
        self.intro = QLabel()
        self.intro.setWordWrap(True)
        self.intro.setObjectName("pageIntro")
        root.addWidget(self.intro)
        controls = QHBoxLayout()
        self.source = QComboBox()
        self.source.setMinimumWidth(180)
        self.source.addItems(["", ""])
        self.source.currentIndexChanged.connect(self._render)
        controls.addWidget(self.source)
        self.auto = QCheckBox()
        self.auto.setChecked(True)
        self.auto.toggled.connect(self._auto_changed)
        controls.addWidget(self.auto)
        controls.addStretch()
        self.refresh = QPushButton()
        self.refresh.clicked.connect(self.refresh_now)
        controls.addWidget(self.refresh)
        root.addLayout(controls)
        self.context = QLabel()
        self.context.setWordWrap(True)
        root.addWidget(self.context)
        self.task_group = QGroupBox()
        self.task_group.setObjectName("liveTaskGroup")
        task_layout = QVBoxLayout(self.task_group)
        self.message = QLabel()
        self.message.setTextFormat(Qt.PlainText)
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        task_layout.addWidget(self.message)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        task_layout.addWidget(self.progress)
        root.addWidget(self.task_group)
        self.cards = QGridLayout()
        self._cards = []
        self.labels, self.values = {}, {}
        for index, key in enumerate(("count", "eff", "pr", "excluded")):
            card = QGroupBox()
            card.setObjectName("statCard")
            body = QVBoxLayout(card)
            label, value = QLabel(), QLabel("—")
            label.setObjectName("subtitle")
            value.setObjectName("pageTitle")
            body.addWidget(label)
            body.addWidget(value)
            self.cards.addWidget(card, index // 2, index % 2)
            self._cards.append(card)
            self.labels[key], self.values[key] = label, value
        root.addLayout(self.cards)
        self.issues = QLabel()
        self.issues.setTextFormat(Qt.PlainText)
        self.issues.setWordWrap(True)
        self.issues.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.issues)
        self.charts = QBoxLayout(QBoxLayout.LeftToRight)
        self.scatter, self.hv_chart = ResultsChart(), ResultsChart()
        self.charts.addWidget(self.scatter, 1)
        self.charts.addWidget(self.hv_chart, 1)
        root.addLayout(self.charts)
        self.provenance = QLabel()
        self.provenance.setTextFormat(Qt.PlainText)
        self.provenance.setWordWrap(True)
        self.provenance.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.provenance)
        self.note = QLabel()
        self.note.setObjectName("subtitle")
        self.note.setWordWrap(True)
        root.addWidget(self.note)
        self.table_title = QLabel()
        root.addWidget(self.table_title)
        self.table = QTableWidget(0, len(DISPLAY_COLUMNS))
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(240)
        root.addWidget(self.table)
        self.timer = QTimer(self)
        self.timer.setInterval(3000)
        self.timer.timeout.connect(self.refresh_now)
        self.set_language("zh")

    def tr(self, key):
        return TEXT[self._language][key]

    def start(self):
        self._stopped = False
        if self.auto.isChecked():
            self.timer.start()
        self.refresh_now()

    def stop(self):
        self._stopped = True
        self.timer.stop()
        self._generation += 1

    def _auto_changed(self, enabled):
        if enabled and not self._stopped:
            self.timer.start()
            self.refresh_now()
        else:
            self.timer.stop()

    def set_config(self, config):
        resolved = config.resolved()
        if resolved.to_dict() == self._config.to_dict():
            return
        self._config = resolved
        self._generation += 1
        self._snapshot = None
        self._render()
        self.refresh_now()

    def refresh_now(self):
        if self._stopped or self._job is not None:
            return
        self.refresh.setEnabled(False)
        self.refresh.setText(self.tr("reading"))
        self._job_generation = self._generation
        self._job = SnapshotJob(self._config)
        self._job.finished.connect(self._received)
        self._job.start()

    def _received(self, snapshot):
        generation = self._job_generation
        self._job.deleteLater()
        self._job = None
        self.refresh.setEnabled(True)
        self.refresh.setText(self.tr("refresh"))
        if self._stopped:
            return
        if generation != self._generation:
            self.refresh_now()
            return
        self._snapshot = snapshot
        self._render()

    def set_language(self, language):
        self._language = language
        self.intro.setText(self.tr("intro"))
        self.auto.setText(self.tr("auto"))
        self.refresh.setText(self.tr("refresh") if self._job is None else self.tr("reading"))
        self.source.setItemText(0, self.tr("doe"))
        self.source.setItemText(1, self.tr("pool"))
        self.task_group.setTitle(self.tr("task"))
        for key, label in self.labels.items():
            label.setText(self.tr(key))
        self.note.setText(self.tr("note"))
        self.table_title.setText(self.tr("table"))
        self.table.setHorizontalHeaderLabels([
            self.tr("eff_axis"), self.tr("pr_axis"), self.tr("flow"), self.tr("power"),
            "P_out (Pa)", "nBl", self.tr("boundary"),
        ])
        self._render_event()
        self._render()

    def show_event(self, event, final=False):
        self._event, self._final = event, final
        self._render_event()
        if final:
            self.refresh_now()

    def _render_event(self):
        event = self._event
        if event is None:
            self.message.setText(self.tr("waiting"))
            self.progress.setFormat(self.tr("progress_unknown"))
            return
        message = getattr(event, "message", str(event))
        metrics = getattr(event, "metrics", {})
        scalars = [f"{key}: {value}" for key, value in metrics.items()
                   if isinstance(value, (int, float, bool)) and not isinstance(value, complex)]
        self.message.setText(str(message)[-1200:] + ("\n" + " · ".join(scalars[:8]) if scalars else ""))
        progress = getattr(event, "progress", None)
        if self._final:
            status = getattr(event, "status", "")
            self.progress.setRange(0, 1000)
            self.progress.setValue(1000 if status == "succeeded" else 0)
            self.progress.setFormat(TEXT[self._language].get(status, status))
        elif isinstance(progress, (int, float)) and math.isfinite(progress) and 0 <= progress <= 1:
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(progress * 1000))
            self.progress.setFormat("%p%")
        else:
            self.progress.setRange(0, 0)
            self.progress.setFormat(self.tr("progress_unknown"))

    @staticmethod
    def _time(value):
        return datetime.fromtimestamp(value).astimezone().strftime("%Y-%m-%d %H:%M:%S") if value else "—"

    def _render(self):
        if not hasattr(self, "table"):
            return
        runtime = self._config.runtime
        self.context.setText(f"{self.tr('pressure')}: {runtime.optimization_outlet_static_pressure_pa:g} ± {runtime.operating_point_pressure_tolerance_pa:g} Pa")
        key = "doe" if self.source.currentIndex() == 0 else "pool"
        snapshot = self._snapshot
        samples = snapshot.samples.get(key) if snapshot else None
        for value in self.values.values():
            value.setText("—")
        if samples and samples.count is not None:
            self.values["count"].setText(str(samples.count))
            self.values["excluded"].setText(str(samples.excluded))
            if samples.best_efficiency is not None:
                self.values["eff"].setText(f"{samples.best_efficiency:.5f}")
                self.values["pr"].setText(f"{samples.best_pressure_ratio:.5f}")
        points = samples.points if samples else []
        title = self.tr("scatter") + ("" if points else " · " + self.tr("empty"))
        self.scatter.render_series(title, self.tr("eff_axis"), "总压比" if self._language == "zh" else "Total PR",
                                   [(self.tr("samples"), PALETTE["accent"], points)], scatter=True)
        # Missing HV values split curves instead of bridging unknown rounds.
        series = []
        for column, name, color in ((1, "true", "accent"), (2, "predicted", "blue")):
            segment = []
            previous = None
            for row in (snapshot.hv if snapshot else []):
                if previous is not None and row[0] > previous + 1 and segment:
                    series.append((self.tr(name), PALETTE[color], segment))
                    segment = []
                previous = row[0]
                if row[column] is None:
                    if segment:
                        series.append((self.tr(name), PALETTE[color], segment))
                        segment = []
                else:
                    segment.append((row[0], row[column]))
            if segment:
                series.append((self.tr(name), PALETTE[color], segment))
        self.hv_chart.render_series(self.tr("hv") + ("" if series else " · " + self.tr("empty")), self.tr("iteration"), "HV", series)
        issues = [issue for issue in snapshot.issues if issue.source in (key, "hv", "monitor")] if snapshot else []
        self.issues.setText("\n".join(f"{TEXT[self._language].get(i.source, i.source)}: {self.tr(i.code)}" for i in issues))
        self.issues.setToolTip("\n".join(i.detail for i in issues))
        self.issues.setVisible(bool(issues))
        provenance = []
        if samples:
            provenance.append(f"{self.tr('source')}: {samples.source}\n{self.tr('modified')}: {self._time(samples.modified)} · {self.tr('dedup')}: {samples.duplicates}")
        if snapshot:
            provenance.append(f"HV: {snapshot.hv_source}\n{self.tr('stage')}: {snapshot.stage or '—'} · {self.tr('modified')}: {self._time(snapshot.hv_modified)}\n{self.tr('checked')}: {datetime.fromisoformat(snapshot.checked_at).astimezone().strftime('%Y-%m-%d %H:%M:%S')}")
        self.provenance.setText("\n".join(provenance))
        rows = samples.rows if samples else []
        scroll = self.table.verticalScrollBar().value()
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column, value in enumerate(row):
                item = QTableWidgetItem(f"{value:.6g}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row_index, column, item)
        self.table.verticalScrollBar().setValue(scroll)

    def resizeEvent(self, event):
        wide = self.width() >= 900
        self.charts.setDirection(QBoxLayout.LeftToRight if wide else QBoxLayout.TopToBottom)
        for card in self._cards:
            self.cards.removeWidget(card)
        for index, card in enumerate(self._cards):
            self.cards.addWidget(card, 0 if wide else index // 2, index if wide else index % 2)
        super().resizeEvent(event)
