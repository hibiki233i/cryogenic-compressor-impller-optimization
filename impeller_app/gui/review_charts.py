"""Interactive QtCharts used by the analytical workbench."""
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QCursor, QPen
from PySide6.QtCharts import QChartView, QLineSeries, QHorizontalBarSeries, QBarSet, QBarCategoryAxis, QValueAxis
from PySide6.QtWidgets import QToolTip

from .live_results import ResultsChart
from .theme import PALETTE


class ReviewChart(ResultsChart):
    point_selected = Signal(str)

    def __init__(self):
        super().__init__()
        self.setRubberBand(QChartView.RectangleRubberBand | QChartView.ClickThroughRubberBand)

    def scatter(self, title, x_label, y_label, points, labels=None, diagonal=False):
        self.render_series(title, x_label, y_label, [("", PALETTE["accent"], points)], scatter=True)
        self.chart().legend().hide()
        if not points:
            return
        series = self.chart().series()[0]
        lookup = {tuple(map(float, point)): str(label) for point, label in zip(points, labels or [])}
        series.hovered.connect(lambda point, state: QToolTip.showText(
            QCursor.pos(), f"{lookup.get((point.x(), point.y()), '')}\n{x_label}: {point.x():.7g}\n{y_label}: {point.y():.7g}"
        ) if state else QToolTip.hideText())
        series.clicked.connect(lambda point: self.point_selected.emit(lookup.get((point.x(), point.y()), "")))
        if diagonal:
            low = min(min(x, y) for x, y in points)
            high = max(max(x, y) for x, y in points)
            if low == high:
                low, high = low - 0.01, high + 0.01
            reference = QLineSeries()
            reference.append(low, low)
            reference.append(high, high)
            reference.setPen(QPen(QColor(PALETTE["muted"]), 1, Qt.DashLine))
            self.chart().addSeries(reference)
            for axis in self.chart().axes():
                reference.attachAxis(axis)
                axis.setRange(low, high)

    def wheelEvent(self, event):
        self.chart().zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.chart().zoomReset()
        event.accept()


class BarChart(ResultsChart):
    def render_bars(self, title, labels, values, axis_label="", symmetric=False):
        self.setMinimumHeight(max(290, 120 + len(labels) * 25))
        self.render_series(title, "", "", [])
        if not values:
            return
        series = QHorizontalBarSeries()
        bar = QBarSet(axis_label)
        bar.append([float(value) for value in values])
        bar.setColor(QColor(PALETTE["accent"]))
        bar.setBorderColor(QColor(PALETTE["accent"]))
        series.append(bar)
        self.chart().addSeries(series)
        categories, axis = QBarCategoryAxis(), QValueAxis()
        categories.append([str(label) for label in labels])
        categories.setLabelsColor(QColor(PALETTE["muted"]))
        categories.setGridLineVisible(False)
        axis.setLabelsColor(QColor(PALETTE["muted"]))
        axis.setGridLinePen(QPen(QColor(PALETTE["border"])))
        axis.setTitleBrush(QColor(PALETTE["muted"]))
        axis.setTitleText(axis_label)
        axis.setRange(-1 if symmetric else min(0, min(values)), 1 if symmetric else max(1e-9, max(values) * 1.1))
        self.chart().addAxis(categories, Qt.AlignLeft)
        self.chart().addAxis(axis, Qt.AlignBottom)
        series.attachAxis(categories)
        series.attachAxis(axis)
        self.chart().legend().hide()
        bar.hovered.connect(lambda state, index: QToolTip.showText(QCursor.pos(), f"{labels[index]}: {values[index]:.7g}") if state else QToolTip.hideText())
