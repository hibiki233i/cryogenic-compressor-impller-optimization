"""Reusable native Qt controls for engineering inspection pages."""
from __future__ import annotations

import threading
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal, QAbstractTableModel, Qt, QSortFilterProxyModel
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QTableView,
    QAbstractItemView, QHeaderView, QFileDialog, QMessageBox,
)


class ReadJob(QObject):
    completed = Signal(object, object)

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            value = self.operation()
        except Exception as exc:
            self.completed.emit(None, str(exc))
        else:
            self.completed.emit(value, None)


class AsyncReader(QObject):
    """One in-flight read, one latest pending request; discard obsolete results."""
    ready = Signal(object)
    error = Signal(str)
    busy = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job = None
        self._pending = None
        self._generation = 0
        self._job_generation = 0
        self._closed = False

    def submit(self, operation):
        if self._closed:
            return
        self._generation += 1
        self._pending = operation
        if self._job is None:
            self._start()

    def _start(self):
        operation, self._pending = self._pending, None
        self._job_generation = self._generation
        self._job = ReadJob(operation)
        self._job.completed.connect(self._received)
        self.busy.emit(True)
        self._job.start()

    def _received(self, value, error):
        generation = self._job_generation
        self._job.deleteLater()
        self._job = None
        if self._closed:
            return
        if generation == self._generation:
            if error is None:
                self.ready.emit(value)
            else:
                self.error.emit(error)
        if self._pending is not None:
            self._start()
        else:
            self.busy.emit(False)

    def invalidate(self):
        self._generation += 1
        self._pending = None

    def close(self):
        self._closed = True
        self.invalidate()


class FrameModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = pd.DataFrame()

    def set_frame(self, frame):
        self.beginResetModel()
        self.frame = frame.reset_index(drop=True).copy()
        self.endResetModel()

    def rowCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.frame)

    def columnCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.frame.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self.frame.iat[index.row(), index.column()]
        if value is None or (np.isscalar(value) and pd.isna(value)):
            return "—" if role == Qt.DisplayRole else None
        if role == Qt.UserRole:
            return value.item() if isinstance(value, np.generic) else value
        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            if isinstance(value, (float, np.floating)):
                return f"{value:.7g}"
            return str(value)
        if role == Qt.TextAlignmentRole and isinstance(value, (int, float, np.number)):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(self.frame.columns[section]) if orientation == Qt.Horizontal else str(section + 1)
        return None


class DataTable(QWidget):
    selected = Signal(object)
    activated = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._language = "zh"
        self.protected_paths = set()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.export = QPushButton()
        row.addWidget(self.search, 1)
        row.addWidget(self.export)
        layout.addLayout(row)
        self.model = FrameModel(self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.UserRole)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.search.textChanged.connect(self.proxy.setFilterFixedString)
        self.view = QTableView()
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.setAlternatingRowColors(True)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.horizontalHeader().setDefaultSectionSize(130)
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.verticalHeader().setVisible(False)
        self.view.selectionModel().currentRowChanged.connect(lambda index, _: self._select(index))
        self.view.doubleClicked.connect(lambda index: self._select(index, True))
        layout.addWidget(self.view)
        self.export.clicked.connect(self._export)
        self.setMinimumHeight(220)
        self.set_language("zh")

    def _select(self, proxy_index, activate=False):
        index = self.proxy.mapToSource(proxy_index)
        if index.isValid():
            row = self.model.frame.iloc[index.row()].to_dict()
            (self.activated if activate else self.selected).emit(row)

    def set_frame(self, frame):
        self.model.set_frame(frame)

    def filtered_frame(self):
        rows = [self.proxy.mapToSource(self.proxy.index(i, 0)).row() for i in range(self.proxy.rowCount())]
        return self.model.frame.iloc[rows].copy()

    def set_language(self, language):
        self._language = language
        self.search.setPlaceholderText("搜索表格中的任意字段…" if language == "zh" else "Search any table field…")
        self.export.setText("导出筛选表格" if language == "zh" else "Export filtered table")

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, self.export.text(), "review.csv", "CSV (*.csv)")
        if not path:
            return
        target = Path(path).with_suffix(".csv")
        temporary = target.with_name(target.name + "." + uuid4().hex + ".tmp")
        try:
            if target.resolve() in {Path(p).resolve() for p in self.protected_paths}:
                raise ValueError("Cannot overwrite a monitored source file / 不能覆盖正在查看的源文件")
            self.filtered_frame().to_csv(temporary, index=False, encoding="utf-8-sig")
            temporary.replace(target)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, self.export.text(), str(exc))
        finally:
            temporary.unlink(missing_ok=True)
