"""Native PySide6 case browser: searchable inventory and read-only evidence."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import pandas as pd
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap, QImageReader
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QSplitter, QTabWidget, QPlainTextEdit, QStackedWidget, QMessageBox,
)

from ..core.cases import CaseReader, TEXT_SUFFIXES, preview_file, safe_child
from .review_widgets import AsyncReader, DataTable


TEXT = {
    "zh": {
        "intro": "浏览 DOE 与主动学习算例，核对参数、状态、收敛证据和日志。登记成功与当前结果通过验证分别显示。",
        "refresh": "刷新算例", "reading": "正在读取…", "all_sources": "全部来源", "doe": "DOE", "active_learning": "主动学习",
        "all_status": "全部状态", "succeeded": "登记成功", "failed": "登记失败", "submitted": "已提交", "canceled": "已取消", "unknown": "未登记",
        "overview": "参数与结果", "files": "文件与日志", "documents": "审计与元数据", "open": "打开所选算例目录", "choose": "选择一个算例查看详情",
        "run_id": "算例 ID", "source": "来源", "status": "登记状态", "failure_stage": "失败阶段", "attempts": "尝试次数", "updated": "记录更新时间", "exists": "目录存在", "yes": "是", "no": "否",
        "empty": "暂无匹配算例", "total": "算例", "errors": "读取诊断", "log": "仅文本日志", "reload": "重读所选文件", "binary": "二进制工程文件不在此解析，可打开算例目录查看。",
        "tail": "预览最多读取末尾 256 KiB / 800 行", "truncated": "内容已截取", "verified": "当前 CFX 结果通过定义与收敛证据校验", "unverified": "结果存在，但未通过当前定义/收敛/时效校验", "missing": "尚无可验证的 CFX 结果",
        "convergence": "末次 CFX 收敛证据", "parameters": "输入参数", "metrics": "已验证性能（整机流量/功率）", "audit": "登记状态仅描述任务记录；软件/许可证/数值失败不自动等于物理不可行。",
        "not_found": "该配置中没有找到对应算例", "file": "文件", "bytes": "字节", "modified": "文件更新时间", "image_error": "图片无法解码或尺寸超过预览限制",
        "stage_mismatch": "同名目录属于其他阶段，不能与当前数据记录关联；请打开对应阶段配置。",
        "executed_blades": "执行叶片数（按 runner 取整；上方保留原始输入）",
    },
    "en": {
        "intro": "Inspect DOE and AL cases, parameters, recorded status, convergence evidence and logs. Recorded success and currently verified results are shown separately.",
        "refresh": "Refresh cases", "reading": "Reading…", "all_sources": "All sources", "doe": "DOE", "active_learning": "Active learning",
        "all_status": "All statuses", "succeeded": "Recorded success", "failed": "Recorded failure", "submitted": "Submitted", "canceled": "Canceled", "unknown": "Unregistered",
        "overview": "Parameters & results", "files": "Files & logs", "documents": "Audit & metadata", "open": "Open selected case folder", "choose": "Select a case to inspect",
        "run_id": "Run ID", "source": "Source", "status": "Recorded status", "failure_stage": "Failure stage", "attempts": "Attempts", "updated": "Record updated", "exists": "Folder exists", "yes": "Yes", "no": "No",
        "empty": "No matching cases", "total": "Cases", "errors": "Read diagnostics", "log": "Text logs only", "reload": "Reload selected file", "binary": "Binary engineering files are not parsed here. Open the case folder to inspect them.",
        "tail": "Preview limited to the last 256 KiB / 800 lines", "truncated": "Content truncated", "verified": "Current CFX result passes definition and convergence checks", "unverified": "Result exists but fails current definition, convergence or freshness checks", "missing": "No verifiable CFX result yet",
        "convergence": "Latest CFX convergence evidence", "parameters": "Input parameters", "metrics": "Verified performance (full-impeller flow/power)", "audit": "Recorded status describes the task outcome. Software, license or numerical failures do not automatically imply physical infeasibility.",
        "not_found": "No matching case in this configuration", "file": "File", "bytes": "Bytes", "modified": "File modified", "image_error": "Image cannot be decoded or exceeds preview dimension limits",
        "stage_mismatch": "The same-named folder belongs to another stage; open that stage's configuration before linking its data.",
        "executed_blades": "Executed blade count (runner rounding; raw input retained above)",
    },
}


class CasesPage(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config.resolved()
        self.language = "zh"
        self.inventory = None
        self.detail = None
        self._selected_key = None
        self._selected_file = None
        self._pending_case = None
        self._records = {}
        self._preview_value = None
        self.auto_refresh = True
        self._file_to_restore = None
        self.inventory_loader, self.detail_loader, self.file_loader = (AsyncReader(self) for _ in range(3))
        self.inventory_loader.ready.connect(self._received_inventory)
        self.inventory_loader.error.connect(self._inventory_error)
        self.inventory_loader.busy.connect(lambda busy: self.refresh_button.setEnabled(not busy))
        self.detail_loader.ready.connect(self._received_detail)
        self.detail_loader.error.connect(self._detail_error)
        self.file_loader.ready.connect(self._received_file)
        self.file_loader.error.connect(self._file_error)
        self.timer = QTimer(self)
        self.timer.setInterval(15000)
        self.timer.timeout.connect(lambda: self.refresh() if self.isVisible() and self.inventory_loader._job is None and self.detail_loader._job is None and self.file_loader._job is None else None)
        root = QVBoxLayout(self)
        self.intro = self._label(root)
        controls = QHBoxLayout()
        self.source, self.status = QComboBox(), QComboBox()
        for key in ("all_sources", "doe", "active_learning"):
            self.source.addItem("", key)
        for key in ("all_status", "succeeded", "failed", "submitted", "canceled", "unknown"):
            self.status.addItem("", key)
        self.refresh_button = QPushButton()
        self.refresh_button.clicked.connect(self.refresh)
        for widget in (self.source, self.status):
            controls.addWidget(widget)
        controls.addStretch()
        controls.addWidget(self.refresh_button)
        root.addLayout(controls)
        self.summary = self._label(root)
        self.table = DataTable()
        self.table.selected.connect(self._select_row)
        self.table.search.textChanged.connect(self._check_filtered_selection)
        self.browser_splitter = QSplitter(Qt.Horizontal)
        self.browser_splitter.setChildrenCollapsible(False)
        self.browser_splitter.addWidget(self.table)
        detail_panel = QWidget()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(6, 0, 0, 0)
        self.browser_splitter.addWidget(detail_panel)
        self.browser_splitter.setSizes([430, 680])
        root.addWidget(self.browser_splitter)
        header = QHBoxLayout()
        self.selection = QLabel()
        self.selection.setTextFormat(Qt.PlainText)
        self.selection.setWordWrap(True)
        self.open_button = QPushButton()
        self.open_button.clicked.connect(self._open_folder)
        self.open_button.setEnabled(False)
        header.addWidget(self.selection, 1)
        header.addWidget(self.open_button)
        detail_layout.addLayout(header)
        self.notice = self._label(detail_layout)
        self.tabs = QTabWidget()
        detail_layout.addWidget(self.tabs)
        self.overview = QPlainTextEdit()
        self.overview.setReadOnly(True)
        self.overview.setMinimumHeight(320)
        self.tabs.addTab(self.overview, "")
        files = QWidget()
        f = QVBoxLayout(files)
        toolbar = QHBoxLayout()
        self.file_filter = QComboBox()
        self.file_filter.addItems(["", ""])
        self.reload_button = QPushButton()
        self.reload_button.clicked.connect(self._reload_file)
        toolbar.addWidget(self.file_filter)
        toolbar.addStretch()
        toolbar.addWidget(self.reload_button)
        f.addLayout(toolbar)
        self.file_table = DataTable()
        self.file_table.export.hide()
        self.file_table.selected.connect(self._select_file)
        f.addWidget(self.file_table)
        self.preview_info = self._label(f)
        self.preview = QStackedWidget()
        self.text_preview = QPlainTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setMaximumBlockCount(800)
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumHeight(280)
        self.image_preview.setMaximumHeight(600)
        self.preview.addWidget(self.text_preview)
        self.preview.addWidget(self.image_preview)
        self.preview.setMinimumHeight(300)
        f.addWidget(self.preview)
        self.tabs.addTab(files, "")
        self.documents = QPlainTextEdit()
        self.documents.setReadOnly(True)
        self.documents.setMaximumBlockCount(5000)
        self.documents.setMinimumHeight(320)
        self.tabs.addTab(self.documents, "")
        self.issues = self._label(root)
        self.source.currentIndexChanged.connect(self._filter)
        self.status.currentIndexChanged.connect(self._filter)
        self.file_filter.currentIndexChanged.connect(self._populate_files)
        self.set_language("zh")

    def tr(self, key):
        return TEXT[self.language][key]

    def _label(self, layout):
        label = QLabel()
        label.setWordWrap(True)
        label.setTextFormat(Qt.PlainText)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(label)
        return label

    def set_config(self, config):
        self.config = config.resolved()
        for loader in (self.inventory_loader, self.detail_loader, self.file_loader):
            loader.invalidate()
        self.inventory = None
        self._records = {}
        self._clear_detail()
        self._filter()
        if self.isVisible():
            self.refresh()

    def refresh(self):
        config = self.config
        self.inventory_loader.submit(lambda: CaseReader(config).inventory())

    def _received_inventory(self, inventory):
        self.inventory = inventory
        self._records = {str(record.key): record for record in inventory.records}
        selected = self._selected_key
        self._filter()
        self.issues.setText("\n".join(inventory.issues))
        self.table.protected_paths = {str(self.config.workspace.failure_records_csv), str(self.config.workspace.al_query_validation_csv)}
        if self._pending_case:
            source, run = self._pending_case
            self._pending_case = None
            self.select_case(source, run)
        elif self._selected_key and selected in self._records:
            self._load_detail(selected)
        else:
            self._clear_detail()

    def _filter(self):
        if not hasattr(self, "issues"):
            return
        records = list(self._records.values())
        source, status = self.source.currentData(), self.status.currentData()
        rows = []
        for record in records:
            if source != "all_sources" and record.source != source:
                continue
            if status != "all_status" and record.status != status:
                continue
            rows.append({self.tr("run_id"): record.run_id, self.tr("source"): self.tr(record.source),
                         self.tr("status"): TEXT[self.language].get(record.status, record.status),
                         self.tr("failure_stage"): record.failure_stage, self.tr("attempts"): record.attempts,
                         self.tr("updated"): record.updated, self.tr("exists"): self.tr("yes" if record.exists else "no"),
                         "reason": record.reason, "_key": str(record.key)})
        self.table.set_frame(pd.DataFrame(rows))
        if rows:
            self.table.view.setColumnHidden(len(rows[0]) - 1, True)
        self._update_summary()
        # Detail panels never remain attached to a case excluded by the controls.
        if self._selected_key and self._selected_key not in {r["_key"] for r in rows}:
            self._clear_detail()
        self._restore_table_selection()

    def _select_row(self, row):
        key = row.get("_key")
        if key in self._records:
            self._load_detail(key)

    def _check_filtered_selection(self):
        self._update_summary()
        if self._selected_key:
            frame = self.table.filtered_frame()
            if "_key" not in frame or self._selected_key not in set(frame["_key"]):
                self._clear_detail()

    def _update_summary(self):
        records = list(self._records.values())
        self.summary.setText(f"{self.tr('total')}: {self.table.proxy.rowCount()} / {len(records)} · " + " · ".join(f"{self.tr(key)}: {sum(r.status == key for r in records)}" for key in ("succeeded", "failed", "unknown")))

    def _restore_table_selection(self):
        if not self._selected_key:
            return
        for row in range(self.table.proxy.rowCount()):
            index = self.table.proxy.mapToSource(self.table.proxy.index(row, 0))
            if self.table.model.frame.iloc[index.row()]["_key"] == self._selected_key:
                selection = self.table.view.selectionModel()
                selection.blockSignals(True)
                self.table.view.selectRow(row)
                selection.blockSignals(False)
                break

    def _clear_detail(self):
        self.detail_loader.invalidate()
        self.file_loader.invalidate()
        self.detail = None
        self._selected_key = None
        self._selected_file = None
        self._preview_value = None
        self.selection.setText(self.tr("choose"))
        self.notice.clear()
        self.overview.clear()
        self.documents.clear()
        self.file_table.set_frame(pd.DataFrame())
        self.text_preview.clear()
        self.image_preview.clear()
        self.preview_info.clear()
        self.open_button.setEnabled(False)

    def _load_detail(self, key):
        old_file = self._selected_file if key == self._selected_key else None
        self._clear_detail()
        self._file_to_restore = old_file
        record = self._records[key]
        self._selected_key = key
        self.selection.setText(f"{self.tr('reading')} · {record.run_id}\n{record.path}")
        config = self.config
        self.detail_loader.submit(lambda: CaseReader(config).detail(record))

    def _received_detail(self, detail):
        if str(detail.record.key) != self._selected_key:
            return
        self.detail = detail
        self._render_detail()
        if self._file_to_restore in {item["file"] for item in detail.files}:
            self._selected_file = self._file_to_restore
            self._reload_file()
        self._file_to_restore = None
        self._restore_table_selection()

    def _render_detail(self):
        if self.detail is None:
            return
        detail = self.detail
        record = detail.record
        self.selection.setText(f"{self.tr(record.source)} / {record.run_id}\n{record.path}")
        self.open_button.setEnabled(record.path.is_dir())
        self.notice.setText(self.tr(detail.result_state) + "\n" + self.tr("audit") + ("\n" + "\n".join(detail.issues) if detail.issues else ""))
        blocks = [f"{self.tr('status')}: {TEXT[self.language].get(record.status, record.status)}",
                  f"{self.tr('failure_stage')}: {record.failure_stage or '—'}\n{record.reason}",
                  f"{self.tr('parameters')} ({detail.parameter_source})\n" + json.dumps(detail.parameters, ensure_ascii=False, indent=2),
                  f"{self.tr('executed_blades')}: {detail.effective_blades if detail.effective_blades is not None else '—'}",
                  self.tr("metrics") + "\n" + (json.dumps(detail.metrics, ensure_ascii=False, indent=2) if detail.metrics else "—"),
                  self.tr("convergence") + "\n" + (json.dumps(detail.convergence, ensure_ascii=False, indent=2) if detail.convergence else "—")]
        self.overview.setPlainText("\n\n".join(blocks))
        self.documents.setPlainText(json.dumps({"audit": record.audit, **detail.documents}, ensure_ascii=False, indent=2, default=str)[:256 * 1024])
        self._populate_files()

    def _populate_files(self):
        if not hasattr(self, "file_table"):
            return
        rows = []
        for item in self.detail.files if self.detail else []:
            if self.file_filter.currentIndex() == 1 and Path(item["file"]).suffix.lower() not in TEXT_SUFFIXES:
                continue
            rows.append({self.tr("file"): item["file"], self.tr("bytes"): item["bytes"],
                         self.tr("modified"): datetime.fromtimestamp(item["modified"]).strftime("%Y-%m-%d %H:%M:%S"), "_relative": item["file"]})
        self.file_table.set_frame(pd.DataFrame(rows))
        if rows:
            self.file_table.view.setColumnHidden(3, True)

    def _select_file(self, row):
        relative = row.get("_relative")
        if relative and self.detail:
            self._selected_file = relative
            self._reload_file()

    def _reload_file(self):
        if self.detail and self._selected_file:
            root, relative = self.detail.record.path, self._selected_file
            self._preview_value = None
            self.preview_info.setText(str(root / relative))
            self.image_preview.clear()
            self.text_preview.setPlainText(self.tr("reading"))
            self.preview.setCurrentIndex(0)
            self.file_loader.submit(lambda: preview_file(root, relative))

    def _received_file(self, value):
        self._preview_value = value
        self._render_preview()

    def _render_preview(self):
        value = self._preview_value
        if value is None:
            return
        self.preview_info.setText(value.path + "\n" + self.tr("tail") + (" · " + self.tr("truncated") if value.truncated else ""))
        if value.kind == "image":
            from PySide6.QtCore import QBuffer, QByteArray, QIODevice
            buffer = QBuffer()
            buffer.setData(QByteArray(value.data))
            buffer.open(QIODevice.ReadOnly)
            reader = QImageReader(buffer)
            size = reader.size()
            if size.width() <= 0 or size.height() <= 0 or size.width() * size.height() > 32_000_000:
                self._file_error(self.tr("image_error"))
                return
            reader.setScaledSize(size.scaled(1000, 560, Qt.KeepAspectRatio))
            image = reader.read()
            if image.isNull():
                self._file_error(self.tr("image_error"))
                return
            self.image_preview.setPixmap(QPixmap.fromImage(image))
            self.preview.setCurrentIndex(1)
        else:
            self.text_preview.setPlainText(value.text if value.kind == "text" else self.tr("binary"))
            self.preview.setCurrentIndex(0)

    def _open_folder(self):
        if self.detail and self.detail.record.path.is_dir():
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.detail.record.path))):
                QMessageBox.warning(self, self.tr("open"), str(self.detail.record.path))

    def _inventory_error(self, message):
        self.inventory = None
        self._records = {}
        self._filter()
        self._clear_detail()
        self.issues.setText(message)

    def _detail_error(self, message):
        self._clear_detail()
        self.notice.setText(message)

    def _file_error(self, message):
        self._preview_value = None
        self.image_preview.clear()
        self.text_preview.setPlainText(message)
        self.preview.setCurrentIndex(0)

    def select_case(self, source, run_id):
        if self.inventory is None:
            self._pending_case = (source, run_id)
            self.refresh()
            return
        self.source.setCurrentIndex(0)
        self.status.setCurrentIndex(0)
        self.table.search.clear()
        for row in range(self.table.proxy.rowCount()):
            index = self.table.proxy.mapToSource(self.table.proxy.index(row, 0))
            key = self.table.model.frame.iloc[index.row()]["_key"]
            record = self._records[key]
            if record.source == source and record.run_id == run_id:
                if not record.workspace_match:
                    self._clear_detail()
                    self.notice.setText(self.tr("stage_mismatch"))
                    return
                self.table.view.selectRow(row)
                self.table.view.scrollTo(self.table.proxy.index(row, 0))
                if self._selected_key != key:
                    self._load_detail(key)
                return
        self._clear_detail()
        self.notice.setText(f"{self.tr('not_found')}: {source}/{run_id}")

    def set_language(self, language):
        self.language = language
        self.intro.setText(self.tr("intro"))
        for combo in (self.source, self.status):
            for index in range(combo.count()):
                combo.setItemText(index, self.tr(combo.itemData(index)))
        self.refresh_button.setText(self.tr("refresh"))
        self.open_button.setText(self.tr("open"))
        self.reload_button.setText(self.tr("reload"))
        self.file_filter.setItemText(0, self.tr("files"))
        self.file_filter.setItemText(1, self.tr("log"))
        for index, key in enumerate(("overview", "files", "documents")):
            self.tabs.setTabText(index, self.tr(key))
        self.table.set_language(language)
        self.file_table.set_language(language)
        self._filter()
        if self.detail:
            self._render_detail()
            self._render_preview()
        else:
            self.selection.setText(self.tr("choose"))

    def showEvent(self, event):
        super().showEvent(event)
        if self.auto_refresh:
            self.timer.start()
        self.refresh()

    def resizeEvent(self, event):
        self.browser_splitter.setOrientation(Qt.Horizontal if self.width() >= 1000 else Qt.Vertical)
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
        for loader in (self.inventory_loader, self.detail_loader, self.file_loader):
            loader.close()
