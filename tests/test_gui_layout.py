"""Small offscreen checks for the desktop workflow shell."""

import os
import unittest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QScrollArea
    from impeller_app.gui.main import MainWindow
except ImportError:
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class GuiLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from impeller_app.config import AppConfig, WorkspacePaths
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config = AppConfig(workspace=WorkspacePaths(project_root=Path(temporary.name)))
        loader = patch("impeller_app.gui.main.AppConfig.load", return_value=config)
        loader.start()
        self.addCleanup(loader.stop)
        # Async integration is tested explicitly below; layout checks do not read files.
        starter = patch("impeller_app.gui.live_results.LiveResultsPage.start")
        starter.start()
        self.addCleanup(starter.stop)

    def test_navigation_and_collapsible_sections(self):
        window = MainWindow()
        try:
            self.assertEqual(window.pages.count(), 9)
            self.assertTrue(all(isinstance(window.pages.widget(i), QScrollArea) for i in range(9)))
            self.assertFalse(window.engineering_defaults_group.isVisible())
            window._toggle_advanced()
            self.assertFalse(window.engineering_defaults_group.isHidden())
            window._toggle_log()
            self.assertTrue(window.log.isHidden())
            window._on_language_changed(1)
            self.assertEqual(window.navigation.item(window._tab_indexes["tab_environment"]).text(), "Environment")
            self.assertEqual(window.advanced_button.text(), "Hide engineering thresholds")
        finally:
            with patch.object(window, "_persist_current_config"):
                window.close()

    def test_live_events_and_source_switch_clear_previous_rows(self):
        from impeller_app.core.live_results import ResultsSnapshot, SampleSnapshot
        from impeller_app.models import TaskUpdate, TaskResult
        window = MainWindow()
        try:
            page = window.live_results
            snapshot = ResultsSnapshot(samples={"doe": SampleSnapshot(
                source="test fixture", count=1, best_efficiency=0.8, best_pressure_ratio=1.4,
                points=[(0.8, 1.4)], rows=[(0.8, 1.4, 3.6, 100, 12, 10, 0)])})
            page._snapshot = snapshot
            page._render()
            self.assertEqual(page.table.rowCount(), 1)
            page.source.setCurrentIndex(1)
            self.assertEqual(page.table.rowCount(), 0)
            self.assertEqual(page.values["count"].text(), "—")
            window._handle_update(TaskUpdate("running", "case completed", progress=0.5, metrics={"completed_runs": 2}))
            self.assertEqual(page.progress.value(), 500)
            self.assertIn("completed_runs: 2", page.message.text())
            with patch.object(page, "refresh_now"):
                window._handle_result(TaskResult("canceled", "cancel confirmed"))
            self.assertEqual(window._status_key, "status_canceled")
            self.assertEqual(page.progress.format(), "已取消")
            window._on_language_changed(1)
            self.assertEqual(page.progress.format(), "Canceled")
        finally:
            with patch.object(window, "_persist_current_config"):
                window.close()

    def test_async_refresh_discards_old_workspace_and_stops_cleanly(self):
        from dataclasses import replace
        from PySide6.QtCore import QObject, Signal
        from impeller_app.core.live_results import ResultsSnapshot
        class ControlledJob(QObject):
            finished = Signal(object)
            def __init__(self, config):
                super().__init__()
            def start(self):
                pass
        window = MainWindow()
        try:
            page = window.live_results
            with patch("impeller_app.gui.live_results.SnapshotJob", ControlledJob):
                page.refresh_now()
                old_job = page._job
                page.refresh_now()
                self.assertIs(page._job, old_job)
                changed = replace(page._config, runtime=replace(page._config.runtime, optimization_outlet_static_pressure_pa=11))
                page.set_config(changed)
                old_job.finished.emit(ResultsSnapshot(stage="old"))
                self.assertIsNone(page._snapshot)
                self.assertIsNot(page._job, old_job)
                current = ResultsSnapshot(stage="new")
                page._job.finished.emit(current)
                self.assertIs(page._snapshot, current)
                page.refresh_now()
                page.stop()
                page._job.finished.emit(ResultsSnapshot(stage="late"))
                self.assertIs(page._snapshot, current)
                self.assertFalse(page.timer.isActive())
        finally:
            with patch.object(window, "_persist_current_config"):
                window.close()

    def test_background_reader_delivers_updates_and_pause_stops_polling(self):
        from impeller_app.core.live_results import ResultsSnapshot
        window = MainWindow()
        page = window.live_results
        def wait_for_reader():
            deadline = time.monotonic() + 3
            while page._job is not None and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.assertIsNone(page._job)
        try:
            first = ResultsSnapshot(stage="initial")
            second = ResultsSnapshot(stage="updated")
            with patch("impeller_app.gui.live_results.LiveResultsReader.read", side_effect=[first, second]):
                page.refresh_now()
                wait_for_reader()
                self.assertIs(page._snapshot, first)
                page.timer.timeout.emit()
                wait_for_reader()
                self.assertIs(page._snapshot, second)
                page.auto.setChecked(False)
                self.assertFalse(page.timer.isActive())
        finally:
            page.stop()
            wait_for_reader()
            with patch.object(window, "_persist_current_config"):
                window.close()

    def test_page_heading_status_and_compact_layout(self):
        window = MainWindow()
        try:
            window.resize(860, 600)
            window.show()
            for language in (0, 1):
                window._on_language_changed(language)
                for key, index in window._tab_indexes.items():
                    window.navigation.setCurrentRow(index)
                    self.app.processEvents()
                    self.assertEqual(window.pages.currentIndex(), index)
                    self.assertEqual(window.title_label.text(), window.tr(key))
                    self.assertIn(f"Ctrl+{index + 1}", window.navigation.item(index).toolTip())
                    self.assertLessEqual(window.minimumSizeHint().width(), 860)
            for status in ("status_running", "status_stopping", "status_done", "status_failed"):
                window._set_status(status)
                self.assertEqual(window.status_label.property("state"), status)
                self.assertEqual(window.status_label.text(), window.tr(status))
            self.assertEqual(window.run_active_learning_button.property("role"), "primary")
            self.assertIsNone(window.resume_button.property("role"))
        finally:
            with patch.object(window, "_persist_current_config"):
                window.close()

    def test_starting_worker_restores_activity_toggle(self):
        window = MainWindow()
        try:
            window._toggle_log()
            with patch("impeller_app.gui.main.Worker"):
                window._run_worker(lambda *_: None)
            self.assertFalse(window.log.isHidden())
            self.assertEqual(window.log_toggle.text(), window.tr("hide_log"))
        finally:
            window._workers = []
            with patch.object(window, "_persist_current_config"):
                window.close()
