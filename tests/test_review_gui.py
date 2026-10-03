"""Native Qt integration checks for analytical and case-browser interactions."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import pandas as pd
from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtWidgets import QApplication

from design_variables import geometry_variable_names, DEFAULT_VARIABLE_SPECS
from impeller_app.config import AppConfig, WorkspacePaths
from impeller_app.core.analysis import AnalysisSnapshot
from impeller_app.core.cases import CaseRecord, CaseInventory, CaseDetail, FilePreview
from impeller_app.gui.analytics import AnalyticsPage
from impeller_app.gui.cases import CasesPage
from impeller_app.gui.review_widgets import DataTable, AsyncReader


class ReviewGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = AppConfig(workspace=WorkspacePaths(project_root=self.root)).resolved()

    def test_table_sorts_numerically_and_selection_tracks_proxy_row(self):
        table = DataTable()
        table.set_frame(pd.DataFrame({"metric": [2.0, 10.0, 1.0], "run_id": ["two", "ten", "one"]}))
        selected = []
        table.selected.connect(selected.append)
        table.proxy.sort(0, Qt.DescendingOrder)
        table._select(table.proxy.index(0, 0))
        self.assertEqual(selected[-1]["run_id"], "ten")
        table.search.setText("two")
        self.assertEqual(table.filtered_frame().run_id.tolist(), ["two"])
        table.close()

    def test_export_uses_filtered_order_and_refuses_source_overwrite(self):
        table = DataTable()
        table.set_frame(pd.DataFrame({"value": [1, 2], "kind": ["keep", "omit"]}))
        table.search.setText("keep")
        output = self.root / "export.csv"
        with patch("impeller_app.gui.review_widgets.QFileDialog.getSaveFileName", return_value=(str(output), "")):
            table._export()
            self.assertEqual(pd.read_csv(output).value.tolist(), [1])
            original = output.read_bytes()
            table.protected_paths = {str(output)}
            with patch("impeller_app.gui.review_widgets.QMessageBox.warning") as warning:
                table._export()
                warning.assert_called_once()
            self.assertEqual(output.read_bytes(), original)
        table.close()

    def snapshot(self):
        rows = []
        for i in range(4):
            row = {spec["name"]: (spec["lower"] + spec["upper"]) / 2 for spec in DEFAULT_VARIABLE_SPECS}
            row.update(nBl=9+i, P_out=12, d1s=.34+i*.01, Efficiency=.7+i*.01, totalpressureratio=1.4+i*.02, MassFlow=3.6, Power=100, is_boundary=i%2)
            rows.append(row)
        pairs = pd.DataFrame([{"run_id": "AL_Iter01_P1_A00001", "iter": 1,
                               "true_eff": .7, "pred_eff": .72, "true_pr": 1.4, "pred_pr": 1.5,
                               "true_mf": 3.6, "pred_mf": 3.8, "uncertainty_real_eff": .02}])
        cv = pd.DataFrame([{"iter": 1, "fold": 1, "rmse_eff": .1, "cv_protocol": "old"},
                           {"iter": 2, "fold": 1, "rmse_eff": .05, "cv_protocol": "new"}])
        return AnalysisSnapshot(frames={"doe": pd.DataFrame(rows), "pool": pd.DataFrame(rows)},
                                audits={"query": pairs, "fixed": pairs.iloc[:0], "cv": cv, "validation": pd.DataFrame()},
                                pairs={"query": pairs, "fixed": pairs.iloc[:0]},
                                geometry_names=geometry_variable_names(DEFAULT_VARIABLE_SPECS))

    def test_analysis_filters_categories_and_links_online_case(self):
        page = AnalyticsPage(self.config)
        try:
            page._received(self.snapshot())
            self.assertNotIn("nBl", page.correlation_table.model.frame.variable.tolist())
            self.assertEqual(len(page.group_table.model.frame), 4)
            page.blades.setCurrentIndex(page.blades.findData(10))
            self.assertEqual(len(page.observations.model.frame), 1)
            page.boundary.setCurrentIndex(page.boundary.findData("0"))
            self.assertEqual(len(page.observations.model.frame), 0)
            requests = []
            page.case_requested.connect(lambda source, run: requests.append((source, run)))
            page.parity.point_selected.emit("AL_Iter01_P1_A00001")
            self.assertEqual(requests, [("active_learning", "AL_Iter01_P1_A00001")])
            page.prediction_source.setCurrentIndex(1)
            page.parity.point_selected.emit("AL_Iter01_P1_A00001")
            self.assertEqual(len(requests), 1)
            page.objective.setCurrentText("Power")
            self.assertTrue(page.pair_table.model.frame.empty)
        finally:
            page.stop()
            page.close()

    def test_cv_protocols_not_merged_and_language_switch_keeps_selection(self):
        page = AnalyticsPage(self.config)
        try:
            page._received(self.snapshot())
            page.protocol.setCurrentIndex(page.protocol.findData("new"))
            self.assertEqual(page.cv_table.model.frame.cv_protocol.tolist(), ["new"])
            page.set_language("en")
            self.assertEqual(page.protocol.currentData(), "new")
            self.assertEqual(page.tabs.tabText(0), "Variables & performance")
            self.assertEqual(page.cv_table.model.frame.rmse_eff.tolist(), [.05])
        finally:
            page.stop()
            page.close()

    def test_case_filter_keeps_unknown_separate_and_uses_source_identity(self):
        page = CasesPage(self.config)
        try:
            records = [CaseRecord("doe", "same", self.root/"doe"/"same", "succeeded"),
                       CaseRecord("active_learning", "same", self.root/"al"/"same", "failed", "infrastructure"),
                       CaseRecord("doe", "Run_003", self.root/"Run_003")]
            page._received_inventory(CaseInventory(records=records))
            page.status.setCurrentIndex(page.status.findData("failed"))
            self.assertEqual(page.table.proxy.rowCount(), 1)
            with patch.object(page, "_load_detail") as load:
                page.table._select(page.table.proxy.index(0, 0))
                self.assertEqual(load.call_args.args[0], str(records[1].key))
            page.status.setCurrentIndex(page.status.findData("unknown"))
            self.assertEqual(page.table.proxy.rowCount(), 1)
            page.set_language("en")
            self.assertEqual(page.status.currentData(), "unknown")
            self.assertEqual(page.tabs.tabText(0), "Parameters & results")
        finally:
            page.stop()
            page.close()

    def test_changing_case_clears_old_file_and_obsolete_detail_is_ignored(self):
        page = CasesPage(self.config)
        try:
            first, second = CaseRecord("doe", "first", self.root/"first"), CaseRecord("doe", "second", self.root/"second")
            page._received_inventory(CaseInventory(records=[first, second]))
            with patch.object(page.detail_loader, "submit"):
                page._load_detail(str(first.key))
                page._received_detail(CaseDetail(first))
                page._received_file(FilePreview("old", "text", "old contents"))
                page._load_detail(str(second.key))
                self.assertEqual(page.text_preview.toPlainText(), "")
                page._received_detail(CaseDetail(first))
                self.assertIsNone(page.detail)
        finally:
            page.stop()
            page.close()

    def test_async_reader_discards_superseded_and_closed_results(self):
        class Job(QObject):
            completed = Signal(object, object)
            def __init__(self, operation):
                super().__init__()
                self.operation = operation
            def start(self):
                pass
        reader = AsyncReader()
        values = []
        reader.ready.connect(values.append)
        with patch("impeller_app.gui.review_widgets.ReadJob", Job):
            reader.submit(lambda: "old")
            old = reader._job
            reader.submit(lambda: "new")
            old.completed.emit("old", None)
            self.assertEqual(values, [])
            reader._job.completed.emit("new", None)
            self.assertEqual(values, ["new"])
            reader.submit(lambda: "late")
            reader.close()
            reader._job.completed.emit("late", None)
            self.assertEqual(values, ["new"])

    def test_case_link_refuses_same_id_from_another_stage(self):
        page = CasesPage(self.config)
        try:
            record = CaseRecord("active_learning", "same", self.root/"different"/"same", workspace_match=False)
            page._received_inventory(CaseInventory(records=[record]))
            with patch.object(page, "_load_detail") as load:
                page.select_case("active_learning", "same")
                load.assert_not_called()
            self.assertIn("其他阶段", page.notice.text())
        finally:
            page.stop()
            page.close()

    def test_native_chart_click_and_zoom_reset(self):
        from PySide6.QtCore import QPointF
        from PySide6.QtTest import QTest
        from impeller_app.gui.review_charts import ReviewChart
        chart = ReviewChart()
        chart.resize(640, 400)
        chart.scatter("test", "x", "y", [(1, 1), (2, 2), (3, 1)], ["one", "two", "three"])
        chart.show()
        self.app.processEvents()
        selected = []
        chart.point_selected.connect(selected.append)
        series = chart.chart().series()[0]
        position = chart.chart().mapToPosition(QPointF(2, 2), series)
        QTest.mouseClick(chart.viewport(), Qt.LeftButton, pos=position.toPoint())
        self.app.processEvents()
        self.assertEqual(selected, ["two"])
        chart.chart().zoom(1.5)
        self.assertTrue(chart.chart().isZoomed())
        QTest.mouseDClick(chart.viewport(), Qt.LeftButton, pos=position.toPoint())
        self.assertFalse(chart.chart().isZoomed())
        chart.close()

    def test_review_config_does_not_change_runner_config_or_live_monitor(self):
        from impeller_app.gui.main import MainWindow
        with patch("impeller_app.gui.main.AppConfig.load", return_value=self.config):
            window = MainWindow()
        try:
            other = replace(self.config, runtime=replace(self.config.runtime, optimization_outlet_static_pressure_pa=11))
            window._set_review_config(other, self.root/"stage_config.json")
            self.assertEqual(window.analytics_page.config.runtime.optimization_outlet_static_pressure_pa, 11)
            self.assertEqual(window.cases_page.config.runtime.optimization_outlet_static_pressure_pa, 11)
            self.assertEqual(window.config.runtime.optimization_outlet_static_pressure_pa, 12)
            self.assertEqual(window.live_results._config.runtime.optimization_outlet_static_pressure_pa, 12)
        finally:
            with patch.object(window, "_persist_current_config"):
                window.close()


if __name__ == "__main__":
    unittest.main()
