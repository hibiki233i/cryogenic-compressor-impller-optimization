"""The live monitor must never fabricate, mix, or modify engineering results."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from design_variables import (
    DEFAULT_VARIABLE_SPECS, PERFORMANCE_DATA_SCHEMA_VERSION,
    TOTAL_PRESSURE_DEFINITION, write_performance_data_metadata,
)
from impeller_app.config import AppConfig, WorkspacePaths
from impeller_app.core.live_results import LiveResultsReader


class LiveResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = AppConfig(workspace=WorkspacePaths(project_root=Path(self.temp.name))).resolved()
        self.reader = LiveResultsReader(self.config)

    def row(self, **changes):
        row = {spec["name"]: (spec["lower"] + spec["upper"]) / 2 for spec in DEFAULT_VARIABLE_SPECS}
        row.update(P_out=12, nBl=10, Efficiency=0.75, totalpressureratio=1.4,
                   MassFlow=3.6, Power=100, is_boundary=0)
        row.update(changes)
        return row

    def write_samples(self, rows, path=None):
        path = path or self.config.workspace.training_csv
        pd.DataFrame(rows).to_csv(path, index=False)
        write_performance_data_metadata(path)

    def write_history(self, rows=None):
        ws = self.config.workspace
        stage = {"stage_id": "stage-a", "hv_policy_version": 3,
                 "runtime": self.config.to_dict()["runtime"]}
        (ws.checkpoint_meta_json.parent / "al_stage.json").write_text(json.dumps(stage))
        meta = {"stage_id": "stage-a", "hv_policy_version": 3, "completed_iters": 1,
                "performance_data_schema_version": PERFORMANCE_DATA_SCHEMA_VERSION,
                "total_pressure_definition": TOTAL_PRESSURE_DEFINITION}
        ws.checkpoint_meta_json.write_text(json.dumps(meta))
        pd.DataFrame(rows or [
            {"iter": 0, "true_hv": 0.1, "surrogate_hv": None, "stage_id": "stage-a", "hv_policy_version": 3},
            {"iter": 1, "true_hv": 0.2, "surrogate_hv": 0.3, "stage_id": "stage-a", "hv_policy_version": 3},
        ]).to_csv(ws.hv_history_csv, index=False)

    def test_missing_files_remain_unknown_and_no_files_created(self):
        snapshot = self.reader.read()
        self.assertIsNone(snapshot.samples["doe"].count)
        self.assertFalse(snapshot.hv)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_filters_pressure_and_nonfinite_values_without_mixing_sources(self):
        self.write_samples([self.row(), self.row(Efficiency=0.8),
                            self.row(P_out=13, Efficiency=0.99), self.row(Efficiency=float("nan")),
                            self.row(d2=0.52, is_boundary=1, Efficiency=0.7)])
        self.write_samples([self.row(Efficiency=0.9)], self.config.workspace.pool_checkpoint_csv)
        result = self.reader.read()
        doe = result.samples["doe"]
        self.assertEqual((doe.count, doe.excluded, doe.duplicates), (2, 2, 1))
        self.assertEqual(doe.best_efficiency, 0.8)
        self.assertEqual(result.samples["pool"].best_efficiency, 0.9)
        self.assertEqual(doe.rows[0][-1], 1)

    def test_rejects_old_schema_and_missing_columns(self):
        self.write_samples([self.row()])
        metadata = self.config.workspace.training_csv.with_suffix(".csv.meta.json")
        metadata.write_text('{"schema_version": 1}')
        self.assertIsNone(self.reader.read().samples["doe"].count)
        write_performance_data_metadata(self.config.workspace.training_csv)
        pd.DataFrame([{"Efficiency": 0.8}]).to_csv(self.config.workspace.training_csv, index=False)
        self.assertIsNone(self.reader.read().samples["doe"].count)

    def test_refresh_reflects_new_data_and_clears_deleted_or_partial_file(self):
        self.write_samples([self.row()])
        self.assertEqual(self.reader.read().samples["doe"].count, 1)
        self.write_samples([self.row(), self.row(d2=0.52)])
        self.assertEqual(self.reader.read().samples["doe"].count, 2)
        self.config.workspace.training_csv.write_text("Efficiency,totalpressureratio\n0.8,")
        self.assertIsNone(self.reader.read().samples["doe"].count)
        self.config.workspace.training_csv.unlink()
        self.assertIsNone(self.reader.read().samples["doe"].count)

    def test_changed_file_during_read_is_rejected(self):
        from impeller_app.core import live_results
        self.write_samples([self.row()])
        original = live_results._csv
        def racing_read(path):
            frame = original(path)
            if path == self.config.workspace.training_csv:
                with path.open("a") as output:
                    output.write("\n")
            return frame
        with patch.object(live_results, "_csv", side_effect=racing_read):
            snapshot = self.reader.read()
        self.assertIsNone(snapshot.samples["doe"].count)
        self.assertTrue(any("changed while reading" in issue.detail for issue in snapshot.issues))

    def test_hv_keeps_baseline_and_separates_prediction(self):
        self.write_history()
        snapshot = self.reader.read()
        self.assertEqual(snapshot.hv, [(0, 0.1, None), (1, 0.2, 0.3)])
        self.assertEqual(snapshot.stage, "stage-a")

    def test_hv_hides_uncommitted_iteration(self):
        self.write_history()
        path = self.config.workspace.checkpoint_meta_json
        meta = json.loads(path.read_text())
        meta["completed_iters"] = 0
        path.write_text(json.dumps(meta))
        self.assertEqual(self.reader.read().hv, [(0, 0.1, None)])

    def test_hv_rejects_mixed_stage_policy_and_duplicate_iteration(self):
        for column, value in (("stage_id", "another"), ("hv_policy_version", 2), ("iter", 0)):
            with self.subTest(column=column):
                self.write_history()
                frame = pd.read_csv(self.config.workspace.hv_history_csv)
                frame.loc[1, column] = value
                frame.to_csv(self.config.workspace.hv_history_csv, index=False)
                self.assertFalse(self.reader.read().hv)

    def test_hv_rejects_changed_pressure_or_missing_provenance(self):
        self.write_history()
        config = replace(self.config, runtime=replace(self.config.runtime, optimization_outlet_static_pressure_pa=11))
        self.assertFalse(LiveResultsReader(config).read().hv)
        self.config.workspace.checkpoint_meta_json.unlink()
        self.assertFalse(self.reader.read().hv)

    def test_monitor_does_not_write_to_existing_artifacts(self):
        self.write_samples([self.row()])
        self.write_history()
        before = {p: p.read_bytes() for p in Path(self.temp.name).iterdir()}
        self.reader.read()
        self.assertEqual(before, {p: p.read_bytes() for p in Path(self.temp.name).iterdir()})


if __name__ == "__main__":
    unittest.main()
