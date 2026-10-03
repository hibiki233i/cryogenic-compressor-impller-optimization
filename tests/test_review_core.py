"""Functional checks for offline analysis and case inspection (no solvers)."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from cfx_convergence import REQUIRED
from design_variables import DEFAULT_VARIABLE_SPECS, variable_names, write_performance_data_metadata
from impeller_app.config import AppConfig, WorkspacePaths
from impeller_app.core.analysis import AnalysisReader, correlations, blade_groups, prediction_metrics, verified_pairs, filter_observations
from impeller_app.core.cases import CaseReader, CaseRecord, preview_file, safe_child
from impeller_app.core.review_config import load_review_config


class ReviewCoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = AppConfig(workspace=WorkspacePaths(project_root=self.root)).resolved()
        self.names = variable_names(DEFAULT_VARIABLE_SPECS)

    def row(self, **changes):
        row = {spec["name"]: (spec["lower"] + spec["upper"]) / 2 for spec in DEFAULT_VARIABLE_SPECS}
        row.update(nBl=10, P_out=12, Efficiency=0.75, totalpressureratio=1.4, MassFlow=3.6, Power=100, is_boundary=0)
        row.update(changes)
        return row

    def query(self, row=None, **changes):
        row = row or self.row()
        query = {name: row[name] for name in self.names}
        query.update(run_id="AL_Iter01_P1_A00001", iter=1, status="success", result_valid=True,
                     true_eff=row["Efficiency"], true_pr=row["totalpressureratio"], true_mf=row["MassFlow"],
                     pred_eff=0.77, pred_pr=1.5, pred_mf=3.5)
        query.update(changes)
        return query

    def test_correlation_excludes_categorical_blades_fixed_condition_and_constants(self):
        frame = pd.DataFrame([self.row(d1s=0.34 + i * 0.01, Efficiency=0.7 + i * 0.01, nBl=9 + i) for i in range(4)])
        result = correlations(frame, "Efficiency", ["d1s", "d2", "nBl"])
        self.assertEqual(result.variable.tolist(), ["d1s"])
        self.assertAlmostEqual(result.iloc[0].r, 1)
        self.assertEqual(len(blade_groups(frame, "Efficiency")), 4)
        self.assertTrue(blade_groups(frame, "Efficiency")["std"].isna().all())
        self.assertEqual(len(filter_observations(frame, "0", 10)), 1)
        self.assertEqual(len(correlations(frame.iloc[:2], "Efficiency", ["d1s"])), 0)

    def test_prediction_error_sign_and_constant_target_r2(self):
        pairs = pd.DataFrame([self.query()])
        metrics = prediction_metrics(pairs, "eff")
        self.assertAlmostEqual(metrics["bias"], 0.02)
        self.assertAlmostEqual(metrics["rmse"], 0.02)
        self.assertIsNone(metrics["r2"])
        metrics = prediction_metrics(pd.DataFrame([self.query(), self.query()]), "eff")
        self.assertIsNone(metrics["r2"])

    def test_pair_truth_must_match_v2_reference_not_merely_have_true_columns(self):
        reference = pd.DataFrame([self.row()])
        audit = pd.DataFrame([self.query(), self.query(run_id="wrong", true_pr=9),
                              self.query(run_id="other_pressure", P_out=11), self.query(run_id="failed", status="failed"),
                              self.query(run_id="invalid", result_valid=False)])
        pairs = verified_pairs(audit, reference, self.names, "legacy", "query")
        self.assertEqual(pairs.run_id.tolist(), ["AL_Iter01_P1_A00001"])

    def test_named_stage_does_not_adopt_untagged_or_other_stage_predictions(self):
        audit = pd.DataFrame([self.query(stage_id="fresh"), self.query(run_id="wrong", stage_id="old")])
        reference = pd.DataFrame([self.row()])
        pairs = verified_pairs(audit, reference, self.names, "fresh", "query")
        self.assertEqual(len(pairs), 1)
        self.assertTrue(verified_pairs(pd.DataFrame([self.query()]), reference, self.names, "fresh", "query").empty)

    def test_latest_duplicate_query_and_fixed_tests_remain_separate(self):
        reference = pd.DataFrame([self.row()])
        audit = pd.DataFrame([self.query(pred_eff=0.72), self.query(pred_eff=0.81)])
        pairs = verified_pairs(audit, reference, self.names, "legacy", "query")
        self.assertEqual(pairs.pred_eff.tolist(), [0.81])
        fixed = pd.DataFrame([self.query(test_index=0), self.query(test_index=0, iter=2)])
        self.assertEqual(len(verified_pairs(fixed, reference, self.names, "legacy", "fixed")), 2)

    def test_analysis_rejects_schema_without_destroying_audits(self):
        ws = self.config.workspace
        pd.DataFrame([self.row()]).to_csv(ws.pool_checkpoint_csv, index=False)
        pd.DataFrame([self.query()]).to_csv(ws.al_query_validation_csv, index=False)
        snapshot = AnalysisReader(self.config).read()
        self.assertEqual(len(snapshot.audits["query"]), 1)
        self.assertTrue(snapshot.pairs["query"].empty)
        write_performance_data_metadata(ws.pool_checkpoint_csv)
        snapshot = AnalysisReader(self.config).read()
        self.assertEqual(len(snapshot.pairs["query"]), 1)

    def test_case_inventory_uses_source_and_latest_success_no_permanent_failure(self):
        records = []
        for source in ("doe", "active_learning"):
            records.append({"source": source, "run_id": "shared", "status": "failed", "failure_stage": "infrastructure", **self.row()})
        records.append({"source": "doe", "run_id": "shared", "status": "succeeded", **self.row()})
        records.append({"source": "doe", "run_id": "../escape", "status": "failed"})
        pd.DataFrame(records).to_csv(self.config.workspace.failure_records_csv, index=False)
        inventory = CaseReader(self.config).inventory()
        by_source = {r.source: r for r in inventory.records}
        self.assertEqual(len(inventory.records), 2)
        self.assertEqual(by_source["doe"].status, "succeeded")
        self.assertEqual(by_source["doe"].failure_stage, "")
        self.assertEqual(by_source["active_learning"].failure_stage, "infrastructure")
        self.assertFalse(by_source["doe"].exists)
        self.assertTrue(any("Invalid run ID" in issue for issue in inventory.issues))

    def test_discovers_unregistered_case_and_does_not_call_it_failed(self):
        path = self.config.workspace.doe_runs_dir / "Run_001"
        path.mkdir(parents=True)
        inventory = CaseReader(self.config).inventory()
        self.assertEqual(len(inventory.records), 1)
        self.assertEqual(inventory.records[0].status, "unknown")

    def test_recent_cancellation_is_not_overridden_by_old_failure_audit(self):
        run_id = "AL_Iter01_P1_A00001"
        pd.DataFrame([{"source": "active_learning", "run_id": run_id, "status": "failed",
                       "failure_stage": "cfx_solver", "updated_at_utc": "2026-01-01T00:00:00Z", **self.row()}]).to_csv(self.config.workspace.failure_records_csv, index=False)
        pd.DataFrame([self.query(status="canceled", completed_at_utc="2026-01-02T00:00:00Z")]).to_csv(self.config.workspace.al_query_validation_csv, index=False)
        record = CaseReader(self.config).inventory().records[0]
        self.assertEqual(record.status, "canceled")
        self.assertEqual(record.audit["status"], "failed")

    def test_cross_stage_same_id_is_not_joined_to_current_audit(self):
        stage = self.root / "another_stage"
        case = stage / "ActiveLearning_Runs" / "AL_Iter01_P1_A00001"
        case.mkdir(parents=True)
        (stage / "al_stage.json").write_text('{"stage_id":"another"}')
        pd.DataFrame([{"source": "active_learning", "run_id": case.name, "status": "failed", **self.row()}]).to_csv(self.config.workspace.failure_records_csv, index=False)
        config = replace(self.config, workspace=replace(self.config.workspace, active_learning_runs_dir=stage))
        inventory = CaseReader(config).inventory()
        self.assertEqual(inventory.records[0].status, "unknown")
        self.assertTrue(any("associations disabled" in x for x in inventory.issues))

    def test_preview_is_bounded_and_does_not_execute_scripts_or_load_results(self):
        (self.root / "log.out").write_text("line\n" * 100000, encoding="utf-8")
        (self.root / "script.ps1").write_text("Write-Host harmless-text", encoding="utf-8")
        (self.root / "big.res").write_bytes(b"binary")
        preview = preview_file(self.root, "log.out")
        self.assertTrue(preview.truncated)
        self.assertEqual(len(preview.text.splitlines()), 800)
        self.assertEqual(preview_file(self.root, "script.ps1").kind, "text")
        self.assertEqual(preview_file(self.root, "big.res").kind, "binary")
        with self.assertRaises(ValueError):
            preview_file(self.root, "../elsewhere.txt")
        with self.assertRaises(ValueError):
            safe_child(self.root, ".")

    def test_cfx_detail_uses_convergence_proof_and_full_impeller_units(self):
        case = self.root / "Run_000"
        case.mkdir()
        (case / "input_parameters.json").write_text(json.dumps(self.row()))
        text = "OUTER LOOP ITERATION = 10\n| Equation | Rate | RMS Res | Max Res | Linear | State |\n"
        text += "".join(f"| {eq} | 0.95 | 1e-5 | 1e-1 | 2 | OK |\n" for eq in REQUIRED)
        text += "This run of the ANSYS CFX Solver has finished.\n"
        out = case / "result.out"
        out.write_text(text)
        (case / "result.res").write_bytes(b"unit-test-fixture")
        (case / "CFX_Results.txt").write_text("0.75,1.1,10,0.36,1.4")
        (case / "CFX_Results.meta.json").write_text(json.dumps({"schema_version": 2, "total_pressure_frame": "stationary", "total_pressure_averaging": "massFlowAve"}))
        record = CaseRecord("doe", case.name, case, exists=True)
        detail = CaseReader(self.config).detail(record)
        self.assertEqual(detail.result_state, "verified")
        self.assertEqual(detail.metrics["Power"], 100)
        self.assertAlmostEqual(detail.metrics["MassFlow"], 3.6)
        self.assertEqual(detail.metrics["totalpressureratio"], 1.4)
        self.assertTrue(detail.convergence["accepted"])
        (case / "input_parameters.json").write_text(json.dumps(self.row(nBl=11.4)))
        detail = CaseReader(self.config).detail(record)
        self.assertEqual(detail.effective_blades, 11)
        self.assertAlmostEqual(detail.metrics["MassFlow"], 3.96)
        self.assertEqual(detail.parameters["nBl"], 11.4)
        (case / "CFX_INVALID.json").write_text("{}")
        detail = CaseReader(self.config).detail(record)
        self.assertFalse(detail.metrics)
        self.assertEqual(detail.result_state, "unverified")

    def test_readonly_config_resolution_is_relative_to_config_file(self):
        folder = self.root / "review"
        folder.mkdir()
        file = folder / "stage_config.json"
        file.write_text(json.dumps({"workspace": {"project_root": ".", "training_csv": "custom.csv"}}))
        result = load_review_config(file)
        self.assertEqual(result.workspace.project_root, folder)
        self.assertEqual(result.workspace.training_csv, folder / "custom.csv")
        file.write_text("{}")
        with self.assertRaises(ValueError):
            load_review_config(file)

    def test_inventory_and_analysis_do_not_create_missing_files(self):
        AnalysisReader(self.config).read()
        CaseReader(self.config).inventory()
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
