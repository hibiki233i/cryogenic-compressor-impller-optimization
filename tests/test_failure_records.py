from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from failure_records import (
    classify_cfx_failure_stage,
    classify_geometry_failure_stage,
    load_failure_training_sets,
    load_run_outcome,
    record_run_outcome,
)


VARIABLE_NAMES = [f"g{i}" for i in range(13)] + ["P_out"]
GEOMETRY_NAMES = VARIABLE_NAMES[:-1]


class FailureRecordTests(unittest.TestCase):
    def sample(self, p_out=12.0):
        return {name: float(idx + 1) for idx, name in enumerate(GEOMETRY_NAMES)} | {
            "P_out": float(p_out)
        }

    def test_system_failures_are_not_mapped_to_physical_negative_stages(self):
        self.assertEqual(
            classify_geometry_failure_stage("exit 1", "License checkout failed"),
            "infrastructure",
        )
        self.assertEqual(
            classify_geometry_failure_stage("geometry command returned exit 1"),
            "infrastructure",
        )
        self.assertEqual(
            classify_cfx_failure_stage("CFX-Post 后处理失败。Exit Code: 1"),
            "postprocess",
        )
        self.assertEqual(
            classify_cfx_failure_stage("CFD 计算发散或崩溃。Exit Code: 1"),
            "cfx_solver",
        )
        self.assertEqual(
            classify_cfx_failure_stage("CFX FATAL OVERFLOW. Exit Code: 2"),
            "fatal_overflow",
        )

    def test_confirmation_requires_consecutive_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures.csv"
            first = record_run_outcome(
                path, VARIABLE_NAMES, self.sample(), source="doe", run_id="run",
                status="failed", failure_stage="cfx_solver",
            )
            self.assertFalse(first["confirmed"])
            record_run_outcome(
                path, VARIABLE_NAMES, self.sample(), source="doe", run_id="run",
                status="succeeded",
            )
            after_success = record_run_outcome(
                path, VARIABLE_NAMES, self.sample(), source="doe", run_id="run",
                status="failed", failure_stage="cfx_solver",
            )
            self.assertFalse(after_success["confirmed"])
            repeated = record_run_outcome(
                path, VARIABLE_NAMES, self.sample(), source="doe", run_id="run",
                status="failed", failure_stage="cfx_solver",
            )
            self.assertTrue(repeated["confirmed"])
            self.assertEqual(repeated["consecutive_failure_count"], 2)

    def test_fatal_overflow_is_immediately_confirmed_for_batch_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures.csv"
            record = record_run_outcome(
                path,
                VARIABLE_NAMES,
                self.sample(),
                source="doe",
                run_id="overflow",
                status="failed",
                failure_stage="fatal_overflow",
            )
            self.assertTrue(record["confirmed"])

    def test_blockage_outcome_can_be_reloaded_as_confirmed_terminal_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures.csv"
            record_run_outcome(
                path,
                VARIABLE_NAMES,
                self.sample(),
                source="doe",
                run_id="Run_001",
                status="failed",
                failure_stage="blockage",
                reason="进出口持续100%堵塞，提前终止",
            )

            record = load_run_outcome(
                path, source="doe", run_id="Run_001"
            )

            self.assertIsNotNone(record)
            self.assertEqual(record["failure_stage"], "blockage")
            self.assertTrue(record["confirmed"])

    def test_loader_separates_feature_spaces_and_excludes_system_faults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures.csv"
            record_run_outcome(
                path, VARIABLE_NAMES, self.sample(8), source="doe", run_id="geom",
                status="failed", failure_stage="geometry_mesh", confirmed=True,
            )
            record_run_outcome(
                path, VARIABLE_NAMES, self.sample(12), source="doe", run_id="solver",
                status="failed", failure_stage="cfx_solver", confirmed=True,
            )
            record_run_outcome(
                path, VARIABLE_NAMES, self.sample(11), source="doe", run_id="blockage",
                status="failed", failure_stage="blockage", confirmed=True,
            )
            record_run_outcome(
                path, VARIABLE_NAMES, self.sample(10), source="doe", run_id="overflow",
                status="failed", failure_stage="fatal_overflow",
            )
            record_run_outcome(
                path, VARIABLE_NAMES, self.sample(13), source="doe", run_id="post",
                status="failed", failure_stage="postprocess", confirmed=True,
            )

            geometry, operating, records = load_failure_training_sets(
                path, VARIABLE_NAMES, GEOMETRY_NAMES,
            )
            self.assertEqual(geometry.shape, (1, 13))
            self.assertEqual(operating.shape, (2, 14))
            self.assertEqual(set(operating[:, -1]), {10.0, 11.0})
            self.assertEqual(len(records), 5)

    def test_successful_point_suppresses_matching_negative_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures.csv"
            record_run_outcome(
                path, VARIABLE_NAMES, self.sample(), source="doe", run_id="failed",
                status="failed", failure_stage="blockage", confirmed=True,
            )
            record_run_outcome(
                path, VARIABLE_NAMES, self.sample(), source="doe", run_id="succeeded",
                status="succeeded",
            )
            geometry, operating, _ = load_failure_training_sets(
                path, VARIABLE_NAMES, GEOMETRY_NAMES,
            )
            self.assertEqual(geometry.shape, (0, 13))
            self.assertEqual(operating.shape, (0, 14))


if __name__ == "__main__":
    unittest.main()
