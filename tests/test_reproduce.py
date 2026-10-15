"""Exercise the offline entry point with real training and no solver access."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class OfflineReproductionTests(unittest.TestCase):
    def test_documented_default_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result"
            run = subprocess.run(
                [sys.executable, "-m", "impeller_app.reproduce", "--config",
                 str(ROOT / "examples/offline/config.json"), "--output-dir", str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["samples"], 384)
            self.assertEqual(summary["analytic_front_size"], 50)
            for key, expected, tolerance in (
                ("analytic_max_efficiency", 0.7807738769, 1e-8),
                ("analytic_max_total_pressure_ratio", 2.01885, 1e-8),
            ):
                self.assertAlmostEqual(summary[key], expected, delta=tolerance)
            for key, expected, tolerance in (
                ("fixed_test_rmse_eff", 0.00384668, 0.001),
                ("fixed_test_rmse_pr", 0.02401822, 0.01),
                ("fixed_test_rmse_mf", 0.04061173, 0.02),
                ("surrogate_hv", 0.15848537, 0.03),
            ):
                self.assertAlmostEqual(summary["nsga2"][key], expected, delta=tolerance)
            self.assertEqual(summary["nsga2"]["train_samples"], 326)
            self.assertEqual(summary["nsga2"]["test_samples"], 58)
            self.assertTrue(1 <= summary["nsga2"]["front_size"] <= 48)
            self.assertAlmostEqual(summary["selection"]["engineering_score"], 0.8300447961, delta=1e-6)
            # POLA-1's analytic reference has no model/BLAS randomness. Its
            # values and all 50 observed Pareto rows use tight CSV tolerances.
            for key, expected in (("Efficiency", 0.7486677356),
                                  ("totalpressureratio", 1.6819375),
                                  ("MassFlow", 4.49925)):
                self.assertAlmostEqual(summary["selection"]["true_outputs"][key], expected, delta=1e-8)
            generated = pd.read_csv(output / "Compressor_Training_Data.csv")
            pd.testing.assert_frame_equal(
                generated, pd.read_csv(ROOT / "examples/offline/Compressor_Training_Data.csv"),
                check_exact=False, rtol=0, atol=1e-10)
            ranked = pd.read_csv(output / "pareto_engineering_ranked.csv")
            self.assertEqual(len(ranked), 50)
            np.testing.assert_allclose(ranked.iloc[0][["Efficiency", "totalpressureratio", "MassFlow"]],
                                       [0.7486677356, 1.6819375, 4.49925], rtol=0, atol=1e-8)
            predicted = pd.read_csv(output / "nsga2_surrogate_pareto.csv")
            self.assertEqual(len(predicted), summary["nsga2"]["front_size"])
            # The runner already fixes CPU threads, deterministic torch and seeds.
            # Discrete NSGA-II membership can differ with torch/BLAS versions
            # (POLA-1 recorded 30 and 31 rows), so retain its 1..population bound
            # and compare the objective extrema as well as RMSE and HV above.
            # 0.01 efficiency / 0.05 total PR permit small model drift; losing
            # either end of the tradeoff must still fail. Not CFD tolerances.
            for column, expected, tolerance in (("pred_Efficiency", 0.775687, 0.01),
                                                ("pred_totalpressureratio", 2.040334, 0.05)):
                self.assertTrue(np.isfinite(predicted[column]).all(), column)
                self.assertAlmostEqual(predicted[column].max(), expected, delta=tolerance)
            np.testing.assert_allclose(predicted.surrogate_hv, summary["nsga2"]["surrogate_hv"],
                                       rtol=0, atol=1e-12)

    def test_seeded_pipeline_exports_reproducible_results_without_solvers(self):
        if importlib.util.find_spec("impeller_app.reproduce") is None:
            self.fail("offline reproduction entry point is missing")
        # Any accidental external command must fail even on an engineering PC.
        script = """
import subprocess, sys
from unittest.mock import patch
from impeller_app.reproduce import main
with patch.object(subprocess, 'Popen', side_effect=AssertionError('external command forbidden')):
    sys.exit(main())
"""
        env = dict(os.environ, IMPELLER_BOUNDARY_FLOW_G_S="999",
                   IMPELLER_DESIGN_VARIABLES_PATH="nonexistent-user-config.json")
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({"seed": 42, "samples": 128,
                                         "max_epochs": 80, "patience": 40,
                                         "pop_size": 24, "n_gen": 8}))
            results = []
            for name in ("first", "second"):
                output = Path(tmp) / name
                run = subprocess.run(
                    [sys.executable, "-c", script, "--config", str(config),
                     "--output-dir", str(output)], cwd=ROOT, env=env,
                    capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["data_source"], "synthetic_analytic_v1")
                self.assertEqual(summary["nsga2"]["training_scope"], "doe_only")
                self.assertLessEqual(summary["nsga2"]["epochs"], 80)
                self.assertEqual(summary["nsga2"]["train_samples"] +
                                 summary["nsga2"]["test_samples"], 128)
                predicted = pd.read_csv(output / "nsga2_surrogate_pareto.csv")
                ranked = pd.read_csv(output / "pareto_engineering_ranked.csv")
                self.assertGreater(len(predicted), 0)
                self.assertGreater(len(ranked), 0)
                self.assertTrue((predicted.P_out == 12).all())
                self.assertTrue((ranked.MassFlow >= 3.6).all())
                self.assertEqual(ranked.engineering_rank.tolist(), list(range(1, len(ranked)+1)))
                self.assertEqual(summary["selection"]["front_index"], int(ranked.iloc[0].front_index))
                self.assertTrue((output / "nsga2_doe_model" / "surrogate_model_manifest.json").exists())
                results.append((summary, predicted, ranked))
            self.assertEqual(results[0][0]["nsga2"]["pool_digest"], results[1][0]["nsga2"]["pool_digest"])
            np.testing.assert_allclose(results[0][1], results[1][1], rtol=0, atol=1e-7)
            np.testing.assert_allclose(results[0][2], results[1][2], rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
