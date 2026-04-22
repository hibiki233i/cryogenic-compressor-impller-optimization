from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from impeller_app.config import AppConfig, RuntimeSettings, SolverPaths, WorkspacePaths
from impeller_app.core.active_learning import ActiveLearningService
from impeller_app.core.pareto import ParetoService
from impeller_app.runner.external import RunnerAPI
import NN_NSGA2_ActiveLearning_refactored as legacy_al


class ImpellerAppTests(unittest.TestCase):
    def make_config(self, root: Path) -> AppConfig:
        return AppConfig(
            solver=SolverPaths(
                powershell_exe=root / "pwsh.exe",
                geometry_script_path=root / "Run-GeometryMeshing.ps1",
                cfx_bin_dir=root / "cfx-bin",
                template_cfx=root / "BaseModel.cfx",
                template_cse=root / "Extract_Results.cse",
            ),
            workspace=WorkspacePaths(project_root=root),
            runtime=RuntimeSettings(),
        ).resolved()

    def test_validate_environment_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp))
            result = RunnerAPI(config).validate_environment()
            self.assertEqual(result.status, "failed")
            self.assertIn("powershell_exe", result.metrics["missing"])

    def test_config_round_trip_persists_paths_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "app-config.json"
            config = AppConfig(
                solver=SolverPaths(
                    powershell_exe=Path(r"C:\tools\pwsh.exe"),
                    geometry_script_path=Path(r"D:\work\Run-GeometryMeshing.ps1"),
                    cfx_bin_dir=Path(r"D:\ANSYS\CFX\bin"),
                    template_cfx=Path(r"D:\work\BaseModel.cfx"),
                    template_cse=Path(r"D:\work\Extract_Results.cse"),
                ),
                workspace=WorkspacePaths(
                    project_root=Path(r"D:\BOUNDYR"),
                    doe_runs_dir=Path(r"D:\BOUNDYR\Runs"),
                    active_learning_runs_dir=Path(r"D:\BOUNDYR\ActiveLearning_Runs"),
                    training_csv=Path(r"D:\BOUNDYR\Compressor_Training_Data.csv"),
                    pareto_export_dir=Path(r"D:\BOUNDYR\pareto_cft_cases"),
                ),
                runtime=RuntimeSettings(
                    cfx_cores=12,
                    doe_initial_samples=64,
                    doe_target_samples=128,
                    active_learning_additional_iters=3,
                    pareto_geom_safe_threshold=0.55,
                ),
            )

            saved_path = config.save(config_path)
            restored = AppConfig.load(saved_path)

            self.assertEqual(saved_path, config_path)
            self.assertEqual(restored.solver.geometry_script_path, Path(r"D:\work\Run-GeometryMeshing.ps1"))
            self.assertEqual(restored.workspace.project_root, Path(r"D:\BOUNDYR"))
            self.assertEqual(restored.workspace.pareto_export_dir, Path(r"D:\BOUNDYR\pareto_cft_cases"))
            self.assertEqual(restored.runtime.cfx_cores, 12)
            self.assertEqual(restored.runtime.doe_target_samples, 128)

    def test_recover_runs_rebuilds_training_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root)
            run_dir = root / "Runs" / "Run_000"
            run_dir.mkdir(parents=True)
            (run_dir / "CFX_Results.txt").write_text("0.71,1.91,10.0,0.40,2.01", encoding="utf-8")
            result = RunnerAPI(config).recover_runs()
            self.assertEqual(result.status, "succeeded")
            rebuilt = pd.read_csv(root / "Compressor_Training_Data.csv")
            self.assertEqual(len(rebuilt), 1)
            self.assertIn("MassFlow", rebuilt.columns)

    def test_build_geometry_command_rounds_nbl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root)
            sample = {
                "d1s": 0.35,
                "dH": 0.05,
                "beta1hb": 72.0,
                "beta1sb": 24.0,
                "d2": 0.47,
                "b2": 0.05,
                "beta2hb": 44.0,
                "beta2sb": 46.0,
                "Lz": 0.2,
                "t": 0.002,
                "TipClear": 0.001,
                "nBl": 10.7,
                "rake_te_s": -18.0,
                "P_out": 10.0,
            }
            cmd = RunnerAPI(config).build_geometry_command(root / "Runs" / "Run_000", sample)
            self.assertIn("-nBl", cmd)
            self.assertIn("11", cmd)
            self.assertNotIn("-P_out", cmd)

    def test_resume_from_checkpoint_reads_existing_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Compressor_Training_Data.csv").write_text(
                "d1s,dH,beta1hb,beta1sb,d2,b2,beta2hb,beta2sb,Lz,t,TipClear,nBl,rake_te_s,P_out,Efficiency,totalpressureratio,Power,MassFlow,is_boundary\n",
                encoding="utf-8",
            )
            meta = {"completed_iters": 3, "total_attempts": 5}
            (root / "al_checkpoint_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            config = self.make_config(root)
            result = ActiveLearningService(config).resume_from_checkpoint()
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.metrics["completed_iters"], 3)
            self.assertEqual(Path(result.metrics["pool_checkpoint_path"]).resolve(), (root / "al_training_pool_checkpoint.csv").resolve())
            self.assertFalse(result.metrics["pool_checkpoint_exists"])
            self.assertEqual(Path(result.metrics["checkpoint_meta_path"]).resolve(), (root / "al_checkpoint_meta.json").resolve())
            self.assertTrue(result.metrics["checkpoint_meta_exists"])

    def test_resume_from_checkpoint_uses_in_progress_iter_when_completed_iters_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Compressor_Training_Data.csv").write_text(
                "d1s,dH,beta1hb,beta1sb,d2,b2,beta2hb,beta2sb,Lz,t,TipClear,nBl,rake_te_s,P_out,Efficiency,totalpressureratio,Power,MassFlow,is_boundary\n",
                encoding="utf-8",
            )
            meta = {"completed_iters": 0, "in_progress_iter": 124}
            (root / "al_checkpoint_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            config = self.make_config(root)
            result = ActiveLearningService(config).resume_from_checkpoint()
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.metrics["completed_iters"], 0)
            self.assertEqual(result.metrics["in_progress_iter"], 124)
            self.assertEqual(result.metrics["effective_resume_iter"], 123)

    def test_split_with_fixed_testset_rejects_empty_training_data(self):
        df = pd.DataFrame(columns=legacy_al.VAR_NAMES + legacy_al.ALL_OUTPUT_NAMES + ["is_boundary"])
        with self.assertRaisesRegex(ValueError, "TRAINING_CSV 中没有可用于主动学习的样本"):
            legacy_al.split_with_fixed_testset(df)

    def test_runtime_path_overrides_apply_to_resume_and_pool_checkpoint_helpers(self):
        original_meta = legacy_al.CHECKPOINT_META_PATH
        original_pool = legacy_al.POOL_CHECKPOINT_CSV
        original_test = legacy_al.TEST_SET_CSV
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                meta_path = root / "al_checkpoint_meta.json"
                pool_path = root / "al_training_pool_checkpoint.csv"
                test_path = root / "fixed_test_set.csv"

                meta_path.write_text(json.dumps({"completed_iters": 0, "in_progress_iter": 5}), encoding="utf-8")
                pd.DataFrame(columns=legacy_al.VAR_NAMES + legacy_al.ALL_OUTPUT_NAMES + ["is_boundary"]).to_csv(pool_path, index=False)

                legacy_al.configure_runtime(
                    CHECKPOINT_META_PATH=str(meta_path),
                    POOL_CHECKPOINT_CSV=str(pool_path),
                    TEST_SET_CSV=str(test_path),
                )

                self.assertEqual(legacy_al.get_resume_iter(), 4)
                loaded = legacy_al.load_pool_checkpoint()
                self.assertIsNotNone(loaded)
                self.assertEqual(len(loaded), 0)
                self.assertEqual(legacy_al.TEST_SET_CSV, str(test_path))
        finally:
            legacy_al.configure_runtime(
                CHECKPOINT_META_PATH=original_meta,
                POOL_CHECKPOINT_CSV=original_pool,
                TEST_SET_CSV=original_test,
            )

    def test_run_active_learning_iteration_returns_failed_result_on_validation_error(self):
        class FailingLegacy:
            def get_resume_iter(self):
                return 0

            def main_multiobjective_active_learning(self, max_al_iters=None):
                raise ValueError("TRAINING_CSV 中没有可用于主动学习的样本。")

        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp))
            service = ActiveLearningService(config)
            service._legacy = FailingLegacy()
            result = service.run_active_learning_iteration(1)
            self.assertEqual(result.status, "failed")
            self.assertIn("TRAINING_CSV 中没有可用于主动学习的样本", result.message)

    def test_compute_pareto_front_without_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df = pd.DataFrame(
                [
                    [0.35, 0.05, 72.0, 24.0, 0.47, 0.05, 44.0, 46.0, 0.2, 0.002, 0.001, 10, -18.0, 10.0, 0.70, 2.0, 120.0, 4.0, 0],
                    [0.36, 0.05, 73.0, 24.5, 0.48, 0.05, 44.5, 46.5, 0.2, 0.002, 0.001, 10, -18.5, 10.0, 0.74, 2.1, 118.0, 4.1, 0],
                    [0.37, 0.05, 74.0, 25.0, 0.49, 0.05, 45.0, 47.0, 0.2, 0.002, 0.001, 10, -19.0, 10.0, 0.68, 1.9, 121.0, 4.0, 0],
                ],
                columns=[
                    "d1s", "dH", "beta1hb", "beta1sb", "d2", "b2", "beta2hb", "beta2sb", "Lz", "t", "TipClear",
                    "nBl", "rake_te_s", "P_out", "Efficiency", "totalpressureratio", "Power", "MassFlow", "is_boundary",
                ],
            )
            training = root / "Compressor_Training_Data.csv"
            df.to_csv(training, index=False)
            config = self.make_config(root)
            result = ParetoService(config).compute_pareto_front()
            self.assertEqual(result.status, "succeeded")
            self.assertTrue((root / "pareto_front_points.csv").exists())

    @mock.patch("impeller_app.runner.external.run_cfx_pipeline")
    @mock.patch("impeller_app.runner.external.subprocess.run")
    def test_run_doe_sample_uses_mocked_subprocess_and_cfx(self, mock_run, mock_cfx):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root)
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            mock_cfx.return_value = (True, {"Efficiency": 0.7, "PressureRatio": 1.8, "Power": 100.0, "MassFlow": 4.0, "totalpressureratio": 2.0}, "Success")
            sample = RunnerAPI(config).generate_lhs_samples(1)[0]
            result = RunnerAPI(config).run_doe_sample(0, sample)
            self.assertEqual(result.status, "succeeded")
            mock_run.assert_called_once()
            mock_cfx.assert_called_once()
            stored = pd.read_csv(root / "Compressor_Training_Data.csv")
            self.assertEqual(len(stored), 1)
            self.assertAlmostEqual(float(stored.iloc[0]["Efficiency"]), 0.7)

    @mock.patch("impeller_app.runner.external.subprocess.run")
    def test_export_cases_generates_mesh_files(self, mock_run):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root)
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")

            front = pd.DataFrame(
                [
                    {
                        "front_index": 3,
                        "engineering_rank": 1,
                        "engineering_score": 0.91,
                        "d1s": 0.35,
                        "dH": 0.05,
                        "beta1hb": 72.0,
                        "beta1sb": 24.0,
                        "d2": 0.47,
                        "b2": 0.05,
                        "beta2hb": 44.0,
                        "beta2sb": 46.0,
                        "Lz": 0.2,
                        "t": 0.002,
                        "TipClear": 0.001,
                        "nBl": 11,
                        "rake_te_s": -18.0,
                        "P_out": 10.0,
                        "Efficiency": 0.74,
                        "totalpressureratio": 2.1,
                        "Power": 118.0,
                        "MassFlow": 4.1,
                    }
                ]
            )
            front.to_csv(root / "pareto_engineering_ranked.csv", index=False)
            front.to_csv(root / "pareto_front_points.csv", index=False)
            templates = root / "Templates"
            templates.mkdir()
            (templates / "0908-2.cft").write_text("base", encoding="utf-8")
            (templates / "BaseModel.cft-batch").write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<Root>
  <dS>0</dS><d2>0</d2><b2>0</b2><DeltaZ>0</DeltaZ><nBl>0</nBl><dH>0</dH>
  <xTipInlet>0</xTipInlet><xTipOutlet>0</xTipOutlet><sLEH>0</sLEH><sLES>0</sLES><sTEH>0</sTEH><sTES>0</sTES>
  <Beta2><Value Index="0">0</Value><Value Index="1">0</Value></Beta2>
  <RakeTE><Value Index="1">0</Value></RakeTE>
  <Beta1><Value Index="0">0</Value><Value Index="1">0</Value></Beta1>
  <mFlow>0</mFlow><nRot>0</nRot>
</Root>
""",
                encoding="utf-8",
            )

            result = ParetoService(config).export_cases(top_n=1, force=True)

            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.metrics["case_count"], 1)
            mock_run.assert_called_once()
            case_dir = root / "pareto_cft_cases" / "ParetoCase_01_F03"
            self.assertTrue((case_dir / "geometry_parameters.csv").exists())
            report = json.loads((root / "pareto_cft_cases" / "export_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["count"], 1)
            self.assertEqual(report["failed_count"], 0)
            self.assertTrue(report["cases"][0]["mesh_generated"])


if __name__ == "__main__":
    unittest.main()
