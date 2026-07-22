from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from design_variables import (
    PERFORMANCE_DATA_SCHEMA_VERSION,
    TOTAL_PRESSURE_DEFINITION,
    write_performance_data_metadata,
)
from impeller_app.config import AppConfig, RuntimeSettings, SolverPaths, WorkspacePaths
from impeller_app.core.active_learning import ActiveLearningService
from impeller_app.core.pareto import ParetoService
from impeller_app.models import TaskResult
from impeller_app.runner.external import RunnerAPI
from failure_records import load_failure_training_sets, record_run_outcome
import NN_NSGA2_ActiveLearning_refactored as legacy_al


class ImpellerAppTests(unittest.TestCase):
    def test_failure_records_require_consecutive_failure_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure_records.csv"
            names = list(legacy_al.VAR_NAMES)
            sample = dict(zip(names, (legacy_al.L_BOUNDS + legacy_al.U_BOUNDS) / 2.0))

            first = record_run_outcome(
                path, names, sample, source="doe", run_id="Run_001",
                status="failed", failure_stage="cfx_solver", reason="diverged",
            )
            self.assertFalse(first["confirmed"])
            record_run_outcome(
                path, names, sample, source="doe", run_id="Run_001",
                status="succeeded",
            )
            after_success = record_run_outcome(
                path, names, sample, source="doe", run_id="Run_001",
                status="failed", failure_stage="cfx_solver", reason="diverged",
            )
            self.assertFalse(after_success["confirmed"])
            repeated = record_run_outcome(
                path, names, sample, source="doe", run_id="Run_001",
                status="failed", failure_stage="cfx_solver", reason="diverged",
            )
            self.assertTrue(repeated["confirmed"])
            self.assertEqual(repeated["consecutive_failure_count"], 2)

    def test_failure_loader_separates_geometry_and_operating_feature_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure_records.csv"
            names = list(legacy_al.VAR_NAMES)
            geometry_names = list(legacy_al.GEOMETRY_VAR_NAMES)
            midpoint = (legacy_al.L_BOUNDS + legacy_al.U_BOUNDS) / 2.0
            sample = dict(zip(names, midpoint))

            record_run_outcome(
                path, names, sample, source="doe", run_id="geom",
                status="failed", failure_stage="geometry_mesh", confirmed=True,
            )
            solver_sample = dict(sample)
            solver_sample["P_out"] = 12.0
            record_run_outcome(
                path, names, solver_sample, source="doe", run_id="solver",
                status="failed", failure_stage="cfx_solver", confirmed=True,
            )
            blockage_sample = dict(sample)
            blockage_sample["P_out"] = 11.9
            record_run_outcome(
                path, names, blockage_sample, source="doe", run_id="blockage",
                status="failed", failure_stage="blockage", confirmed=True,
            )
            record_run_outcome(
                path, names, sample, source="doe", run_id="post",
                status="failed", failure_stage="postprocess", confirmed=True,
            )

            geometry, operating, records = load_failure_training_sets(
                path, names, geometry_names,
            )
            self.assertEqual(geometry.shape, (1, 13))
            self.assertEqual(operating.shape, (1, 14))
            self.assertEqual(float(operating[0, legacy_al.P_OUT_IDX]), 11.9)
            self.assertEqual(len(records), 4)

    def test_cfx_post_template_uses_stationary_frame_total_pressure(self):
        template = Path(__file__).resolve().parents[1] / "cfx_post" / "Extract_Results.cse"
        content = template.read_text(encoding="utf-8")
        self.assertIn("Total Pressure in Stn Frame", content)
        self.assertNotIn("Total Pressure in Rel Frame", content)

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
            gui_turbogrid_template = Path(r"F:\optimazition\Templates\new_base.tst")
            config = AppConfig(
                solver=SolverPaths(
                    powershell_exe=Path(r"C:\tools\pwsh.exe"),
                    geometry_script_path=Path(r"D:\work\Run-GeometryMeshing.ps1"),
                    cfx_bin_dir=Path(r"D:\ANSYS\CFX\bin"),
                    template_cfx=Path(r"D:\work\BaseModel.cfx"),
                    template_cse=Path(r"D:\work\Extract_Results.cse"),
                    turbogrid_template=gui_turbogrid_template,
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
                    default_boundary_flow_g_s=4.2,
                    default_min_d2_d1s_gap=0.08,
                ),
            )

            saved_path = config.save(config_path)
            restored = AppConfig.load(saved_path)

            self.assertEqual(saved_path, config_path)
            self.assertEqual(restored.solver.geometry_script_path, Path(r"D:\work\Run-GeometryMeshing.ps1"))
            self.assertEqual(restored.solver.turbogrid_template, gui_turbogrid_template)
            self.assertEqual(restored.workspace.project_root, Path(r"D:\BOUNDYR"))
            self.assertEqual(restored.workspace.pareto_export_dir, Path(r"D:\BOUNDYR\pareto_cft_cases"))
            self.assertEqual(restored.runtime.cfx_cores, 12)
            self.assertEqual(restored.runtime.doe_target_samples, 128)
            self.assertEqual(restored.runtime.default_boundary_flow_g_s, 4.2)
            self.assertEqual(restored.runtime.default_min_d2_d1s_gap, 0.08)
            self.assertEqual(
                restored.runtime.optimization_outlet_static_pressure_pa,
                12.0,
            )

    def test_recover_runs_rebuilds_training_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root)
            run_dir = root / "Runs" / "Run_000"
            run_dir.mkdir(parents=True)
            (run_dir / "CFX_Results.txt").write_text("0.71,1.91,10.0,0.40,2.01", encoding="utf-8")
            (run_dir / "CFX_Results.meta.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "total_pressure_frame": "stationary",
                        "total_pressure_averaging": "massFlowAve",
                    }
                ),
                encoding="utf-8",
            )
            result = RunnerAPI(config).recover_runs()
            self.assertEqual(result.status, "succeeded")
            rebuilt = pd.read_csv(root / "Compressor_Training_Data.csv")
            self.assertEqual(len(rebuilt), 1)
            self.assertIn("MassFlow", rebuilt.columns)

    @mock.patch("impeller_app.runner.external.run_cfx_pipeline")
    def test_recover_runs_reprocesses_res_without_result_txt(self, mock_run_cfx_pipeline):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root)
            run_dir = root / "Runs" / "Run_000"
            run_dir.mkdir(parents=True)
            (run_dir / "Impeller_001.res").write_text("placeholder", encoding="utf-8")
            mock_run_cfx_pipeline.return_value = (
                True,
                {
                    "Efficiency": 0.72,
                    "PressureRatio": 1.93,
                    "Power": 120.0,
                    "MassFlow": 4.1,
                    "totalpressureratio": 2.03,
                },
                "Success",
            )

            result = RunnerAPI(config).recover_runs()

            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.metrics["reposted_runs"], 1)
            self.assertEqual(result.metrics["partial_runs"], 0)
            rebuilt = pd.read_csv(root / "Compressor_Training_Data.csv")
            self.assertEqual(len(rebuilt), 1)
            self.assertEqual(rebuilt.loc[0, "Efficiency"], 0.72)
            mock_run_cfx_pipeline.assert_called_once()

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
            template_flag = cmd.index("-TurboGridTemplate")
            self.assertEqual(
                Path(cmd[template_flag + 1]),
                config.solver.turbogrid_template,
            )

    def test_geometry_script_requires_gui_turbogrid_template(self):
        script = Path(__file__).resolve().parents[1] / "Run-GeometryMeshing.ps1"
        content = script.read_text(encoding="utf-8-sig")
        self.assertIn("[Parameter(Mandatory = $true)]", content)
        self.assertIn("[string]$TurboGridTemplate,", content)
        self.assertNotIn(
            '$TurboGridTemplate = "F:\\optimazition\\Templates\\BaseMeshing.tst"',
            content,
        )
        self.assertNotIn('$value.SetAttribute("Caption", "Impeller")', content)
        self.assertNotIn('$value.InnerText = "3"', content)
        self.assertIn('$selectedComponents = @($components.SelectNodes("Value"))', content)

    @mock.patch.object(RunnerAPI, "run_doe_sample")
    def test_doe_batch_retries_existing_failed_run_before_new_index(self, mock_run_doe_sample):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root)
            config.runtime.doe_initial_samples = 1
            config.runtime.doe_target_samples = 1
            (config.workspace.doe_runs_dir / "Run_000").mkdir(parents=True)
            mock_run_doe_sample.return_value = TaskResult(
                status="succeeded",
                message="retry succeeded",
            )

            result = RunnerAPI(config).run_doe_batch()

            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.metrics["completed_runs"], 1)
            args, kwargs = mock_run_doe_sample.call_args
            self.assertEqual(args[0], 0)
            self.assertTrue(kwargs["force"])

    def test_resume_from_checkpoint_reads_existing_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Compressor_Training_Data.csv").write_text(
                "d1s,dH,beta1hb,beta1sb,d2,b2,beta2hb,beta2sb,Lz,t,TipClear,nBl,rake_te_s,P_out,Efficiency,totalpressureratio,Power,MassFlow,is_boundary\n",
                encoding="utf-8",
            )
            meta = {
                "completed_iters": 3,
                "total_attempts": 5,
                "performance_data_schema_version": PERFORMANCE_DATA_SCHEMA_VERSION,
                "total_pressure_definition": TOTAL_PRESSURE_DEFINITION,
            }
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
            meta = {
                "completed_iters": 0,
                "in_progress_iter": 124,
                "performance_data_schema_version": PERFORMANCE_DATA_SCHEMA_VERSION,
                "total_pressure_definition": TOTAL_PRESSURE_DEFINITION,
            }
            (root / "al_checkpoint_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            config = self.make_config(root)
            result = ActiveLearningService(config).resume_from_checkpoint()
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.metrics["completed_iters"], 0)
            self.assertEqual(result.metrics["in_progress_iter"], 124)
            self.assertEqual(result.metrics["effective_resume_iter"], 123)

    def test_query_log_prevents_attempt_id_reuse_after_stale_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            query_path = Path(tmp) / "al_query_validation.csv"
            pd.DataFrame(
                [
                    {"run_id": "AL_Iter11_P3_A00045", "status": "failed"},
                    {"run_id": "AL_Iter11_P4_A00046", "status": "submitted"},
                    {"run_id": "AL_Iter10_P1_A00041", "status": "success"},
                ]
            ).to_csv(query_path, index=False)
            attempts, successes = legacy_al.recover_query_counters(
                str(query_path),
                total_attempts=44,
                total_success=0,
            )
            self.assertEqual(attempts, 46)
            self.assertEqual(successes, 1)

    def test_split_with_fixed_testset_rejects_empty_training_data(self):
        df = pd.DataFrame(columns=legacy_al.VAR_NAMES + legacy_al.ALL_OUTPUT_NAMES + ["is_boundary"])
        with self.assertRaisesRegex(ValueError, "TRAINING_CSV 中没有可用于主动学习的样本"):
            legacy_al.split_with_fixed_testset(df)

    def test_nsga2_geometry_decisions_are_expanded_at_fixed_operating_point(self):
        geometry = legacy_al.L_BOUNDS[legacy_al.GEOMETRY_VAR_IDX][None, :]
        full = legacy_al.expand_geometry_decisions(geometry, fixed_p_out=12.0)
        self.assertEqual(full.shape, (1, len(legacy_al.VAR_NAMES)))
        self.assertEqual(full[0, legacy_al.P_OUT_IDX], 12.0)

    def test_active_learning_candidate_quota_is_two_plus_one_plus_one(self):
        midpoint = (legacy_al.L_BOUNDS + legacy_al.U_BOUNDS) / 2.0
        nbl_idx = legacy_al.VAR_NAMES.index("nBl")
        vary_idx = legacy_al.VAR_NAMES.index("d1s")

        candidates = np.tile(midpoint, (6, 1))
        candidates[:, vary_idx] = np.linspace(
            legacy_al.L_BOUNDS[vary_idx],
            legacy_al.U_BOUNDS[vary_idx],
            len(candidates),
        )
        candidates[:, nbl_idx] = [9, 9, 10, 11, 12, 12]
        candidates = legacy_al.pin_optimization_operating_point(candidates)

        pool = np.tile(midpoint, (11, 1))
        pool[:, vary_idx] = np.linspace(
            legacy_al.L_BOUNDS[vary_idx],
            legacy_al.U_BOUNDS[vary_idx],
            len(pool),
        )
        pool[:, legacy_al.VAR_NAMES.index("beta1hb")] = (
            legacy_al.L_BOUNDS[legacy_al.VAR_NAMES.index("beta1hb")]
        )
        pool[:, nbl_idx] = [9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11]
        pool = legacy_al.pin_optimization_operating_point(pool)

        scaler = MinMaxScaler().fit(
            np.vstack([legacy_al.L_BOUNDS, legacy_al.U_BOUNDS])
        )
        ehvi = np.array([0.9, 0.8, 0.2, 0.1, 0.05, 0.04])
        uncertainty = np.full((len(candidates), 3), 0.01)
        uncertainty[2] = 0.5
        selected, labels = legacy_al.select_candidates_diverse(
            acq_X=candidates,
            ehvi_vals=ehvi,
            scaler_X=scaler,
            acq_info={
                "valid_mask": np.ones(len(candidates), dtype=bool),
                "pred_std_norm": uncertainty,
            },
            X_pool_raw=pool,
            n_pick=4,
            min_dist_norm=0.01,
            n_ehvi=2,
        )

        self.assertEqual(len(selected), 4)
        self.assertTrue(labels[0].startswith("Pareto/EHVI#1"))
        self.assertTrue(labels[1].startswith("Pareto/EHVI#2"))
        self.assertTrue(labels[2].startswith("最大不确定性"))
        self.assertTrue(labels[3].startswith("欠采样nBl/空间填充"))
        self.assertEqual(int(round(selected[3][nbl_idx])), 12)

    def test_failure_classifiers_exclude_nbl_and_are_nbl_invariant(self):
        midpoint = (legacy_al.L_BOUNDS + legacy_al.U_BOUNDS) / 2.0
        nbl_idx = legacy_al.VAR_NAMES.index("nBl")
        vary_idx = legacy_al.VAR_NAMES.index("d1s")

        successful = np.tile(midpoint, (12, 1))
        successful[:, vary_idx] = np.linspace(
            legacy_al.L_BOUNDS[vary_idx], midpoint[vary_idx], len(successful)
        )
        successful[:, nbl_idx] = np.resize([9, 10, 11, 12], len(successful))

        failed = np.tile(midpoint, (4, 1))
        failed[:, vary_idx] = np.linspace(
            midpoint[vary_idx] + 1e-4,
            legacy_al.U_BOUNDS[vary_idx],
            len(failed),
        )
        failed[:, nbl_idx] = [12, 12, 11, 12]

        operating_clf = legacy_al.train_feasibility_classifier(successful, failed)
        self.assertNotIn("nBl", operating_clf.impeller_feature_names_)
        self.assertNotIn("P_out", operating_clf.impeller_feature_names_)
        self.assertEqual(
            operating_clf.n_features_in_,
            len(legacy_al.VAR_NAMES) - 2,
        )

        geometry_x, geometry_y = legacy_al.build_geometry_classifier_dataset(
            successful,
            failed[:, legacy_al.GEOMETRY_VAR_IDX],
        )
        geometry_clf = legacy_al.train_geometry_feasibility_classifier(
            geometry_x,
            geometry_y,
        )
        self.assertNotIn("nBl", geometry_clf.impeller_feature_names_)
        self.assertEqual(
            geometry_clf.n_features_in_,
            len(legacy_al.GEOMETRY_VAR_NAMES) - 1,
        )

        probes = np.tile(midpoint, (2, 1))
        probes[:, nbl_idx] = [9, 12]
        p_operating = legacy_al.predict_feasible_prob(operating_clf, probes)
        p_geometry = legacy_al.predict_geometry_safe_prob(geometry_clf, probes)
        self.assertAlmostEqual(float(p_operating[0]), float(p_operating[1]), places=12)
        self.assertAlmostEqual(float(p_geometry[0]), float(p_geometry[1]), places=12)

    def test_exploration_bypasses_surrogate_mask_and_counts_failed_attempts(self):
        midpoint = (legacy_al.L_BOUNDS + legacy_al.U_BOUNDS) / 2.0
        nbl_idx = legacy_al.VAR_NAMES.index("nBl")
        vary_idx = legacy_al.VAR_NAMES.index("d1s")
        candidates = np.tile(midpoint, (6, 1))
        candidates[:, vary_idx] = np.linspace(
            legacy_al.L_BOUNDS[vary_idx],
            legacy_al.U_BOUNDS[vary_idx],
            len(candidates),
        )
        candidates[:, nbl_idx] = [10, 11, 12, 9, 10, 11]
        candidates = legacy_al.pin_optimization_operating_point(candidates)

        pool = np.tile(midpoint, (13, 1))
        pool[:, vary_idx] = np.linspace(
            legacy_al.L_BOUNDS[vary_idx],
            legacy_al.U_BOUNDS[vary_idx],
            len(pool),
        )
        pool[:, nbl_idx] = [9, 9] + [10] * 5 + [11] * 5 + [12]
        pool = legacy_al.pin_optimization_operating_point(pool)

        failed = np.tile(midpoint, (3, 1))
        failed[:, vary_idx] = np.linspace(
            midpoint[vary_idx] - 0.01,
            midpoint[vary_idx] + 0.01,
            len(failed),
        )
        failed[:, nbl_idx] = 12
        failed = legacy_al.pin_optimization_operating_point(failed)

        scaler = MinMaxScaler().fit(
            np.vstack([legacy_al.L_BOUNDS, legacy_al.U_BOUNDS])
        )
        uncertainty = np.zeros((len(candidates), 3))
        uncertainty[2] = 1.0
        selected, labels = legacy_al.select_candidates_diverse(
            acq_X=candidates,
            ehvi_vals=np.array([0.8, 0.7, -np.inf, -np.inf, -np.inf, -np.inf]),
            scaler_X=scaler,
            acq_info={
                "valid_mask": np.array([True, True, False, False, False, False]),
                "exploration_mask": np.ones(len(candidates), dtype=bool),
                "pred_std_norm": uncertainty,
            },
            X_pool_raw=pool,
            X_failed_raw=failed,
            n_pick=4,
            min_dist_norm=0.01,
            n_ehvi=2,
        )

        self.assertEqual(int(round(selected[2][nbl_idx])), 12)
        self.assertTrue(labels[2].startswith("最大不确定性"))
        self.assertEqual(int(round(selected[3][nbl_idx])), 9)
        self.assertIn("failed=0", labels[3])

    def test_legacy_failure_classifier_is_marginalized_over_nbl(self):
        midpoint = (legacy_al.L_BOUNDS + legacy_al.U_BOUNDS) / 2.0
        nbl_idx = legacy_al.VAR_NAMES.index("nBl")
        legacy_training = np.tile(midpoint, (40, 1))
        legacy_training[:, nbl_idx] = np.resize([9, 10, 11, 12], 40)
        labels = (legacy_training[:, nbl_idx] <= 10).astype(int)
        legacy_clf = RandomForestClassifier(
            n_estimators=40,
            random_state=42,
        ).fit(legacy_training, labels)

        probes = np.tile(midpoint, (2, 1))
        probes[:, nbl_idx] = [9, 12]
        probabilities = legacy_al.predict_feasible_prob(legacy_clf, probes)
        self.assertAlmostEqual(
            float(probabilities[0]),
            float(probabilities[1]),
            places=12,
        )

    def test_verified_cfd_front_is_not_vetoed_by_failure_classifier(self):
        import pareto_front_query as pareto_query

        class AlwaysBadGeometryClassifier:
            n_features_in_ = len(pareto_query.GEOM_WARN_FEATURE_NAMES)
            classes_ = np.array([0, 1])

            def predict_proba(self, features):
                return np.column_stack(
                    [np.zeros(len(features)), np.ones(len(features))]
                )

        midpoint = (pareto_query.L_BOUNDS + pareto_query.U_BOUNDS) / 2.0
        midpoint[pareto_query.P_OUT_IDX] = pareto_query.OPTIMIZATION_P_OUT
        row = dict(zip(pareto_query.VAR_NAMES, midpoint))
        row.update(
            Efficiency=0.72,
            totalpressureratio=2.1,
            Power=110.0,
            MassFlow=4.0,
            is_boundary=0.0,
        )
        front = pareto_query.build_front_dataframe(
            pd.DataFrame([row]),
            geom_warn_clf=AlwaysBadGeometryClassifier(),
            geom_safe_threshold=0.99,
        )
        self.assertEqual(len(front), 1)
        self.assertAlmostEqual(float(front.iloc[0]["pred_geom_safe_prob"]), 0.0)

    def test_verified_true_hv_is_not_vetoed_by_soft_overlap_proxy(self):
        point = (legacy_al.L_BOUNDS + legacy_al.U_BOUNDS) / 2.0
        point[legacy_al.P_OUT_IDX] = legacy_al.OPTIMIZATION_P_OUT
        point[legacy_al.VAR_NAMES.index("nBl")] = 9
        point[legacy_al.VAR_NAMES.index("rake_te_s")] = (
            legacy_al.L_BOUNDS[legacy_al.VAR_NAMES.index("rake_te_s")]
        )
        self.assertGreater(
            float(legacy_al.overlap_proxy_violation(point[None, :])[0]),
            0.0,
        )
        outputs = np.array([[0.72, 2.1, 110.0, 4.0]])
        hv, front_y, front_x = legacy_al.compute_true_cumulative_hv(
            point[None, :],
            outputs,
            np.array([0.0]),
        )
        self.assertTrue(np.isfinite(hv))
        self.assertEqual(len(front_y), 1)
        self.assertEqual(len(front_x), 1)

    def test_density_nbl_weights_upweight_sparse_underrepresented_class(self):
        dense = np.zeros((8, len(legacy_al.VAR_NAMES)))
        dense[:, 0] = np.linspace(0.0, 0.02, len(dense))
        sparse = np.zeros((2, len(legacy_al.VAR_NAMES)))
        sparse[:, 0] = [0.7, 1.0]
        X_norm = np.vstack([dense, sparse])
        nbl = np.array([9] * len(dense) + [12] * len(sparse))

        weights, diagnostics = legacy_al.compute_density_nbl_sample_weights(
            X_norm,
            nbl,
            k_neighbors=3,
        )

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=7)
        self.assertGreater(float(weights[nbl == 12].mean()), float(weights[nbl == 9].mean()))
        self.assertGreater(diagnostics["effective_sample_size"], 0.0)

    def test_regression_metrics_include_rmse_mae_and_r2(self):
        true = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]])
        pred = np.array([[1.0, 1.0, 3.0], [1.0, 3.0, 3.0]])
        metrics = legacy_al.regression_metrics(true, pred, prefix="test_")

        self.assertAlmostEqual(metrics["test_rmse_eff"], 1.0)
        self.assertAlmostEqual(metrics["test_mae_eff"], 1.0)
        self.assertAlmostEqual(metrics["test_r2_eff"], 0.0)
        self.assertIn("test_rmse_macro", metrics)

    def test_validation_csv_upsert_replaces_same_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "metrics.csv"
            legacy_al.upsert_csv_records(
                str(csv_path),
                {"iter": 1, "rmse_eff": 0.2},
                key_columns=["iter"],
            )
            legacy_al.upsert_csv_records(
                str(csv_path),
                {"iter": 1, "rmse_eff": 0.1},
                key_columns=["iter"],
            )
            stored = pd.read_csv(csv_path)
            self.assertEqual(len(stored), 1)
            self.assertAlmostEqual(float(stored.loc[0, "rmse_eff"]), 0.1)
            self.assertTrue(Path(f"{csv_path}.meta.json").exists())

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

                meta_path.write_text(
                    json.dumps(
                        {
                            "completed_iters": 0,
                            "in_progress_iter": 5,
                            "performance_data_schema_version": PERFORMANCE_DATA_SCHEMA_VERSION,
                            "total_pressure_definition": TOTAL_PRESSURE_DEFINITION,
                        }
                    ),
                    encoding="utf-8",
                )
                pd.DataFrame(columns=legacy_al.VAR_NAMES + legacy_al.ALL_OUTPUT_NAMES + ["is_boundary"]).to_csv(pool_path, index=False)
                write_performance_data_metadata(pool_path)

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

    def test_run_nsga2_only_returns_artifacts(self):
        class FakeLegacy:
            def run_nsga2_only_from_lhs(self, output_csv=None, summary_json=None, use_pool_checkpoint=False):
                Path(output_csv).write_text("front_index,pred_Efficiency\n1,0.72\n", encoding="utf-8")
                Path(summary_json).write_text(json.dumps({"front_size": 1}), encoding="utf-8")
                return {
                    "train_samples": 20,
                    "test_samples": 4,
                    "front_size": 1,
                    "surrogate_hv": 0.12,
                    "mse_eff": 0.001,
                    "mse_pr": 0.002,
                    "mse_mf": 0.003,
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root)
            service = ActiveLearningService(config)
            service._legacy = FakeLegacy()
            result = service.run_nsga2_only()
            self.assertEqual(result.status, "succeeded")
            self.assertEqual(result.metrics["front_size"], 1)
            self.assertTrue((root / "nsga2_surrogate_pareto.csv").exists())
            self.assertTrue((root / "nsga2_surrogate_summary.json").exists())

    def test_compute_pareto_front_without_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df = pd.DataFrame(
                [
                    [0.35, 0.05, 72.0, 24.0, 0.47, 0.05, 44.0, 46.0, 0.2, 0.002, 0.001, 10, -18.0, 12.0, 0.70, 2.0, 120.0, 4.0, 0],
                    [0.36, 0.05, 73.0, 24.5, 0.48, 0.05, 44.5, 46.5, 0.2, 0.002, 0.001, 10, -18.5, 12.0, 0.74, 2.1, 118.0, 4.1, 0],
                    [0.37, 0.05, 74.0, 25.0, 0.49, 0.05, 45.0, 47.0, 0.2, 0.002, 0.001, 10, -19.0, 12.0, 0.68, 1.9, 121.0, 4.0, 0],
                ],
                columns=[
                    "d1s", "dH", "beta1hb", "beta1sb", "d2", "b2", "beta2hb", "beta2sb", "Lz", "t", "TipClear",
                    "nBl", "rake_te_s", "P_out", "Efficiency", "totalpressureratio", "Power", "MassFlow", "is_boundary",
                ],
            )
            training = root / "Compressor_Training_Data.csv"
            df.to_csv(training, index=False)
            write_performance_data_metadata(training)
            config = self.make_config(root)
            result = ParetoService(config).compute_pareto_front()
            self.assertEqual(result.status, "succeeded")
            self.assertTrue((root / "pareto_front_points.csv").exists())

    @mock.patch("impeller_app.runner.external.run_cfx_pipeline")
    @mock.patch.object(RunnerAPI, "run_geometry_generation")
    def test_run_doe_sample_uses_mocked_subprocess_and_cfx(self, mock_geometry, mock_cfx):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root)
            mock_geometry.return_value = TaskResult(status="succeeded", message="geometry ok")
            mock_cfx.return_value = (True, {"Efficiency": 0.7, "PressureRatio": 1.8, "Power": 100.0, "MassFlow": 4.0, "totalpressureratio": 2.0}, "Success")
            sample = RunnerAPI(config).generate_lhs_samples(1)[0]
            result = RunnerAPI(config).run_doe_sample(0, sample)
            self.assertEqual(result.status, "succeeded")
            mock_geometry.assert_called_once()
            mock_cfx.assert_called_once()
            stored = pd.read_csv(root / "Compressor_Training_Data.csv")
            self.assertEqual(len(stored), 1)
            self.assertAlmostEqual(float(stored.iloc[0]["Efficiency"]), 0.7)

    @mock.patch.object(RunnerAPI, "run_geometry_generation")
    def test_repeated_doe_geometry_failure_becomes_classifier_training_data(self, mock_geometry):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root)
            mock_geometry.return_value = TaskResult(
                status="failed",
                message="geometry generation failed with exit code 1.",
            )
            runner = RunnerAPI(config)
            sample = runner.generate_lhs_samples(1)[0]
            run_dir = config.workspace.doe_runs_dir / "Run_000"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "failure_status.json").write_text(
                json.dumps({"stage": "geometry", "reason": "confirmed geometry failure"}),
                encoding="utf-8",
            )

            runner.run_doe_sample(0, sample, force=True)
            (run_dir / "failure_status.json").write_text(
                json.dumps({"stage": "geometry", "reason": "confirmed geometry failure"}),
                encoding="utf-8",
            )
            runner.run_doe_sample(0, sample, force=True)

            geometry, operating, records = load_failure_training_sets(
                config.workspace.failure_records_csv,
                runner.variable_names,
                legacy_al.GEOMETRY_VAR_NAMES,
            )
            self.assertEqual(geometry.shape, (1, 13))
            self.assertEqual(operating.shape, (0, 14))
            self.assertTrue(bool(records.iloc[0]["confirmed"]))

    @mock.patch.object(RunnerAPI, "run_geometry_generation")
    def test_export_cases_generates_mesh_files(self, mock_geometry):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_config(root)
            mock_geometry.return_value = TaskResult(status="succeeded", message="geometry ok")

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
            mock_geometry.assert_called_once()
            case_dir = root / "pareto_cft_cases" / "ParetoCase_01_F03"
            self.assertTrue((case_dir / "geometry_parameters.csv").exists())
            report = json.loads((root / "pareto_cft_cases" / "export_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["count"], 1)
            self.assertEqual(report["failed_count"], 0)
            self.assertTrue(report["cases"][0]["mesh_generated"])


if __name__ == "__main__":
    unittest.main()
