from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SolverPaths:
    powershell_exe: Path = Path(r"C:\Program Files\PowerShell\7\pwsh.exe")
    geometry_script_path: Path = Path("Run-GeometryMeshing.ps1")
    cfx_bin_dir: Path = Path(r"D:\ANSYS Inc\v251\CFX\bin")
    template_cfx: Path = Path("Templates/BaseModel.cfx")
    template_cse: Path = Path("Templates/Extract_Results.cse")
    base_cft: Path = Path("Templates/0908-2.cft")
    cft_batch_template: Path = Path("Templates/BaseModel.cft-batch")


@dataclass
class WorkspacePaths:
    project_root: Path = Path.cwd()
    doe_runs_dir: Path = Path("Runs")
    active_learning_runs_dir: Path = Path("ActiveLearning_Runs")
    training_csv: Path = Path("Compressor_Training_Data.csv")
    design_variables_json: Path = Path("design_variables.json")
    extra_samples_json: Path = Path("extra_samples.json")
    pool_checkpoint_csv: Path = Path("al_training_pool_checkpoint.csv")
    checkpoint_meta_json: Path = Path("al_checkpoint_meta.json")
    failed_points_npy: Path = Path("failed_points.npy")
    hv_history_csv: Path = Path("hv_history.csv")
    hv_plot_png: Path = Path("hv_convergence.png")
    scaler_x_pkl: Path = Path("scaler_X.pkl")
    scaler_y_pkl: Path = Path("scaler_Y.pkl")
    best_regressor_pth: Path = Path("best_regressor.pth")
    geom_warn_clf_pkl: Path = Path("geometry_warning_clf.pkl")
    pareto_front_csv: Path = Path("pareto_front_points.csv")
    pareto_plot_png: Path = Path("pareto_front.png")
    pareto_selection_json: Path = Path("pareto_selected_point.json")
    pareto_engineering_csv: Path = Path("pareto_engineering_ranked.csv")
    pareto_engineering_json: Path = Path("pareto_engineering_report.json")
    pareto_export_dir: Path = Path("pareto_cft_cases")

    def resolve_all(self) -> "WorkspacePaths":
        root = self.project_root.resolve()
        return WorkspacePaths(
            project_root=root,
            doe_runs_dir=_resolve(root, self.doe_runs_dir),
            active_learning_runs_dir=_resolve(root, self.active_learning_runs_dir),
            training_csv=_resolve(root, self.training_csv),
            design_variables_json=_resolve(root, self.design_variables_json),
            extra_samples_json=_resolve(root, self.extra_samples_json),
            pool_checkpoint_csv=_resolve(root, self.pool_checkpoint_csv),
            checkpoint_meta_json=_resolve(root, self.checkpoint_meta_json),
            failed_points_npy=_resolve(root, self.failed_points_npy),
            hv_history_csv=_resolve(root, self.hv_history_csv),
            hv_plot_png=_resolve(root, self.hv_plot_png),
            scaler_x_pkl=_resolve(root, self.scaler_x_pkl),
            scaler_y_pkl=_resolve(root, self.scaler_y_pkl),
            best_regressor_pth=_resolve(root, self.best_regressor_pth),
            geom_warn_clf_pkl=_resolve(root, self.geom_warn_clf_pkl),
            pareto_front_csv=_resolve(root, self.pareto_front_csv),
            pareto_plot_png=_resolve(root, self.pareto_plot_png),
            pareto_selection_json=_resolve(root, self.pareto_selection_json),
            pareto_engineering_csv=_resolve(root, self.pareto_engineering_csv),
            pareto_engineering_json=_resolve(root, self.pareto_engineering_json),
            pareto_export_dir=_resolve(root, self.pareto_export_dir),
        )


@dataclass
class RuntimeSettings:
    cfx_cores: int = 8
    rpm: float = 10000.0
    mass_flow: float = 0.0036
    alpha0: float = 0.0
    doe_initial_samples: int = 300
    doe_target_samples: int = 300
    active_learning_additional_iters: int = 1
    pareto_geom_safe_threshold: float = 0.45


@dataclass
class AppConfig:
    solver: SolverPaths = field(default_factory=SolverPaths)
    workspace: WorkspacePaths = field(default_factory=WorkspacePaths)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    def resolved(self) -> "AppConfig":
        return AppConfig(
            solver=SolverPaths(
                powershell_exe=self.solver.powershell_exe,
                geometry_script_path=_resolve(self.workspace.project_root, self.solver.geometry_script_path),
                cfx_bin_dir=self.solver.cfx_bin_dir,
                template_cfx=_resolve(self.workspace.project_root, self.solver.template_cfx),
                template_cse=_resolve(self.workspace.project_root, self.solver.template_cse),
                base_cft=_resolve(self.workspace.project_root, self.solver.base_cft),
                cft_batch_template=_resolve(self.workspace.project_root, self.solver.cft_batch_template),
            ),
            workspace=self.workspace.resolve_all(),
            runtime=self.runtime,
        )

    def legacy_overrides(self) -> dict[str, str]:
        cfg = self.resolved()
        ws = cfg.workspace
        return {
            "PS_SCRIPT_PATH": str(cfg.solver.geometry_script_path),
            "AL_WORKING_BASE": str(ws.active_learning_runs_dir),
            "TRAINING_CSV": str(ws.training_csv),
            "SCALER_X_PATH": str(ws.scaler_x_pkl),
            "SCALER_Y_PATH": str(ws.scaler_y_pkl),
            "BEST_REG_PATH": str(ws.best_regressor_pth),
            "GEOM_WARN_CLF_PATH": str(ws.geom_warn_clf_pkl),
            "HV_CSV_PATH": str(ws.hv_history_csv),
            "HV_PLOT_PATH": str(ws.hv_plot_png),
            "FAILED_POINTS_PATH": str(ws.failed_points_npy),
            "POOL_CHECKPOINT_CSV": str(ws.pool_checkpoint_csv),
            "CHECKPOINT_META_PATH": str(ws.checkpoint_meta_json),
            "DESIGN_VARIABLES_PATH": str(ws.design_variables_json),
        }


def _resolve(root: Path, candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else (root / candidate)
