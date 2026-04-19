from __future__ import annotations

import json
import os
from pathlib import Path

from ..config import AppConfig
from ..legacy import export_module, pareto_module
from ..models import TaskResult


class ParetoService:
    def __init__(self, config: AppConfig):
        self.config = config.resolved()
        os.environ["IMPELLER_DESIGN_VARIABLES_PATH"] = str(self.config.workspace.design_variables_json)
        self.pareto = pareto_module()
        if hasattr(self.pareto, "configure_runtime"):
            self.pareto.configure_runtime(str(self.config.workspace.design_variables_json))
        self.exporter = export_module()

    def compute_pareto_front(self) -> TaskResult:
        cfg = self.config
        df = self.pareto.load_dataset(str(cfg.workspace.pool_checkpoint_csv if cfg.workspace.pool_checkpoint_csv.exists() else cfg.workspace.training_csv))
        geom_warn_clf = self.pareto.load_geometry_warning_classifier(str(cfg.workspace.geom_warn_clf_pkl))
        front = self.pareto.build_front_dataframe(df, geom_warn_clf=geom_warn_clf, geom_safe_threshold=cfg.runtime.pareto_geom_safe_threshold)
        if len(front) == 0:
            return TaskResult(status="failed", message="No feasible Pareto front could be extracted.")
        ranked = self.pareto.compute_engineering_front_scores(front, df, geom_warn_clf=geom_warn_clf)
        front.to_csv(cfg.workspace.pareto_front_csv, index=False)
        ranked.to_csv(cfg.workspace.pareto_engineering_csv, index=False)
        self.pareto.save_pareto_plot(front, str(cfg.workspace.pareto_plot_png))
        report = {
            "front_size": int(len(front)),
            "recommended_front_index": int(ranked.iloc[0]["front_index"]),
            "recommended_engineering_rank": int(ranked.iloc[0]["engineering_rank"]),
        }
        cfg.workspace.pareto_engineering_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return TaskResult(
            status="succeeded",
            message=f"Computed Pareto front with {len(front)} points.",
            metrics=report,
            artifacts={
                "front_csv": str(cfg.workspace.pareto_front_csv),
                "engineering_csv": str(cfg.workspace.pareto_engineering_csv),
                "plot": str(cfg.workspace.pareto_plot_png),
                "report": str(cfg.workspace.pareto_engineering_json),
            },
        )

    def query_front(self, selection: dict) -> TaskResult:
        cfg = self.config
        result = self.compute_pareto_front()
        if result.status != "succeeded":
            return result
        df = self.pareto.load_dataset(str(cfg.workspace.pool_checkpoint_csv if cfg.workspace.pool_checkpoint_csv.exists() else cfg.workspace.training_csv))
        front = self.pareto.load_dataset(str(cfg.workspace.pareto_front_csv))
        if selection.get("front_index") is not None:
            ranked = self.pareto.pd.read_csv(cfg.workspace.pareto_engineering_csv)
            payload = self.pareto.exact_front_selection(ranked, int(selection["front_index"]))
        elif selection.get("target_eff") is not None and selection.get("target_pr") is not None:
            model = self.pareto.load_surrogate_model(str(cfg.workspace.best_regressor_pth))
            scaler_x, scaler_y = self.pareto.make_scalers(df)
            payload = self.pareto.inverse_design_search(
                model=model,
                scaler_x=scaler_x,
                scaler_y=scaler_y,
                target_eff=float(selection["target_eff"]),
                target_pr=float(selection["target_pr"]),
                front=front,
                df_all=df,
            )
        else:
            frac = float(selection.get("curve_frac", 0.5))
            payload = self.pareto.polyline_fraction_point(front, frac)
            payload["selection_mode"] = "curve_fraction"
            payload["curve_fraction"] = frac
        cfg.workspace.pareto_selection_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return TaskResult(
            status="succeeded",
            message="Pareto query completed.",
            metrics=payload,
            artifacts={"selection_json": str(cfg.workspace.pareto_selection_json)},
        )

    def export_cases(self, top_n: int = 1, force: bool = False, base_cft: str | None = None, cft_batch_template: str | None = None) -> TaskResult:
        cfg = self.config
        resolved_base_cft = base_cft or str(cfg.solver.base_cft)
        resolved_batch_template = cft_batch_template or str(cfg.solver.cft_batch_template)
        engineering_df = self.exporter.load_csv(str(cfg.workspace.pareto_engineering_csv))
        front_df = self.exporter.load_csv(str(cfg.workspace.pareto_front_csv))
        args = type(
            "Args",
            (),
            {
                "top_n": top_n,
                "front_indices": None,
                "curve_fractions": None,
                "case_prefix": "ParetoCase",
                "base_cft": resolved_base_cft,
                "cft_batch_template": resolved_batch_template,
                "force": force,
            },
        )()
        rows = self.exporter.select_rows(args, engineering_df, front_df)
        cfg.workspace.pareto_export_dir.mkdir(parents=True, exist_ok=True)
        exported = []
        for i, row in enumerate(rows, start=1):
            case_dir = cfg.workspace.pareto_export_dir / f"ParetoCase_{i:02d}_F{int(row['front_index']):02d}"
            if case_dir.exists() and force:
                self.exporter.shutil.rmtree(case_dir)
            summary = self.exporter.write_case_files(case_dir, row, resolved_base_cft, resolved_batch_template)
            exported.append(summary)
        report_path = cfg.workspace.pareto_export_dir / "export_report.json"
        report_path.write_text(json.dumps({"count": len(exported), "cases": exported}, ensure_ascii=False, indent=2), encoding="utf-8")
        return TaskResult(
            status="succeeded",
            message=f"Exported {len(exported)} Pareto case folders.",
            metrics={"case_count": len(exported)},
            artifacts={"export_dir": str(cfg.workspace.pareto_export_dir), "report": str(report_path)},
        )
