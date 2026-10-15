"""Reproduce the synthetic DOE -> surrogate -> NSGA-II -> Pareto query offline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from importlib.metadata import version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("offline-output"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key, minimum in (("seed", 0), ("samples", 64), ("max_epochs", 1),
                         ("patience", 1), ("pop_size", 4), ("n_gen", 1)):
        value = config.get(key)
        if type(value) is not int or value < minimum:
            parser.error(f"config.{key} must be an integer >= {minimum}")
    output = args.output_dir.resolve()
    # A fresh directory prevents old fixed tests/failure records/stages from
    # changing the experiment and protects existing artifacts from overwrite.
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    previous_cwd = Path.cwd()
    previous_sys_path = list(sys.path)
    previous_environment = dict(os.environ)
    try:
        # The CLI is a separate process; user GUI/engineering settings must not
        # alter this self-contained example or load private training artifacts.
        for key in list(os.environ):
            if key.startswith("IMPELLER_"):
                del os.environ[key]
        os.environ["MPLBACKEND"] = "Agg"
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        os.chdir(output)

        import torch
        from design_variables import DEFAULT_VARIABLE_SPECS, save_variable_specs, write_performance_data_metadata
        from examples.offline.generate import generate_training_data
        import NN_NSGA2_ActiveLearning_refactored as optimizer
        import pareto_front_query as query

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        save_variable_specs(DEFAULT_VARIABLE_SPECS, "design_variables.json")
        frame = generate_training_data(Path("Compressor_Training_Data.csv"), config["samples"], config["seed"])
        Path("al_stage.json").write_text(json.dumps({"stage_id": "synthetic_offline_v1", "seed": config["seed"]}), encoding="utf-8")
        optimizer.configure_runtime(
            DESIGN_VARIABLES_PATH="design_variables.json",
            TRAINING_CSV="Compressor_Training_Data.csv", TEST_SET_CSV="fixed_test_set.csv",
            FAILURE_RECORDS_CSV="failure_records.csv", CHECKPOINT_META_PATH="al_checkpoint_meta.json",
        )
        optimizer.CFG.pop_size = config["pop_size"]
        optimizer.CFG.n_gen = config["n_gen"]
        nsga2 = optimizer.run_nsga2_only_from_lhs(
            max_epochs=config["max_epochs"], patience=config["patience"],
        )
        query.configure_runtime(design_variables_path="design_variables.json")
        # Query analytic DOE rows through the existing observed-data code path.
        # Never rename surrogate predictions to observed performance columns.
        dataset = query.load_dataset("Compressor_Training_Data.csv")
        front = query.build_front_dataframe(dataset)
        ranked = query.compute_engineering_front_scores(front, dataset)
        front.to_csv("pareto_front_points.csv", index=False)
        ranked.to_csv("pareto_engineering_ranked.csv", index=False)
        write_performance_data_metadata("pareto_front_points.csv")
        write_performance_data_metadata("pareto_engineering_ranked.csv")
        selection = query.exact_front_selection(ranked, int(ranked.iloc[0]["front_index"]))
        query.save_selection_json(selection, "pareto_selected_point.json")
        summary = {
            "data_source": "synthetic_analytic_v1",
            "config": config,
            "environment": {"python": platform.python_version(), **{
                name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "pymoo")
            }},
            "samples": len(frame), "analytic_front_size": len(front),
            "analytic_max_efficiency": float(front.Efficiency.max()),
            "analytic_max_total_pressure_ratio": float(front.totalpressureratio.max()),
            "selection_source": "synthetic_analytic_v1",
            "selection": selection, "nsga2": nsga2,
            "elapsed_seconds": time.perf_counter() - start,
        }
        Path("summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Offline reproduction complete: {output / 'summary.json'}")
        return 0
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_sys_path
        os.environ.clear()
        os.environ.update(previous_environment)


if __name__ == "__main__":
    raise SystemExit(main())
