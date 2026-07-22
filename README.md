# BOUNDYR

A Windows-oriented impeller optimization workbench built around:

- active learning
- neural-network surrogate prediction
- NSGA-II multi-objective optimization
- Pareto-front querying and case export
- a PySide6 desktop GUI for workflow orchestration

## What Is Included

- `impeller_app/`
  Desktop application package with config, runner, core services, and GUI.
- `NN_NSGA2_ActiveLearning_refactored.py`
  Legacy active-learning and optimization engine.
- `DOE.py`
  Legacy DOE workflow.
- `cfx_runner.py`
  CFX execution pipeline wrapper.
- `pareto_front_query.py`
  Pareto-front extraction and inverse query utilities.
- `pareto_export_cft_cases.py`
  Pareto case export utilities.

## GUI

Install dependencies:

```bash
python3 -m pip install -r requirements-gui.txt
```

Launch the desktop app:

```bash
python3 -m impeller_app
```

On Windows, you can also use:

```bat
launch_gui.bat
```

## Build Windows `.exe`

```bat
build_windows_exe.bat
```

The packaged executable will be created under:

```text
dist\ImpellerWorkbench\ImpellerWorkbench.exe
```

## External Dependencies

The GUI and packaged app do not bundle ANSYS CFX, CFturbo, or PowerShell. The target Windows machine still needs:

- PowerShell 7
- ANSYS CFX
- the geometry script `Run-GeometryMeshing.ps1`
- valid template files and accessible working directories

Configure these paths in the GUI `Environment` tab before running DOE or active-learning tasks.
The desktop app now auto-saves path and runtime edits and reloads them on the next launch.
On Windows, the default config file is `%APPDATA%\\BOUNDYR\\impeller-app-config.json`.
Default engineering parameters, including the fixed NSGA-II operating point, flow/efficiency
thresholds, and geometry screening thresholds, can be edited in the GUI or JSON config file.

## Optimization Problem Definition

- The DOE may vary both geometry and `P_out` to train a conditional surrogate across operating
  conditions.
- `P_out` is an operating-condition input, not a geometry design variable.
- NSGA-II optimizes the 13 geometry variables at a fixed outlet static pressure. The default
  design point is `P_out = 12 Pa`.
- Total pressure ratio is the mass-flow-averaged stationary-frame total-to-total pressure ratio.
- Total pressure ratio is maximized without a minimum or maximum pressure-ratio constraint.
- Observed-data Pareto extraction only uses samples inside the configured pressure band around
  the fixed design point; it does not mix the full operating map into a geometry Pareto front.

The stationary-frame pressure-ratio definition is performance-data schema version 2. Legacy
`CFX_Results.txt`, training CSV, fixed-test, and active-learning checkpoint data are rejected
unless they carry matching metadata. Use DOE recovery to re-run CFX-Post from existing `.res`
files, then retrain the surrogate and regenerate the Pareto front.

## Surrogate Validation Records

Each active-learning round writes review-ready validation data into the configured workspace:

- `surrogate_validation_history.csv`: fixed-test RMSE, MAE, R² and K-fold mean/std metrics.
- `surrogate_cv_fold_history.csv`: fold-level RMSE, MAE and R² with fold-local scalers.
- `fixed_test_predictions_history.csv`: fixed-test CFD truth and surrogate prediction pairs.
- `al_query_validation.csv`: pre-CFD prediction, MC-Dropout uncertainty, CFD truth, residuals,
  and one-/two-sigma coverage for every queried point.

Surrogate training uses tempered inverse-density and inverse-frequency `nBl` weights, in addition
to the existing boundary-sample weight, so dense local active-learning clusters do not dominate
the loss.

## Failure-Aware Active Learning

DOE and active-learning run outcomes are persisted in `failure_records.csv`. A failure caused by
software, setup, licensing, CFX-Pre, or CFX-Post is retained for audit but is not treated as a
physically infeasible design. Deterministic blockage, invalid-flow, and explicitly detected
fatal-overflow outcomes are confirmed immediately; retry-based confirmation remains available for
repeatable geometry or mesh-generation failures.

- Confirmed CFturbo/TurboGrid generation failures train the geometry-safety classifier with 12
  learned inputs: fixed `P_out` and categorical blade count `nBl` are deliberately excluded.
- Confirmed fatal-overflow, blockage, and invalid-flow outcomes train the operating-feasibility
  classifier with 12 learned inputs. Fixed `P_out` and `nBl` are excluded to prevent stage leakage
  and self-reinforcing blade-count bias.
- A generic CFX solver exit or divergence without an explicit fatal-overflow signature remains an
  audit record and does not train either design-failure classifier.
- Converged near-boundary samples remain in the regression data with `is_boundary`; they are not
  mixed with hard execution failures.
- The legacy untyped `failed_points.npy` file is ignored by the new classifiers and candidate
  filter. The typed CSV is rebuilt each round so a point that later succeeds is not permanently
  blacklisted.

## Notes

- The project is currently structured to support Windows-based engineering workflows.
- Existing legacy research scripts are preserved and wrapped by the desktop app rather than fully replaced.
- DOE recovery, active-learning resume, Pareto querying, and case export are all available through the GUI layer.
- The GUI maintains surrogate-input bounds in `design_variables.json`. Geometry inputs and
  operating-condition inputs have distinct roles.
- Flow, efficiency, and geometry feasibility thresholds are configurable engineering parameters.
- Environment validation can create an empty training CSV automatically when the configured file does not yet exist.
