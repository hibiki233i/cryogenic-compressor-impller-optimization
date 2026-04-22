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

## Notes

- The project is currently structured to support Windows-based engineering workflows.
- Existing legacy research scripts are preserved and wrapped by the desktop app rather than fully replaced.
- DOE recovery, active-learning resume, Pareto querying, and case export are all available through the GUI layer.
