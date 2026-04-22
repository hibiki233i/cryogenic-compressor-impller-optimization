# Windows Desktop GUI

## Launch

Install GUI dependencies first:

```bash
python3 -m pip install -r requirements-gui.txt
```

Start the desktop app:

```bash
python3 -m impeller_app
```

Windows launcher:

```bat
launch_gui.bat
```

## Build `.exe`

Install build dependencies:

```bash
python3 -m pip install -r requirements-gui.txt
python3 -m pip install -r requirements-build.txt
```

Build the Windows executable with PyInstaller:

```bat
build_windows_exe.bat
```

Main build artifacts:

- `dist/ImpellerWorkbench/ImpellerWorkbench.exe`
- `ImpellerWorkbench.spec`

Notes:

- The `.exe` packages the GUI and Python runtime, but not ANSYS CFX, CFturbo, or PowerShell.
- The target Windows machine still needs valid local installations and project-accessible templates/scripts.
- Dynamic legacy modules are bundled through `ImpellerWorkbench.spec`, and the frozen app resolves them via `sys._MEIPASS`.

## Current Structure

- `impeller_app/config.py`
  Centralized `AppConfig` for solver paths, workspace paths, and runtime settings.
- `impeller_app/core/`
  Active-learning and Pareto-facing services that wrap the existing optimization scripts.
- `impeller_app/runner/`
  External runner API for PowerShell geometry generation, CFX execution, DOE recovery, and batch DOE execution.
- `impeller_app/gui/`
  PySide6 desktop GUI with 5 tabs:
  - Environment
  - DOE
  - Active Learning
  - Pareto
  - Export

## Current Capabilities

- Environment validation for PowerShell, geometry script, CFX executables, templates, and training CSV
- DOE run recovery by scanning `Run_*` folders and rebuilding `Compressor_Training_Data.csv`
- DOE batch execution with persistent `extra_samples.json`
- Surrogate training from the desktop app
- Active-learning resume and iterative execution
- Pareto front generation, query, and case export

## Notes

- The desktop app targets Windows workflows and local ANSYS/CFturbo installations.
- The GUI keeps the legacy Python scripts as the computation engine and layers a configurable desktop shell around them.
- `NN_NSGA2_ActiveLearning_refactored.py` and `cfx_runner.py` now accept configuration overrides so the GUI is no longer blocked on hardcoded paths.
- `windows-app-config.template.json` is included as a deployment checklist for Windows target machines.
- The GUI now persists edited paths and runtime parameters to `%APPDATA%\BOUNDYR\impeller-app-config.json` on Windows and reloads them automatically at startup.
