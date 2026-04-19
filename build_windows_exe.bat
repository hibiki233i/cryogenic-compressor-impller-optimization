@echo off
setlocal

cd /d "%~dp0"

call :detect_python
if errorlevel 1 goto :no_python

echo [1/4] Installing build dependencies...
%PYTHON_CMD% -m pip install -r requirements-gui.txt
if errorlevel 1 goto :error
%PYTHON_CMD% -m pip install -r requirements-build.txt
if errorlevel 1 goto :error

echo [2/4] Cleaning previous build output...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [3/4] Building ImpellerWorkbench.exe with PyInstaller...
%PYTHON_CMD% -m PyInstaller --noconfirm ImpellerWorkbench.spec
if errorlevel 1 goto :error

echo [4/4] Build complete.
echo Output folder: %cd%\dist\ImpellerWorkbench
echo Executable   : %cd%\dist\ImpellerWorkbench\ImpellerWorkbench.exe
goto :eof

:detect_python
where py >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_CMD=py"
  goto :eof
)

where python >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_CMD=python"
  goto :eof
)

where python3 >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_CMD=python3"
  goto :eof
)

exit /b 1

:no_python
echo No Python launcher was found.
echo Install Python and make sure one of these commands is available in PATH: py, python, python3
exit /b 1

:error
echo Build failed.
if exist "%cd%\build\ImpellerWorkbench\warn-ImpellerWorkbench.txt" (
  echo PyInstaller warnings: %cd%\build\ImpellerWorkbench\warn-ImpellerWorkbench.txt
)
exit /b 1
