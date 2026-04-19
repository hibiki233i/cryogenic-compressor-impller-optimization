@echo off
setlocal

cd /d "%~dp0"

if exist "%cd%\dist\ImpellerWorkbench\ImpellerWorkbench.exe" (
  start "" "%cd%\dist\ImpellerWorkbench\ImpellerWorkbench.exe"
  goto :eof
)

echo Built executable not found, falling back to Python launcher...
call :detect_python
if errorlevel 1 goto :no_python

%PYTHON_CMD% -m pip install -r requirements-gui.txt
if errorlevel 1 exit /b 1
%PYTHON_CMD% -m impeller_app
exit /b %errorlevel%

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
