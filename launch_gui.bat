@echo off
setlocal

cd /d "%~dp0"

if exist "%cd%\dist\ImpellerWorkbench\ImpellerWorkbench.exe" (
  start "" "%cd%\dist\ImpellerWorkbench\ImpellerWorkbench.exe"
  goto :eof
)

echo Built executable not found, falling back to Python launcher...
py -m pip install -r requirements-gui.txt
if errorlevel 1 exit /b 1
py -m impeller_app
