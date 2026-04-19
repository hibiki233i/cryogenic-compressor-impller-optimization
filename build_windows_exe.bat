@echo off
setlocal

cd /d "%~dp0"

echo [1/4] Installing build dependencies...
py -m pip install -r requirements-gui.txt
py -m pip install -r requirements-build.txt
if errorlevel 1 goto :error

echo [2/4] Cleaning previous build output...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [3/4] Building ImpellerWorkbench.exe with PyInstaller...
py -m PyInstaller --noconfirm ImpellerWorkbench.spec
if errorlevel 1 goto :error

echo [4/4] Build complete.
echo Output folder: %cd%\dist\ImpellerWorkbench
echo Executable   : %cd%\dist\ImpellerWorkbench\ImpellerWorkbench.exe
goto :eof

:error
echo Build failed.
exit /b 1
