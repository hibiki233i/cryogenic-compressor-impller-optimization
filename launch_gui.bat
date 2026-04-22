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
set "PYTHON_CMD="

if exist "%cd%\.venv\Scripts\python.exe" (
  set "PYTHON_CMD=\"%cd%\.venv\Scripts\python.exe\""
  goto :eof
)

if exist "%cd%\venv\Scripts\python.exe" (
  set "PYTHON_CMD=\"%cd%\venv\Scripts\python.exe\""
  goto :eof
)

call :check_python_cmd py -3
if defined PYTHON_CMD goto :eof

call :check_python_cmd py
if defined PYTHON_CMD goto :eof

call :check_python_cmd python
if defined PYTHON_CMD goto :eof

call :check_python_cmd python3
if defined PYTHON_CMD goto :eof

call :check_app_path "HKCU\Software\Microsoft\Windows\CurrentVersion\App Paths\python.exe"
if defined PYTHON_CMD goto :eof

call :check_app_path "HKLM\Software\Microsoft\Windows\CurrentVersion\App Paths\python.exe"
if defined PYTHON_CMD goto :eof

for /d %%D in ("%LocalAppData%\Programs\Python\Python*") do (
  if exist "%%~fD\python.exe" (
    set "PYTHON_CMD=\"%%~fD\python.exe\""
    goto :eof
  )
)

for /d %%D in ("%ProgramFiles%\Python*") do (
  if exist "%%~fD\python.exe" (
    set "PYTHON_CMD=\"%%~fD\python.exe\""
    goto :eof
  )
)

for /d %%D in ("%ProgramFiles(x86)%\Python*") do (
  if exist "%%~fD\python.exe" (
    set "PYTHON_CMD=\"%%~fD\python.exe\""
    goto :eof
  )
)

exit /b 1

:check_python_cmd
%* --version >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_CMD=%*"
)
exit /b 0

:check_app_path
for /f "skip=2 tokens=2,*" %%A in ('reg query %1 /ve 2^>nul') do (
  if /i "%%A"=="REG_SZ" (
    if exist "%%B" (
      set "PYTHON_CMD=\"%%B\""
      goto :eof
    )
  )
)
exit /b 0

:no_python
echo No Python launcher was found.
echo Install Python or add it to PATH. The launcher checked: project virtualenv, py -3, py, python, python3, registry app paths, and common install directories.
exit /b 1
