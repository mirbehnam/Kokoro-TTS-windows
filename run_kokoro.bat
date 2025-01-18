@echo off
setlocal enabledelayedexpansion

:: Set up paths relative to the batch file location
set "PROJECT_ROOT=%~dp0"
set "PYTHON_DIR=%PROJECT_ROOT%runtime\python-3.10.0-embed-amd64"
set "SCRIPTS_DIR=%PYTHON_DIR%\Scripts"
set "PATH=%PYTHON_DIR%;%SCRIPTS_DIR%;%PATH%"

:: Check if Python exists in the runtime folder
if not exist "%PYTHON_DIR%\python.exe" (
    echo Error: Embedded Python not found in runtime folder.
    echo Please ensure Python 3.10.0 embedded is extracted to: %PYTHON_DIR%
    pause
    exit /b 1
)

:: Create/modify python3X._pth file to include necessary paths
echo python310.zip> "%PYTHON_DIR%\python310._pth"
echo .>> "%PYTHON_DIR%\python310._pth"
echo Lib\site-packages>> "%PYTHON_DIR%\python310._pth"
echo ..>> "%PYTHON_DIR%\python310._pth"

:: Install pip if not already installed
if not exist "%SCRIPTS_DIR%\pip.exe" (
    echo Installing pip...
    "%PYTHON_DIR%\python.exe" -c "import ensurepip; ensurepip.bootstrap()"
)

:: Check if requirements are already installed
echo Checking/Installing requirements...
"%PYTHON_DIR%\python.exe" -m pip install --no-warn-script-location -r "%PROJECT_ROOT%requirements.txt"

:: Run the launcher script
echo Starting KokoroTTS...
cd "%PROJECT_ROOT%"
"%PYTHON_DIR%\python.exe" "%PROJECT_ROOT%launcher.py"

pause