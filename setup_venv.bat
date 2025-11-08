@echo off
REM Setup script for 3D-Cam project on Windows
REM Creates a virtual environment and installs dependencies

setlocal EnableExtensions EnableDelayedExpansion

echo Setting up 3D-Cam virtual environment...

REM Check if Python is available
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VERSION=%%v
if not defined PY_VERSION (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.7+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo Found: Python %PY_VERSION%

for /f "tokens=1,2 delims=." %%a in ("%PY_VERSION%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if not defined PY_MINOR set PY_MINOR=0
if defined PY_MAJOR (
    if !PY_MAJOR! GTR 3 (
        echo Warning: Python !PY_MAJOR!.!PY_MINOR! detected. Install Python 3.11 or 3.12 to avoid building scikit-image from source.
    ) else (
        if "!PY_MAJOR!"=="3" (
            if !PY_MINOR! GEQ 13 (
                echo Warning: Python !PY_MAJOR!.!PY_MINOR! detected. Install Python 3.11 or 3.12 to avoid building scikit-image from source.
            )
        )
    )
)

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies from requirements.windows.txt...
pip install -r requirements.windows.txt

if errorlevel 1 (
    echo Error: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate
echo.
echo Reminder: Install Python 3.11 or 3.12 so scikit-image uses prebuilt wheels.
echo.
echo Or use the run scripts:
echo   visualize_3d.bat
echo   detect_pairs.bat
echo   etc.
echo.
pause

