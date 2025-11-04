@echo off
REM Run script for visualize3d.py on Windows
REM Automatically sets up venv if needed and runs the program

if not exist "venv" (
    echo Virtual environment not found. Running setup...
    call setup_venv.bat
    if errorlevel 1 (
        echo Setup failed. Please run setup_venv.bat manually.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Running visualize3d.py...
set PYTHONPATH=%CD%
python apps/visualize3d.py

pause

