@echo off
REM Run script for z_histogram.py on Windows

if not exist "venv" (
    call setup_venv.bat
    if errorlevel 1 (
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat
python z_histogram.py
pause

