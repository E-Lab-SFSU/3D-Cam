@echo off
REM Run script for track_smoother.py on Windows

if not exist "venv" (
    call setup_venv.bat
    if errorlevel 1 (
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat
python track_smoother.py
pause

