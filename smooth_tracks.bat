@echo off
REM Run script for smooth_tracks.py on Windows

if not exist "venv" (
    call setup_venv.bat
    if errorlevel 1 (
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat
set PYTHONPATH=%CD%
python apps/smooth_tracks.py
pause

