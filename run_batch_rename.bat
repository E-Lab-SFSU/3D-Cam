@echo off
REM Run script for batch_rename.py on Windows (Batch)

if not exist "venv" (
    echo Virtual environment not found. Running setup...
    call setup_venv.bat
    if errorlevel 1 (
        echo Setup failed. Please run setup_venv.bat manually.
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat
python batch_rename.py %*
pause

