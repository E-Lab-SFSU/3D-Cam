# Run script for visualize_3d.py on Windows (PowerShell)
# Automatically sets up venv if needed and runs the program

if (-not (Test-Path "venv")) {
    Write-Host "Virtual environment not found. Running setup..." -ForegroundColor Yellow
    & ".\setup_venv.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Setup failed. Please run setup_venv.ps1 manually." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& "venv\Scripts\Activate.ps1"

Write-Host "Running visualize3d.py..." -ForegroundColor Cyan
$env:PYTHONPATH = $PWD
python apps/visualize_3d.py

Read-Host "Press Enter to exit"

