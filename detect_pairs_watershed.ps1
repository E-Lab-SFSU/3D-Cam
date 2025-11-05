# Run script for detect_pairs_watershed.py on Windows (PowerShell)

if (-not (Test-Path "venv")) {
    Write-Host "Virtual environment not found. Running setup..." -ForegroundColor Yellow
    & ".\setup_venv.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Setup failed. Please run setup_venv.ps1 manually." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

& "venv\Scripts\Activate.ps1"
$env:PYTHONPATH = $PWD
python apps/detect_pairs_watershed.py
Read-Host "Press Enter to exit"
