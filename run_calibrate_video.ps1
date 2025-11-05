# Run script for calibrate_video.py on Windows (PowerShell)

if (-not (Test-Path "venv")) {
    & ".\setup_venv.ps1"
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

& "venv\Scripts\Activate.ps1"
$env:PYTHONPATH = $PWD
python apps/calibrate_video.py
Read-Host "Press Enter to exit"

