# Run script for track_smoother.py on Windows (PowerShell)

if (-not (Test-Path "venv")) {
    & ".\setup_venv.ps1"
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

& "venv\Scripts\Activate.ps1"
python track_smoother.py
Read-Host "Press Enter to exit"

