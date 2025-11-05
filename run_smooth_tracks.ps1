# Run script for smooth_tracks.py on Windows (PowerShell)

if (-not (Test-Path "venv")) {
    & ".\setup_venv.ps1"
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

& "venv\Scripts\Activate.ps1"
$env:PYTHONPATH = $PWD
python apps/smooth_tracks.py
Read-Host "Press Enter to exit"

