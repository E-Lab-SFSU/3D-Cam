# Run script for plot_z_histogram.py on Windows (PowerShell)

if (-not (Test-Path "venv")) {
    & ".\setup_venv.ps1"
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

& "venv\Scripts\Activate.ps1"
$env:PYTHONPATH = $PWD
python apps/plot_z_histogram.py
Read-Host "Press Enter to exit"

