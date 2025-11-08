# Setup script for 3D-Cam project on Windows (PowerShell)
# Creates a virtual environment and installs dependencies

Write-Host "Setting up 3D-Cam virtual environment..." -ForegroundColor Cyan

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green

    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $pyMajor = [int]$matches[1]
        $pyMinor = [int]$matches[2]
        if ($pyMajor -gt 3 -or ($pyMajor -eq 3 -and $pyMinor -ge 13)) {
            Write-Host "Warning: Python $pyMajor.$pyMinor detected. Install Python 3.11 (recommended) or 3.12 to avoid building scikit-image from source." -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "Error: Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.7+ from https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create virtual environment." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Yellow
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip | Out-Null

# Install dependencies
Write-Host "Installing dependencies from requirements.windows.txt..." -ForegroundColor Cyan
pip install -r requirements.windows.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install dependencies." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment in the future, run:" -ForegroundColor Cyan
Write-Host "  venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Reminder: Install Python 3.11 (recommended) or 3.12 so scikit-image uses prebuilt wheels." -ForegroundColor Yellow
Write-Host ""
Write-Host "Or use the run scripts:" -ForegroundColor Cyan
Write-Host "  .\visualize_3d.ps1" -ForegroundColor Yellow
Write-Host "  .\detect_pairs.ps1" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to exit"

