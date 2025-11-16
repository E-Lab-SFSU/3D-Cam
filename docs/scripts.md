---
layout: default
title: Scripts Reference
permalink: /scripts/
---

# Scripts Reference

This document lists all available setup and run scripts for the 3D-Cam project.

## Setup Scripts

These scripts create a virtual environment and install all dependencies:

### Windows
- `setup_venv.bat` - Batch file setup script (double-click or run from command prompt)
- `setup_venv.ps1` - PowerShell setup script

### Linux/Raspberry Pi
- `setup_venv.sh` - Bash setup script

## Run Scripts

These scripts automatically set up the virtual environment (if needed) and run the programs:

### Windows Batch Files (.bat)
- `visualize_3d.bat` - Run 3D visualization tool
- `detect_pairs.bat` - Run pair detection and tracking
- `capture_windows.bat` - Run video capture (Windows)
- `smooth_tracks.bat` - Run track smoothing tool
- `calibrate_scale_windows.bat` - Run image calibration (Windows version)
- `calibrate_video.bat` - Run video calibration
- `plot_z_histogram.bat` - Run Z height histogram

### Windows PowerShell (.ps1)
- `visualize_3d.ps1`
- `detect_pairs.ps1`
- `capture_windows.ps1`
- `smooth_tracks.ps1`
- `calibrate_scale_windows.ps1` - Run image calibration (Windows version, PowerShell)
- `calibrate_video.ps1`
- `plot_z_histogram.ps1`

### Linux/Raspberry Pi (.sh)
- `visualize_3d.sh` - Run 3D visualization tool
- `detect_pairs.sh` - Run pair detection and tracking
- `capture_raspi.sh` - Run video capture (Raspberry Pi)
- `smooth_tracks.sh` - Run track smoothing tool
- `calibrate_scale_raspi.sh` - Run image calibration (Raspberry Pi version)
- `calibrate_video.sh` - Run video calibration
- `plot_z_histogram.sh` - Run Z height histogram

## Usage

### First Time Setup (Linux/Raspberry Pi)

```bash
# Make all scripts executable (one-time setup)
chmod +x *.sh

# Run any program (note the ./ before the script name)
./visualize_3d.sh
```

**Why `./` and `chmod +x`?**
- `./` is required - Linux needs this to run scripts in the current directory
- `chmod +x` makes files executable - only needed once per file
- See [Linux Basics]({{ site.baseurl }}/linux-basics) for detailed explanation

### First Time Setup (Windows)

Just double-click any `*.bat` file, or run from command prompt:
```batch
visualize_3d.bat
```

Or use PowerShell:
```powershell
.\visualize_3d.ps1
```

## What the Run Scripts Do

1. Check if virtual environment (`venv/`) exists
2. If not, run the setup script to create it and install dependencies
3. Activate the virtual environment
4. Run the Python program

## Notes

- The run scripts are designed to be user-friendly - just double-click or run them
- They handle all setup automatically
- You can still use the setup scripts manually if you prefer
- The virtual environment is created once and reused for all programs

