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
- `run_visualize3d.bat` - Run 3D visualization tool
- `run_pair_detect.bat` - Run pair detection and tracking
- `run_capture_windows.bat` - Run video capture (Windows)
- `run_track_smoother.bat` - Run track smoothing tool
- `run_calibrate_image.bat` - Run image calibration
- `run_calibrate_video.bat` - Run video calibration
- `run_z_histogram.bat` - Run Z height histogram

### Windows PowerShell (.ps1)
- `run_visualize3d.ps1`
- `run_pair_detect.ps1`
- `run_capture_windows.ps1`
- `run_track_smoother.ps1`
- `run_calibrate_image.ps1`
- `run_calibrate_video.ps1`
- `run_z_histogram.ps1`

### Linux/Raspberry Pi (.sh)
- `run_visualize3d.sh` - Run 3D visualization tool
- `run_pair_detect.sh` - Run pair detection and tracking
- `run_capture_raspi.sh` - Run video capture (Raspberry Pi)
- `run_track_smoother.sh` - Run track smoothing tool
- `run_calibrate_image.sh` - Run image calibration
- `run_calibrate_video.sh` - Run video calibration
- `run_z_histogram.sh` - Run Z height histogram

## Usage

### First Time Setup (Linux/Raspberry Pi)

```bash
# Make all scripts executable (one-time setup)
# This makes all .sh files executable (simpler approach)
chmod +x *.sh

# Run any program (note the ./ before the script name)
./run_visualize3d.sh
```

**Why `./` and `chmod +x`?**
- `./` is required - Linux needs this to run scripts in the current directory
- `chmod +x` makes files executable - only needed once per file
- See `LINUX_BASICS.md` for detailed explanation

### First Time Setup (Windows)

Just double-click any `run_*.bat` file, or run from command prompt:
```batch
run_visualize3d.bat
```

Or use PowerShell:
```powershell
.\run_visualize3d.ps1
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

