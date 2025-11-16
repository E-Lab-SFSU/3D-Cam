---
layout: default
title: Setup Instructions
permalink: /setup/
---

# Setup Instructions for 3D-Cam

This project uses a virtual environment to manage dependencies, which is required on modern Linux systems (including Raspberry Pi OS) that use externally managed Python environments.

## Easiest Way: Use Run Scripts (Recommended)

The simplest way to get started is using the provided run scripts. They automatically handle everything!

### On Windows:

**Double-click or run any of these:**
- `visualize_3d.bat` - 3D visualization
- `detect_pairs.bat` - Pair detection and tracking
- `capture_windows.bat` - Video capture
- `smooth_tracks.bat` - Track smoothing
- `calibrate_scale_windows.bat` - Image calibration
- `calibrate_video.bat` - Video calibration
- `plot_z_histogram.bat` - Z height histogram

**Or in PowerShell:**
```powershell
.\visualize_3d.ps1
.\detect_pairs.ps1
# etc.
```

### On Linux/Raspberry Pi:

```bash
# Make scripts executable (first time only)
chmod +x *.sh

# Run any program (note the ./ before the script name):
./visualize_3d.sh
./detect_pairs.sh
./capture_raspi.sh
./smooth_tracks.sh
./calibrate_scale_raspi.sh
./calibrate_video.sh
./plot_z_histogram.sh
```

**Quick explanation:**
- **`./`** = tells Linux to run the script from the current directory (required for security)
- **`chmod +x`** = makes files executable (only needed once)
- **`*.sh`** = the `*` is a wildcard matching all files starting with "run_" and ending with ".sh"

See [Linux Basics]({{ site.baseurl }}/linux-basics) for more detailed explanation of Linux script basics.

The run scripts will automatically:
1. Create a virtual environment if it doesn't exist
2. Install all dependencies
3. Activate the virtual environment
4. Run the program

## Manual Setup (Alternative)

If you prefer to set up manually or understand the process:

### Option 1: Use the Setup Script

```bash
# Make the script executable
chmod +x setup_venv.sh

# Run the setup script
./setup_venv.sh
```

### Option 2: Manual Setup

1. **Install required system packages** (if not already installed):
   ```bash
   sudo apt install python3-venv python3-full
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

4. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

5. **Install project dependencies** (pick your platform file):
   ```bash
   # Windows (requires Python 3.11 or 3.12 for prebuilt wheels)
   pip install -r requirements.windows.txt

   # Raspberry Pi / Linux
   pip install -r requirements.raspi.txt
   ```

## Using the Virtual Environment

### Using Run Scripts (Recommended)

Just run the appropriate `*.bat` (Windows) or `*.sh` (Linux) script - it handles everything automatically!

### Manual Activation (If Needed)

If you prefer to work manually, activate the virtual environment:

**Linux/Raspberry Pi:**
```bash
source venv/bin/activate
```

**Windows:**
```batch
venv\Scripts\activate.bat
```

**Windows PowerShell:**
```powershell
venv\Scripts\Activate.ps1
```

You'll see `(venv)` at the beginning of your command prompt, indicating the virtual environment is active.

### Run Your Scripts Manually

While the virtual environment is active, run your scripts normally:

```bash
python apps/visualize_3d.py
python apps/detect_pairs.py
python apps/capture_raspi.py
# etc.
```

### Deactivate the Virtual Environment

When you're done working:

```bash
deactivate
```

## Updating Dependencies

To update dependencies to the latest compatible versions:

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Upgrade packages (choose the file for your platform)
#   Windows: pip install --upgrade -r requirements.windows.txt
#   Raspberry Pi / Linux: pip install --upgrade -r requirements.raspi.txt
```

## Troubleshooting

### "python3-venv is not installed"

Install the required system packages:
```bash
sudo apt install python3-venv python3-full
```

### "Permission denied" when running setup script

Make the script executable:
```bash
chmod +x setup_venv.sh
```

### Virtual environment not activating

If you get errors activating the virtual environment, try recreating it:
```bash
rm -rf venv
python3 -m venv venv
# Linux / Raspberry Pi
source venv/bin/activate
pip install -r requirements.raspi.txt

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
pip install -r requirements.windows.txt
```

## Why Use a Virtual Environment?

- **Isolation**: Keeps your project dependencies separate from system packages
- **Compatibility**: Allows you to use specific package versions without conflicts
- **Modern Linux**: Required on systems with externally managed Python (PEP 668)
- **Clean**: Easy to remove and recreate if needed

## Dependencies

The project requires:
- Python 3.7+
- NumPy 2.x (with NumPy 2.x compatible packages)
- Matplotlib 3.9+ (supports NumPy 2.x)
- OpenCV 4.8+
- SciPy 1.11+
- Tkinter (usually included with Python)

See `requirements.base.txt` plus the OS-specific files for exact version specifications.

