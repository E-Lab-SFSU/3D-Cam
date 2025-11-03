# Setup Instructions for 3D-Cam

This project uses a virtual environment to manage dependencies, which is required on modern Linux systems (including Raspberry Pi OS) that use externally managed Python environments.

## Easiest Way: Use Run Scripts (Recommended)

The simplest way to get started is using the provided run scripts. They automatically handle everything!

### On Windows:

**Double-click or run any of these:**
- `run_visualize3d.bat` - 3D visualization
- `run_pair_detect.bat` - Pair detection and tracking
- `run_capture_windows.bat` - Video capture
- `run_track_smoother.bat` - Track smoothing
- `run_calibrate_image.bat` - Image calibration
- `run_calibrate_video.bat` - Video calibration
- `run_z_histogram.bat` - Z height histogram

**Or in PowerShell:**
```powershell
.\run_visualize3d.ps1
.\run_pair_detect.ps1
# etc.
```

### On Linux/Raspberry Pi:

```bash
# Make scripts executable (first time only)
# This makes all .sh files executable (simpler than listing them separately)
chmod +x *.sh

# Run any program (note the ./ before the script name):
./run_visualize3d.sh
./run_pair_detect.sh
./run_capture_raspi.sh
./run_track_smoother.sh
./run_calibrate_image.sh
./run_calibrate_video.sh
./run_z_histogram.sh
```

**Quick explanation:**
- **`./`** = tells Linux to run the script from the current directory (required for security)
- **`chmod +x`** = makes files executable (only needed once)
- **`run_*.sh`** = the `*` is a wildcard matching all files starting with "run_" and ending with ".sh"

See `LINUX_BASICS.md` for more detailed explanation of Linux script basics.

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

5. **Install project dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Using the Virtual Environment

### Using Run Scripts (Recommended)

Just run the appropriate `run_*.bat` (Windows) or `run_*.sh` (Linux) script - it handles everything automatically!

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
python visualize3d.py
python pair_detect.py
python capture_raspi.py
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

# Upgrade packages
pip install --upgrade -r requirements.txt
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
source venv/bin/activate
pip install -r requirements.txt
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

See `requirements.txt` for exact version specifications.

