---
layout: default
title: Home
permalink: /
---

# 3D Particle Tracking Camera

A 3D particle tracking system using a single camera and a perpendicular mirror. This system enables 3D position tracking of particles by analyzing their reflections in a mirror placed perpendicular to the camera's view.

## Purpose

This project implements a cost-effective alternative to multi-camera stereo vision systems for 3D particle tracking. By using a single camera with a perpendicular mirror, particles appear as pairs of points in the image (the direct view and the mirrored view). By analyzing these pairs, we can extract full 3D coordinates (X, Y, Z) of particles in real-world space.

### Key Advantages

- **Single Camera Setup**: No need for camera synchronization or calibration between multiple cameras
- **Simplified Hardware**: Only requires a camera and a mirror positioned at 90° to the camera
- **3D Reconstruction**: Extracts full 3D trajectories from 2D image data

## System Overview

The system consists of several components:

1. **Video Capture** (`apps/capture_raspi.py`, `apps/capture_windows.py`): Record video from USB cameras
2. **Image Calibration** (`apps/calibrate_scale_windows.py`, `apps/calibrate_scale_raspi.py`): Determine pixels-per-millimeter scale (platform-specific versions)
3. **Pair Detection** (`apps/detect_pairs.py`): Detect and track particle pairs in video using blob detection (recommended)
4. **Pair Detection Watershed** (`apps/detect_pairs_watershed.py`): Detect and track particle pairs using watershed segmentation for better handling of overlapping/touching particles
5. **Video Calibration** (`apps/calibrate_video.py`): Calibrate Z-height measurements using known heights from CSV data
6. **Track Smoothing** (`apps/smooth_tracks.py`): Smooth and clean trajectories, remove spikes
7. **3D Visualization** (`apps/visualize_3d.py`): Visualize 3D trajectories interactively
8. **Z Height Histogram** (`apps/plot_z_histogram.py`): Analyze and visualize Z height distribution

## Quick Start

### Easy Setup and Run (Recommended)

The easiest way to get started is using the provided run scripts. They automatically set up the virtual environment if needed and run the programs.

#### On Windows:

**Using Batch Files (.bat):**
```batch
# Double-click or run:
visualize_3d.bat
detect_pairs.bat
capture_windows.bat
smooth_tracks.bat
calibrate_scale_windows.bat
calibrate_video.bat
plot_z_histogram.bat
```

**Using PowerShell (.ps1):**
```powershell
# Run in PowerShell:
.\visualize_3d.ps1
.\detect_pairs.ps1
.\capture_windows.ps1
.\smooth_tracks.ps1
.\calibrate_scale_windows.ps1
.\calibrate_video.ps1
.\plot_z_histogram.ps1
```

#### On Linux/Raspberry Pi:

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

**Note:** The run scripts will automatically:
1. Create a virtual environment if it doesn't exist
2. Install all dependencies
3. Activate the virtual environment
4. Run the program

### Manual Setup

If you prefer to set up manually, see the [Setup Instructions]({{ site.baseurl }}/setup) page.

## Documentation

- **[Setup Guide]({{ site.baseurl }}/setup)** - Detailed installation and setup instructions
- **[Scripts Reference]({{ site.baseurl }}/scripts)** - Complete list of all available scripts
- **[Linux Basics]({{ site.baseurl }}/linux-basics)** - Guide for Linux/Raspberry Pi users

## Usage Workflow

1. **Capture Video** - Record video from your camera
2. **Image Calibration** - Determine pixels-per-millimeter scale
3. **Pair Detection** - Detect and track particle pairs
4. **Video Calibration** - Calibrate Z-height measurements
5. **Track Smoothing** (Optional) - Clean trajectories
6. **3D Visualization** - View 3D trajectories
7. **Z Height Analysis** - Analyze height distribution

For detailed usage instructions, see the [README](https://github.com/e-lab-sfsu/3D-Cam/blob/main/README.md) in the repository.

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- NumPy 2.x
- SciPy
- Matplotlib 3.9+
- Tkinter (for GUIs)

## License

[Specify your license here]

