---
layout: default
title: Usage Guide
permalink: /usage/
---

# Usage Guide

This guide walks you through the complete workflow for using the 3D-Cam system.

## Step 1: Capture Video

### On Raspberry Pi:
```bash
./capture_raspi.sh
# Or manually: python apps/capture_raspi.py
```
- UVC cameras are tested and supported
- PiCamera support coming soon

### On Windows:
```bash
capture_windows.bat
# Or manually: python apps/capture_windows.py
```

**Procedure:**
1. Open the capture application
2. Select your camera
3. Position a ruler or calibration target with known millimeter markings on the mirror surface
4. Capture a single frame image with the scale visible
5. Record video(s) of objects moving at constant Z heights (different heights for calibration)
   - **Note**: Any number of objects can be in each video, as long as they all move at the same constant Z height for that video
   - When clicking "Record", a dialog will prompt for an optional filename prefix:
     - Enter a prefix (e.g., "experiment1") to create custom filenames like `experiment1_YYYYMMDD_HHMMSS.mp4`
     - Leave blank or cancel to use auto-generated names like `video_1920x1080_30fps_YYYYMMDD_HHMMSS.mp4`

**Output:** Videos saved to `inputs_outputs/{prefix}_YYYYMMDD_HHMMSS/` or `inputs_outputs/video_[W]x[H]_[FPS]fps_YYYYMMDD_HHMMSS/` directory

## Step 2: Image Calibration (Scale Calibration)

```bash
# Linux/Raspberry Pi:
./calibrate_scale_raspi.sh

# Windows:
calibrate_scale_windows.bat
```

**Purpose:** Determine the pixels-per-millimeter (px/mm) scale and working distance.

**Procedure:**
1. Load the captured frame image with the millimeter scale
2. Click two points that correspond to a known distance (e.g., 34 mm between two ruler marks)
3. Enter the measurement in millimeters
4. Enter camera parameters:
   - Focal length (mm)
   - Pixel size (microns)
   - Sensor dimensions (mm)
5. Click "Calculate" to compute:
   - `pixels_per_mm`: Scale factor for converting pixel measurements to millimeters
   - `working_distance_mm`: Distance from camera to the reflection surface

**Output:** Calibration JSON file saved to `calibrations/{image_name}_image_calibration_YYYYMMDD_HHMMSS.json`

## Step 3: Pair Detection

**Recommended: Standard Blob Detection**

```bash
# Linux/Raspberry Pi:
./detect_pairs.sh

# Windows:
detect_pairs.bat
```

**Alternative: Watershed Detection (for overlapping particles)**

```bash
# Linux/Raspberry Pi:
./detect_pairs_watershed.sh

# Windows:
python apps/detect_pairs_watershed.py
```

**Purpose:** Detect particle pairs (direct view + mirror reflection) and track them through the video.

**Procedure:**
1. Select a video folder (automatically detects the base video file)
2. Set the optical center (where particles align along radial lines from the mirror edge):
   - **Initial estimate**: Click in the preview window to set a rough optical center
   - **Optimize**: Click "Optimize Optical Center" button to analyze all frames and find the optimal center
   - **Iterate**: Repeat the optimization step until the center position stops changing
3. Tune detection parameters:
   - **Threshold**: Binary threshold for particle detection (0-255)
   - **Blur**: Gaussian blur kernel size to reduce noise
   - **Invert Threshold**: Checkbox to enable inverted threshold mode
   - **Min/Max Area**: Size constraints for valid particles (px²)
4. Adjust pairing weights and choose pairing algorithm (Hungarian recommended)
5. Export the processed video with tracked pairs

**Output:** All files saved in the same folder as the input video:
- Grayscale video with overlays: `{video_name}-grayscale.mp4`
- Binary video with overlays: `{video_name}-binary.mp4`
- CSV file with all pair data: `{video_name}-paired-tracked.csv`
- Preset file: `detect_pairs_preset.json`

## Step 4: Video Calibration (Z-Height Calibration)

```bash
# Linux/Raspberry Pi:
./calibrate_video.sh

# Windows:
calibrate_video.bat
```

**Purpose:** Calibrate the Z-height measurement by using videos of objects at known heights. You need at least 2 videos at different Z heights to perform linear regression.

**Procedure:**
1. Enter the global working distance (mm) - should match the value from image calibration
2. For each calibration CSV:
   - Browse to select a CSV file from pair detection
   - Enter the known Z height (mm) above the reflection surface for that CSV
3. Click "Calculate" to determine:
   - `magic_constant`: Linear scaling factor
   - `magic_offset`: Offset in millimeters
4. Calibration is automatically saved to the `calibrations` folder

**Output:** Calibration JSON file automatically saved to `calibrations/{csv1}_{csv2}_{...}_video_calibration_YYYYMMDD_HHMMSS.json`

## Step 5: Track Smoothing (Optional)

```bash
# Linux/Raspberry Pi:
./smooth_tracks.sh

# Windows:
smooth_tracks.bat
```

**Purpose:** Smooth trajectories and remove noise spikes from tracking data.

**Procedure:**
1. Load a CSV file from your pair detection results
2. Adjust smoothing parameters:
   - **Window Size**: Moving average window (larger = more smoothing)
   - **Spike Threshold**: Statistical outlier threshold
   - **Velocity Threshold**: Maximum allowed velocity jump
3. Toggle display options and view smoothness metrics
4. Export cleaned CSV data

**Output:** Smoothed CSV saved as `{csv_name}-smoothed.csv` in the same folder as the CSV

## Step 6: 3D Visualization

```bash
# Linux/Raspberry Pi:
./visualize_3d.sh

# Windows:
visualize_3d.bat
```

**Purpose:** Interactively visualize 3D trajectories from processed pair data.

**Procedure:**
1. Load a CSV file from your pair detection results
2. Use time slider to scrub through frames
3. Toggle trail visualization and adjust trail length
4. Select specific tracks to display
5. Rotate/zoom/pan the 3D view
6. Export animated video of the 3D trajectories

**Output:** 3D plot video saved as `{csv_name}-3dplot.mp4` in the same folder as the CSV

## Step 7: Z Height Histogram Analysis

```bash
# Linux/Raspberry Pi:
./plot_z_histogram.sh

# Windows:
plot_z_histogram.bat
```

**Purpose:** Analyze and visualize the distribution of Z heights in your tracked data.

**Procedure:**
1. Load a CSV file (automatically loads latest from inputs_outputs, or manually browse)
2. View the histogram showing Z height distribution
3. Adjust histogram settings (bins, log scale)
4. View statistics panel (mean, median, standard deviation, min/max)
5. Export histogram as PNG image

**Output:** Histogram image saved as `{csv_name}-histogram.png` in the same folder as the CSV

## Complete Workflow Summary

1. **Capture** → Record video with calibration frame
2. **Image Calibration** → Determine pixels-per-millimeter scale
3. **Pair Detection** → Detect and track particle pairs
4. **Video Calibration** → Calibrate Z-height measurements
5. **Track Smoothing** (Optional) → Clean trajectories
6. **3D Visualization** → View 3D trajectories
7. **Z Height Analysis** → Analyze height distribution

For more detailed information, see the [main README](https://github.com/e-lab-sfsu/3D-Cam/blob/main/README.md) in the repository.

