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

1. **Video Capture** (`capture_raspi.py`, `capture_windows.py`): Record video from USB cameras
2. **Image Calibration** (`calibrate_image_windows.py`, `calibrate_image_raspi.py`): Determine pixels-per-millimeter scale (platform-specific versions)
3. **Pair Detection** (`pair_detect.py`): Detect and track particle pairs in video
4. **Video Calibration** (`calibrate_video.py`): Calibrate Z-height measurements using known heights from CSV data
5. **Track Smoothing** (`track_smoother.py`): Smooth and clean trajectories, remove spikes
6. **3D Visualization** (`visualize3d.py`): Visualize 3D trajectories interactively
7. **Z Height Histogram** (`z_histogram.py`): Analyze and visualize Z height distribution

## Tools Summary

### Core Processing Tools

- **`pair_detect.py`** - Main pair detection and tracking tool. Detects particle pairs in video, tracks them across frames, and exports processed videos with CSV data containing pair coordinates and metadata.

- **`calibrate_image_windows.py`** / **`calibrate_image_raspi.py`** - Image scale calibration tool (platform-specific versions). Determines the pixels-per-millimeter scale factor and working distance by analyzing a captured frame with known millimeter measurements. The Windows version uses direct synchronous updates, while the Raspberry Pi version uses async updates to prevent GUI freezing.

- **`calibrate_video.py`** - Z-height calibration tool. Uses CSV files from pair detection at known heights to calculate the linear transformation constants needed to convert geometric Z measurements into calibrated heights. Automatically saves calibration files.

### Post-Processing Tools

- **`track_smoother.py`** - Track smoothing and cleaning tool. Removes spikes, applies smoothing filters, and compares original vs smoothed trajectories interactively.

### Visualization Tools

- **`visualize3d.py`** - Interactive 3D trajectory visualizer. Displays particle trajectories in 3D space with time scrubbing, track selection, trail visualization, and video export capabilities.

- **`z_histogram.py`** - Z height distribution analyzer. Creates histograms showing the frequency distribution of Z heights with logarithmic scale, adjustable bins, and statistical summaries (mean, median, standard deviation).

### Capture Tools

- **`capture_raspi.py`** - Raspberry Pi video capture application for recording videos from UVC cameras.

- **`capture_windows.py`** - Windows video capture application for recording videos from USB cameras.

## Quick Start

### Easy Setup and Run (Recommended)

The easiest way to get started is using the provided run scripts. They automatically set up the virtual environment if needed and run the programs.

#### On Windows:

**Using Batch Files (.bat):**
```batch
# Double-click or run:
run_visualize3d.bat
run_pair_detect.bat
run_capture_windows.bat
run_track_smoother.bat
run_calibrate_image.bat
run_calibrate_video.bat
run_z_histogram.bat
```

**Using PowerShell (.ps1):**
```powershell
# Run in PowerShell:
.\run_visualize3d.ps1
.\run_pair_detect.ps1
.\run_capture_windows.ps1
.\run_track_smoother.ps1
.\run_calibrate_image.ps1
.\run_calibrate_video.ps1
.\run_z_histogram.ps1
```

#### On Linux/Raspberry Pi:

```bash
# Make scripts executable (first time only)
# This makes all .sh files executable (includes run_*.sh and setup_venv.sh)
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

**Why `./` and `chmod +x`?**
- **`./`** means "current directory" - Linux requires this to run scripts in the current folder for security (you can't just type `run_visualize3d.sh`)
- **`chmod +x`** makes files executable - Linux doesn't automatically allow files to run for security reasons
- You only need to run `chmod +x` once per file (or use the wildcard `run_*.sh` to do them all at once)

**Note:** The run scripts will automatically:
1. Create a virtual environment if it doesn't exist
2. Install all dependencies
3. Activate the virtual environment
4. Run the program

### Manual Setup

If you prefer to set up manually, see the [Setup Instructions](#setup-instructions) below.

## Usage

### Step 1: Capture Video

#### On Raspberry Pi:
```bash
./run_capture_raspi.sh
# Or manually: python capture_raspi.py
```
- UVC cameras are tested and supported
- PiCamera support coming soon

#### On Windows:
```bash
run_capture_windows.bat
# Or manually: python capture_windows.py
```

**Procedure:**
1. Open the capture application
2. Select your camera
3. Position a ruler or calibration target with known millimeter markings on the mirror surface
4. Capture a single frame image with the scale visible
5. Record video(s) of objects moving at constant Z heights (different heights for calibration)
   - **Note**: Any number of objects can be in each video, as long as they all move at the same constant Z height for that video

**Output:** Videos saved to `inputs_outputs/video_[W]x[H]_[FPS]fps_YYYYMMDD_HHMMSS/` directory

### Step 2: Image Calibration (Scale Calibration)

```bash
# Linux/Raspberry Pi:
./run_calibrate_image.sh

# Windows:
run_calibrate_image.bat
# Or manually: python calibrate_image_windows.py
```

**Purpose:** Determine the pixels-per-millimeter (px/mm) scale and working distance.

**Procedure:**
1. Load the captured frame image with the millimeter scale
2. Click two points that correspond to a known distance (e.g., 34 mm between two ruler marks)
   - **Note**: The window is resizable on both platforms for better visibility
3. Enter the measurement in millimeters
4. Enter camera parameters:
   - Focal length (mm)
   - Pixel size (microns)
   - Sensor dimensions (mm)
5. Click "Calculate" to compute:
   - `pixels_per_mm`: Scale factor for converting pixel measurements to millimeters
   - `working_distance_mm`: Distance from camera to the reflection surface

**Platform-Specific Notes:**
- **Windows**: Uses direct synchronous updates for optimal performance
- **Raspberry Pi**: Uses async updates to prevent GUI freezing after clicking points

**Math Behind It:**

The working distance is calculated using the thin lens equation:

```
working_distance = (focal_length × object_size) / (image_size_on_sensor)
```

where:
- `image_size_on_sensor = pixel_distance × pixel_size_mm`
- `pixel_distance = √((x₂ - x₁)² + (y₂ - y₁)²)` (in pixels)
- `pixel_size_mm = pixel_size_microns / 1000`

**Output:** Calibration JSON file saved to `calibrations/{image_name}_image_calibration_YYYYMMDD_HHMMSS.json`

### Step 3: Pair Detection

```bash
# Linux/Raspberry Pi:
./run_pair_detect.sh

# Windows:
run_pair_detect.bat
# Or manually: python pair_detect.py
```

**Purpose:** Detect particle pairs (direct view + mirror reflection) and track them through the video.

**Procedure:**
1. Select a video folder (automatically detects the base video file, ignoring processed exports)
2. Set the optical center (where particles align along radial lines from the mirror edge):
   - **Initial estimate**: Click in the preview window to set a rough optical center
   - **Optimize**: Click "Optimize Optical Center" button to analyze all frames and find the optimal center using ray intersection voting
   - **Iterate**: Repeat the optimization step until the center position stops changing (converges)
   - **Manual refinement**: You can still manually click to adjust if needed, then re-optimize
3. Tune detection parameters:
   - **Threshold**: Binary threshold for particle detection (0-255)
   - **Blur**: Gaussian blur kernel size to reduce noise
   - **Invert Threshold**: Checkbox to enable inverted threshold mode
     - Unchecked (default): For **white particles on black background**
     - Checked: For **black particles on white background**
   - **Min/Max Area**: Size constraints for valid particles (px²)
   - **Pairing constraints**: Maximum radial gap, angle difference, center offset
4. Adjust pairing weights:
   - `w_theta`: Weight for angular similarity
   - `w_area`: Weight for area similarity
   - `w_center`: Weight for colinearity with optical center
5. Choose pairing algorithm:
   - **Greedy**: Fast, sequential matching
   - **Symmetric**: Ensures mutual best matches
   - **Hungarian**: Optimal global matching (recommended)
6. Export the processed video with tracked pairs (includes Load Process button to restore previous settings)

**Image Processing Pipeline:**

The detection system processes each video frame through the following pipeline:

1. **Frame Loading**: Load BGR color frame from video
2. **Grayscale Conversion**: Convert to single-channel grayscale image
3. **Background Subtraction**: Remove static background using pre-computed averaged background model
   - Background model is built once from the entire video at the start
   - Uses stationary pixel averaging: only pixels with low temporal variation contribute
   - Subtracts background: `result = |frame - background|`
4. **Contrast Enhancement**: Adjust contrast to improve particle visibility
   - Formula: `result = (pixel - 128) × contrast_factor + 128`
   - Default contrast factor: 100% (no change), adjustable 0-200%
5. **Gaussian Blur**: Smooth the image to reduce noise before thresholding
   - Kernel size: adjustable (1, 3, 5, 7, ... pixels)
   - Larger kernels = more smoothing but less detail
6. **Binary Thresholding**: Convert grayscale to binary (black/white) image
   - **Normal mode** (default): pixels above threshold → white (255), below → black (0)
     - Use for **white particles on black background**
   - **Inverted mode**: pixels above threshold → black (0), below → white (255)
     - Enable via checkbox: "Black particles on white background"
     - Use for **black particles on white background**
   - Threshold value: adjustable (0-255)
7. **Blob Detection**: Find connected components (blobs) in binary image
   - Uses contour detection to identify particle candidates
   - Filters by pixel area: `minArea ≤ actual_pixel_area ≤ maxArea`
   - Filters by dimensions: width/height ≤ `maxW`
   - Extracts blob properties: center position (xc, yc), bounding box, actual pixel area
   - Converts to polar coordinates (theta, radius) relative to optical center
8. **Pair Matching**: Score candidate pairs based on:
   - **Angular Similarity** (`S_theta`): How close are the angles from center?
     ```
     S_theta = 1 - (|θ_A - θ_C| / maxDMR)
     ```
   - **Area Similarity** (`S_area`): How similar are the blob areas?
     ```
     S_area = min(area_A, area_C) / max(area_A, area_C)
     ```
   - **Center Colinearity** (`S_center`): How well does the pair line pass through optical center?
     ```
     d_center = distance from optical center to line AC
     S_center = 1 - (d_center / maxCenterOff)
     ```
5. **Pair Score**: Weighted combination
   ```
   Score = w_theta × S_theta + w_area × S_area + w_center × S_center
   ```
   Pairs with `Score ≥ Smin` are accepted.

**Tracking Method:**

The system uses a sophisticated multi-frame tracking algorithm that maintains stable track IDs across the entire video. The tracker combines position prediction, velocity modeling, and morphing support for robust tracking.

**Core Algorithm:**

1. **Position Prediction**:
   - Each track maintains a current midpoint position and velocity vector
   - Predicted position = `previous_position + velocity`
   - This accounts for constant velocity motion between frames

2. **Multi-Criteria Matching**:
   Pairs are matched to tracks using a composite score that considers:
   
   a. **Distance Cost**: Euclidean distance from predicted position
   ```
   distance = ||predicted_position - candidate_position||
   ```
   
   b. **Velocity Smoothness**: Consistency of motion direction and speed
   - **Angle Consistency**: Measures how well the new velocity matches the previous velocity direction
     - Smaller angle difference between old and new velocity = smoother motion
     - Normalized to 0-1 scale (0° difference = 1.0, 180° = 0.0)
   - **Magnitude Consistency**: Measures speed stability
     - Coefficient of variation (CV) of velocity magnitudes across recent history
     - Lower variance = smoother motion
   - **Multi-frame Linearity**: Checks if motion follows a consistent direction over several frames
   
   c. **Size Morphing Smoothness**: Allows gradual size changes (blob area)
   - Tracks can gradually grow or shrink (0.7× to 1.4× per frame)
   - Abrupt size jumps are penalized
   - Uses coefficient of variation to detect smooth size transitions
   
   d. **Length Morphing Smoothness**: Allows gradual pair length changes
   - The distance between the two points (A and C) in a pair can change smoothly
   - Tracks can accommodate pairs that expand/contract (0.8× to 1.25× per frame)
   - Important for objects moving in/out of focus or changing orientation

3. **Composite Scoring**:
   ```
   score = distance + 
           (1 - velocity_smoothness) × velocity_penalty_scale +
           (1 - size_smoothness) × size_penalty_scale +
           (1 - length_smoothness) × length_penalty_scale
   ```
   Lower scores indicate better matches. The tracker uses greedy matching (best match first) to assign pairs to tracks.

4. **Track Lifecycle**:
   - **New Track**: Created for unmatched pairs (within `max_match_dist_px`)
   - **Active Track**: Updated when matched, velocity and properties updated
   - **Missed Track**: Increments miss counter when not matched in a frame
   - **Retired Track**: Removed after `max_misses` consecutive misses
   - **Stable ID**: Once assigned, track IDs persist throughout the video

5. **History Management**:
   - Maintains rolling history of recent velocities, sizes, and lengths
   - Default: Last 5 frames of history for smoothness calculations
   - Enables detection of motion trends and gradual morphing

**Key Parameters:**
- `max_match_dist_px`: Maximum distance (pixels) for matching (default: 25.0)
- `max_misses`: Frames to wait before retiring lost tracks (default: 10)
- Velocity smoothness weights: Angle (0.6) + Magnitude (0.4)
- Size ratio range: 0.7× to 1.4× per frame
- Length ratio range: 0.8× to 1.25× per frame

**Advantages:**
- Handles occlusions: Tracks survive temporary disappearances
- Adapts to motion changes: Velocity prediction handles acceleration/deceleration
- Morphing support: Accommodates objects that change size or shape
- Stable IDs: Consistent track IDs for reliable trajectory analysis

**Output:** All files saved in the same folder as the input video:
- Grayscale video with overlays: `{video_name}-grayscale.mp4` (or `-grayscale-N.mp4` if multiple exports)
- Binary video with overlays: `{video_name}-binary.mp4` (or `-binary-N.mp4` if multiple exports)
- CSV file with all pair data: `{video_name}-paired-tracked.csv` (or `-paired-tracked-N.csv` if multiple exports)
- Preset file: `pair_detect_preset.json` (or `pair_detect_preset-N.json` if multiple exports)

### Step 4: Video Calibration (Z-Height Calibration)

```bash
# Linux/Raspberry Pi:
./run_calibrate_video.sh

# Windows:
run_calibrate_video.bat
# Or manually: python calibrate_video.py
```

**Purpose:** Calibrate the Z-height measurement by using videos of objects at known heights. You need at least 2 videos at different Z heights to perform linear regression.

**Procedure:**
1. Enter the global working distance (mm) - should match the value from image calibration
2. For each calibration CSV:
   - Browse to select a CSV file from pair detection
   - Enter the known Z height (mm) above the reflection surface for that CSV
   - **Note**: Any number of objects can be in the video, as long as they all move at the same constant Z height
3. Click "Calculate" to determine:
   - `magic_constant`: Linear scaling factor
   - `magic_offset`: Offset in millimeters
4. Calibration is automatically saved to the `calibrations` folder

**Math Behind Video Calibration:**

The system uses a two-stage calibration process:

1. **Calculate Zprime** (intermediate Z value from geometry):
   ```
   Zprime = working_distance × (C - A) / (A + C)
   ```
   where:
   - `A` = inner radius (pixels) - distance from optical center to closer point
   - `C` = outer radius (pixels) - distance from optical center to farther point
   - `working_distance` = camera-to-mirror distance (mm)

   **Geometric Reasoning:** In the perpendicular mirror setup, the ratio `(C-A)/(A+C)` is proportional to the height above the mirror. Higher objects produce larger radial separation between the direct and reflected views.

2. **Data Collection**:
   - For each calibration video, the system analyzes the highest quality pairs (top 20% by score, or pairs with score > 0.8)
   - Calculates average Zprime and average B for these quality pairs
   - This works with any number of objects in the video, as long as they're all at the same constant Z height
   - More objects provide more data points and better statistics

3. **Linear Regression** to find calibration constants:
   ```
   Z = Zprime × magic_constant + magic_offset
   ```
   
   Using multiple videos (minimum 2) with known Z heights, we perform linear regression:
   - `Z` = known calibrated height (input) - one per video
   - `Zprime` = average Zprime calculated from pair geometry in that video (dependent variable)
   - `magic_constant` = slope from regression
   - `magic_offset` = intercept from regression

4. **Quality Metric**: R² (coefficient of determination) indicates calibration quality
   - R² close to 1.0 = excellent linear fit
   - Lower R² may indicate setup issues or measurement errors
   - Higher number of calibration videos (3+) improves reliability

**Why Multiple Objects Work:**
- The system averages Zprime values from the best quality pairs in each video
- As long as all objects in a video are at the same Z height, their Zprime values will cluster around the same value
- More objects provide more pair detections, improving the statistical reliability of the average Zprime
- This is especially useful for calibration at each height - you can move multiple objects simultaneously

**Output:** Calibration JSON file automatically saved to `calibrations/{csv1}_{csv2}_{...}_video_calibration_YYYYMMDD_HHMMSS.json`

### Step 5: Track Smoothing (Optional)

```bash
# Linux/Raspberry Pi:
./run_track_smoother.sh

# Windows:
run_track_smoother.bat
# Or manually: python track_smoother.py
```

**Purpose:** Smooth trajectories and remove noise spikes from tracking data.

**Procedure:**
1. Load a CSV file from your pair detection results
2. Adjust smoothing parameters:
   - **Window Size**: Moving average window (larger = more smoothing)
   - **Spike Threshold**: Statistical outlier threshold (higher = fewer spikes removed)
   - **Velocity Threshold**: Maximum allowed velocity jump
3. Toggle display options:
   - Show original trajectories
   - Show smoothed trajectories
   - Show detected spike points
4. View smoothness metrics comparing original vs smoothed data
5. Export cleaned CSV data

**Output:** Smoothed CSV saved as `{csv_name}-smoothed.csv` (or `-smoothed-N.csv` if multiple exports) in the same folder as the CSV

### Step 6: 3D Visualization

```bash
# Linux/Raspberry Pi:
./run_visualize3d.sh

# Windows:
run_visualize3d.bat
# Or manually: python visualize3d.py
```

**Purpose:** Interactively visualize 3D trajectories from processed pair data.

**Procedure:**
1. Load a CSV file from your pair detection results
2. Use time slider to scrub through frames
3. Toggle trail visualization and adjust trail length
4. Select specific tracks to display
5. Rotate/zoom/pan the 3D view
6. Export animated video of the 3D trajectories

**Output:** 3D plot video saved as `{csv_name}-3dplot.mp4` (or `-3dplot-N.mp4` if multiple exports) in the same folder as the CSV

**CSV Column Requirements:**
- **Frame_Number**: Frame index
- **Track_ID**: Unique track identifier
- **X/Y coordinates**: Either `X_mm`/`Y_mm` (calibrated mm) or `Center_X`/`Center_Y` (pixels)
- **Z coordinate**: Either `Z_mm` (fully calibrated) or `Zprime_mm` (geometric height, uncalibrated)

**Note:** If Z_mm is not available, the system automatically uses Zprime_mm for visualization. This allows you to visualize trajectories before completing video calibration.

**3D Coordinate Calculation:**

The coordinate calculation happens in two stages:

1. **Zprime** (geometric height from mirror setup):
   ```
   Zprime = working_distance × (C - A) / (A + C)
   ```
   - This is **always** calculated if working distance is available
   - Use Zprime for uncalibrated visualization

2. **Z** (fully calibrated height):
   ```
   Z = Zprime × magic_constant + magic_offset
   ```
   - Requires video calibration with known heights
   - Use Z for calibrated measurements

3. **B point** (midpoint radius):
   ```
   B = (2 × A × C) / (A + C)
   ```
   The B point represents the radial distance of the particle from the optical center at the reflection surface plane.

4. **X, Y coordinates** (horizontal position):
   ```
   B_mm = B_px / pixels_per_mm
   θ = atan2(midpoint_y - center_y, midpoint_x - center_x)
   X = B_mm × cos(θ)
   Y = -B_mm × sin(θ)  // Negative because image Y increases downward
   ```
   
   The midpoint between A and C gives the horizontal projection of the particle.

**Coordinate System:**
- **Origin (0, 0, 0)**: Optical center at the reflection surface
- **X-axis**: Horizontal (right = positive)
- **Y-axis**: Depth (forward = positive, accounting for image coordinate flip)
- **Z-axis**: Vertical height above mirror (up = positive)


### Step 7: Z Height Histogram Analysis

```bash
# Linux/Raspberry Pi:
./run_z_histogram.sh

# Windows:
run_z_histogram.bat
# Or manually: python z_histogram.py
```

**Purpose:** Analyze and visualize the distribution of Z heights in your tracked data.

**Procedure:**
1. Load a CSV file (automatically loads latest from inputs_outputs, or manually browse to select)
2. View the histogram showing Z height distribution:
   - **X-axis**: Z Height (mm)
   - **Y-axis**: Frequency (Count) - logarithmic scale
3. Adjust histogram settings:
   - **Bins**: Slider to control number of histogram bins (10-200)
   - **Log Scale**: Toggle logarithmic Y-axis
4. View statistics panel showing:
   - Mean Z height
   - Median Z height
   - Standard deviation
   - Minimum and maximum values
5. Export histogram as PNG image

**Use Cases:**
- Analyze height distribution of particles in your video
- Verify calibration quality (should see expected height clusters)
- Detect anomalies or outliers in height measurements
- Compare distributions across different experimental conditions

**Requirements:** CSV file must contain `Z_mm` or `Zprime_mm` column. If Z_mm is not available, Zprime_mm will be used automatically for uncalibrated visualization.

**Output:** Histogram image saved as `{csv_name}-histogram.png` (or `-histogram-N.png` if multiple exports) in the same folder as the CSV

## Methods and Algorithms

### Pair Detection Algorithms

Three pairing algorithms are available, each with different characteristics:

1. **Greedy Algorithm** (`pair_scored`):
   - Sequential matching: processes blobs in order
   - Each blob finds its best match from remaining unmatched blobs
   - Fast O(n²) complexity
   - May not produce optimal global matching

2. **Symmetric Algorithm** (`pair_scored_symmetric`):
   - Ensures mutual best matches (A→B and B→A both best)
   - More stable than greedy
   - Still O(n²) but with bidirectional checking
   - Good for simple scenes

3. **Hungarian Algorithm** (`pair_scored_hungarian`):
   - Optimal global matching using linear sum assignment
   - Maximizes total score across all pairs
   - O(n³) complexity, but very robust
   - Recommended for complex scenes with many particles

### Optical Center Detection

The optical center is the point where particle pairs align along radial lines. The system supports iterative refinement:

1. **Initial Estimate**: 
   - Manual click in the preview window to set rough center
   - Or uses center from previous video/session if available
   - Defaults to frame center if no prior estimate exists

2. **Automatic Optimization**: Uses ray intersection voting
   - Analyzes all pair lines across all frames
   - Finds intersections between pair lines (where lines through A and C points meet)
   - Votes for grid cells where intersections cluster
   - Selects cell with most votes as optimal center
   - This optimization should be run iteratively until convergence

3. **Iterative Refinement**:
   - After initial optimization, pair detection improves (better center = better pairs)
   - Re-run optimization with improved pair detections
   - Repeat until the center position stabilizes (stops moving between iterations)
   - Typically converges in 2-3 iterations

**Why Iteration is Important**: 
- The optimization uses detected pairs to find the center
- With a better center, pair detection improves (higher quality pairs)
- Improved pairs lead to better center estimation
- This feedback loop converges to the true optical center

### Background Subtraction

The system builds an averaged background model from the entire video before processing:

**Background Model Construction:**
1. **Two-Pass Analysis**: 
   - First pass: Running average to identify stationary regions
   - Second pass: Averaged accumulation of only stationary pixels

2. **Running Average** (for motion detection):
   ```
   bg_run = α × bg_run + (1-α) × frame
   ```
   - `α = 0.95` (default): Higher values = slower adaptation
   - Tracks overall scene brightness changes

3. **Stationary Pixel Detection**:
   ```
   diff = |frame - bg_run|
   stationary_mask = (diff < static_thresh)
   ```
   - `static_thresh = 6` (default): Pixels with variation < 6 are considered static
   - Only frames where ≥80% of pixels are static contribute to background

4. **Accumulated Averaging** (for final background):
   - Only stationary pixels from qualifying frames are averaged
   - Final background: `background = accumulated_sum / pixel_count`
   - Results in a clean background model free of moving objects

**Background Subtraction Application**:
- Applied to each frame during detection: `result = |frame_gray - background|`
- Highlights moving objects (particles) while suppressing static scene elements
- Critical for detecting small particles against complex backgrounds

## File Structure

```
3D-Cam/
├── capture_raspi.py          # Raspberry Pi camera capture
├── capture_windows.py         # Windows camera capture
├── calibrate_image_windows.py # Image scale calibration (Windows)
├── calibrate_image_raspi.py   # Image scale calibration (Raspberry Pi)
├── calibrate_video.py         # Z-height calibration
├── pair_detect.py             # Main pair detection and tracking
├── track_smoother.py          # Track smoothing and cleaning
├── visualize3d.py             # 3D trajectory visualization
├── z_histogram.py             # Z height distribution histogram
├── lib/                       # Library modules
│   ├── pair/                  # Pair detection and tracking
│   │   ├── pair_algorithms.py # Detection and pairing logic
│   │   ├── pair_draw.py       # Visualization overlays
│   │   ├── pair_tracker.py    # Multi-frame tracking
│   │   ├── preset_io.py       # Settings persistence
│   │   └── ui.py              # GUI for pair detection
│   ├── capture/               # Video capture and camera
│   │   ├── camera.py          # Camera abstraction
│   │   ├── camera_info.py     # Camera information and controls
│   │   ├── frame_grabber.py   # Frame acquisition thread
│   │   ├── preview_manager.py # Preview window manager
│   │   ├── recording_manager.py # Video recording manager
│   │   └── util_paths.py      # Path utilities
│   └── visualizing/           # Visualization components
│       └── base_visualizer.py # Base 3D visualizer class
├── calibrations/              # Calibration JSON files
├── inputs_outputs/            # All video and data outputs organized by capture
│   └── video_[W]x[H]_[FPS]fps_YYYYMMDD_HHMMSS/
│       ├── video_[W]x[H]_[FPS]fps_YYYYMMDD_HHMMSS.mp4  # Original capture
│       ├── *-grayscale.mp4    # Processed grayscale video
│       ├── *-binary.mp4       # Processed binary video
│       ├── *-paired-tracked.csv  # Pair tracking data
│       ├── *-smoothed.csv     # Smoothed data (from track_smoother)
│       ├── *-3dplot.mp4       # 3D visualization
│       ├── *-3dplot-smoothed.mp4  # 3D visualization of smoothed data
│       ├── *-histogram.png    # Z height histogram
│       ├── pair_detect_preset.json  # Processing parameters
│       └── track_smoother_preset.json  # Smoothing parameters
```

## File Organization and Naming Conventions

### Centralized Output Structure

All outputs are organized by capture session in the `inputs_outputs/` directory:

**Capture** → Creates unique folder: `video_[W]x[H]_[FPS]fps_YYYYMMDD_HHMMSS/`
- Original video saved inside this folder

**Pair Detection** → Processes video in the same folder:
- `{video_name}-grayscale.mp4` - Grayscale video with overlays
- `{video_name}-binary.mp4` - Binary video with overlays
- `{video_name}-paired-tracked.csv` - Tracking data
- `pair_detect_preset.json` - Processing parameters and calibration

**Post-Processing** → Uses CSV as input, saves outputs in same folder:
- `{csv_name}-smoothed.csv` - Cleaned tracking data
- `{csv_name}-3dplot.mp4` - 3D visualization
- `{csv_name}-3dplot-smoothed.mp4` - 3D visualization of smoothed data
- `{csv_name}-histogram.png` - Z height distribution
- `track_smoother_preset.json` - Smoothing parameters

### Multiple Export Handling

When exporting multiple times from the same input:
- First export: Uses base suffix (e.g., `-grayscale.mp4`)
- Subsequent exports: Appends counter (e.g., `-grayscale-1.mp4`, `-grayscale-2.mp4`)
- Prevents overwriting and allows version comparison

### Input Methods

- **Video folders**: `pair_detect.py` - Selects base video automatically
- **CSV files**: All other tools - Direct file selection
- **JSON presets**: "Load Process" button - Restores previous settings

## Configuration Files

- `pair_detect_default.json`: Default settings loaded on startup or when opening a new video  
- `lib/pair/tracker_config.json`: Advanced tracking algorithm smoothness parameters  
  - Controls how pairs are matched across frames (velocity, size, length consistency)  
  - Not exposed in GUI - modify directly for advanced tuning  
- `calibrations/*.json`: Calibration data (image and video)

## Dependencies

- Python 3.7+
- OpenCV (`cv2`)
- NumPy 2.x (with NumPy 2.x compatible packages)
- SciPy (for Hungarian algorithm)
- Matplotlib 3.9+ (supports NumPy 2.x)
- Tkinter (for GUIs)

### Setup Instructions

**Important**: On modern Linux systems (including Raspberry Pi OS) and for clean dependency management, you should use a virtual environment.

#### Automatic Setup (Recommended)

The run scripts handle everything automatically! Just run any `run_*.sh` (Linux) or `run_*.bat`/`run_*.ps1` (Windows) script and it will:
- Create the virtual environment if needed
- Install all dependencies
- Run the program

#### Manual Setup (Optional)

If you prefer to set up manually:

**Linux/Raspberry Pi:**
```bash
# Make setup script executable
chmod +x setup_venv.sh

# Run setup (creates venv and installs dependencies)
./setup_venv.sh

# Activate virtual environment
source venv/bin/activate

# Now you can run scripts manually
python visualize3d.py
```

**Windows:**
```batch
REM Run setup script
setup_venv.bat

REM Or manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### System Requirements

**Linux/Raspberry Pi:**
```bash
# Install if not already installed:
sudo apt install python3-venv python3-full
```

**Windows:**
- Python 3.7+ from [python.org](https://www.python.org/downloads/)
- Make sure Python is added to PATH during installation

**Note:** This project supports Windows and Raspberry Pi (Linux) only. macOS support has been removed.

See `SETUP.md` for detailed setup instructions and troubleshooting.

## Troubleshooting

### Setup and Installation Issues

**"externally-managed-environment" error**
- Your system uses externally managed Python environments (PEP 668)
- Use a virtual environment as described in the Setup section above
- See `SETUP.md` for detailed instructions

**NumPy/Matplotlib compatibility errors**
- Ensure you're using matplotlib 3.9+ which supports NumPy 2.x
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`
- Make sure virtual environment is activated


### Poor Pair Detection
- Adjust threshold and blur parameters
- **Check threshold mode**: Enable "Black particles on white background" if your particles are dark on a light background
- Ensure good lighting and contrast
- Check that optical center is correctly set
- Try different pairing algorithm (Hungarian recommended)

### Z-Height Inaccuracy
- Ensure working distance is accurate
- Use multiple calibration videos at different heights
- Check R² value in calibration (should be > 0.95)
- Verify objects are at constant height during calibration

### Missing 3D Coordinates in CSV
- Ensure video calibration JSON is loaded in `pair_detect.py`
- Check that image calibration provides `pixels_per_mm`
- Verify both calibrations are completed before export

## Future Improvements

- PiCamera support in `capture_raspi.py`
- Real-time processing mode
- Advanced filtering and smoothing of trajectories
- Batch processing of multiple videos
- Export to common trajectory formats (HDF5, CSV variants)

## License

[Specify your license here]

