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
2. **Image Calibration** (`calibrate_image.py`): Determine pixels-per-millimeter scale
3. **Pair Detection** (`pair_detect.py`): Detect and track particle pairs in video
4. **Video Calibration** (`calibrate_video.py`): Calibrate Z-height measurements using known heights
5. **3D Visualization** (`visualize3d.py`): Visualize 3D trajectories interactively

## Usage

### Step 1: Capture Video

#### On Raspberry Pi:
```bash
python capture_raspi.py
```
- UVC cameras are tested and supported
- PiCamera support coming soon

#### On Windows:
```bash
python capture_windows.py
```

**Procedure:**
1. Open the capture application
2. Select your camera
3. Position a ruler or calibration target with known millimeter markings on the mirror surface
4. Capture a single frame image with the scale visible
5. Record video(s) of objects moving at constant Z heights (different heights for calibration)
   - **Note**: Any number of objects can be in each video, as long as they all move at the same constant Z height for that video

**Output:** Videos saved to `capture_output/` directory

### Step 2: Image Calibration (Scale Calibration)

```bash
python calibrate_image.py
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

**Math Behind It:**

The working distance is calculated using the thin lens equation:

```
working_distance = (focal_length × object_size) / (image_size_on_sensor)
```

where:
- `image_size_on_sensor = pixel_distance × pixel_size_mm`
- `pixel_distance = √((x₂ - x₁)² + (y₂ - y₁)²)` (in pixels)
- `pixel_size_mm = pixel_size_microns / 1000`

**Output:** Calibration JSON file saved to `calibrations/image_calibration_YYYYMMDD_HHMMSS.json`

### Step 3: Pair Detection

```bash
python pair_detect.py
```

**Purpose:** Detect particle pairs (direct view + mirror reflection) and track them through the video.

**Procedure:**
1. Open a video file from `capture_output/` or `input/`
2. Set the optical center (where particles align along radial lines from the mirror edge):
   - **Initial estimate**: Click in the preview window to set a rough optical center
   - **Optimize**: Click "Optimize Optical Center" button to analyze all frames and find the optimal center using ray intersection voting
   - **Iterate**: Repeat the optimization step until the center position stops changing (converges)
   - **Manual refinement**: You can still manually click to adjust if needed, then re-optimize
3. Tune detection parameters:
   - **Threshold**: Binary threshold for particle detection (0-255)
   - **Blur**: Gaussian blur kernel size to reduce noise
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
6. Export the processed video with tracked pairs

**Detection Method:**

1. **Binary Thresholding**: Convert grayscale to binary using adaptive threshold
2. **Background Subtraction**: Remove static background using running average
3. **Blob Detection**: Find connected components that meet size constraints
4. **Pair Matching**: Score candidate pairs based on:
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

**Output:** 
- Processed video with overlays: `pair_detect_output/[video_name]_[timestamp]/tracked_export.mp4`
- CSV file with all pair data: `pair_detect_output/[video_name]_[timestamp]/pairs.csv`

### Step 4: Video Calibration (Z-Height Calibration)

```bash
python calibrate_video.py
```

**Purpose:** Calibrate the Z-height measurement by using videos of objects at known heights. You need at least 2 videos at different Z heights to perform linear regression.

**Procedure:**
1. Enter the global working distance (mm) - should match the value from image calibration
2. For each calibration video:
   - Browse to the `pair_detect_output` folder containing `pairs.csv`
   - Enter the known Z height (mm) above the reflection surface
   - **Note**: Any number of objects can be in the video, as long as they all move at the same constant Z height
3. Click "Calculate" to determine:
   - `magic_constant`: Linear scaling factor
   - `magic_offset`: Offset in millimeters
4. Save the calibration data

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

**Output:** Calibration JSON file saved to `calibrations/video_calibration_YYYYMMDD_HHMMSS.json`

### Step 5: 3D Visualization

```bash
python visualize3d.py
```

**Purpose:** Interactively visualize 3D trajectories from processed pair data.

**Procedure:**
1. Load a CSV file from `pair_detect_output` (requires X_mm, Y_mm, Z_mm columns)
2. Use time slider to scrub through frames
3. Toggle trail visualization and adjust trail length
4. Select specific tracks to display
5. Rotate/zoom/pan the 3D view
6. Export animated video of the 3D trajectories

**3D Coordinate Calculation:**

Once calibrated, full 3D coordinates are computed:

1. **Z coordinate** (height):
   ```
   Zprime = working_distance × (C - A) / (A + C)
   Z = Zprime × magic_constant + magic_offset
   ```

2. **B point** (midpoint radius):
   ```
   B = (2 × A × C) / (A + C)
   ```
   The B point represents the radial distance of the particle from the optical center at the reflection surface plane.

3. **X, Y coordinates** (horizontal position):
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

**Output:** Video file saved to `3dvis_output/` directory

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

Averaged background model built from stationary pixels:
- Running average: `background = α × background + (1-α) × frame`
- Only pixels with small temporal variation contribute
- Threshold: `diff < static_thresh` (default 6 pixels)
- Minimum static ratio: `min_static_ratio` of frame must be static

## File Structure

```
3D-Cam/
├── capture_raspi.py          # Raspberry Pi camera capture
├── capture_windows.py         # Windows camera capture
├── calibrate_image.py         # Image scale calibration
├── calibrate_video.py         # Z-height calibration
├── pair_detect.py             # Main pair detection and tracking
├── visualize3d.py             # 3D trajectory visualization
├── lib/                       # Library modules
│   ├── camera.py              # Camera abstraction
│   ├── pair/                  # Pair detection algorithms
│   │   ├── pair_algorithms.py # Detection and pairing logic
│   │   ├── pair_draw.py       # Visualization overlays
│   │   ├── pair_tracker.py    # Multi-frame tracking
│   │   └── preset_io.py       # Settings persistence
│   └── capture/               # Video capture utilities
├── calibrations/              # Calibration JSON files
├── capture_output/            # Raw captured videos
├── pair_detect_output/        # Processed pair data
│   └── [video_name]_[timestamp]/
│       ├── pairs.csv          # Tracked pair data
│       ├── tracked_export.mp4 # Video with overlays
│       └── binary_overlay_export.mp4
└── 3dvis_output/              # 3D visualization videos
```

## Configuration Files

- `pair_preset.json`: Saves pair detection parameters and settings
- `tracker_config.json`: Tracking algorithm parameters
- `calibrations/*.json`: Calibration data (image and video)

## Dependencies

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- SciPy (for Hungarian algorithm)
- Matplotlib (for 3D visualization)
- Tkinter (for GUIs)

## Troubleshooting

### Poor Pair Detection
- Adjust threshold and blur parameters
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

