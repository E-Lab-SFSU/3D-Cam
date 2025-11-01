#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pair Detector v4.5 — Single-window Tkinter GUI (stable, descriptive)

This app lets you:
  • Open a video file (persisted in preset)
  • Preview at ~15 FPS and click to set the optical center (live re-clickable)
  • Tune binarization (threshold, blur) and blobbing constraints
  • Pair blob candidates using a score that favors:
       - Similar angle from center (wθ)
       - Similar bounding-box area (wA)
       - Colinearity of the pair line relative to the center (wC)
  • Export (resave) the ENTIRE video (fast, non-real-time) to:
       - <prefix>_tracked_export.mp4  (grayscale + overlays)
       - <prefix>_binary_overlay_export.mp4  (binary + overlays)
       - <prefix>_pairs.csv  (UTF-8 metadata; no text burned into video)
  • Save/restore all settings + optical center + last video path to pair_preset.json

NOTES
  - “Tracked” OpenCV window: click once to update the optical center (yellow crosshair)
  - “Binary” window shows the thresholded/blurred mask used for detection
  - Checkboxes in GUI toggle overlays on both preview and export
  - Export prints progress (%) in the terminal
"""

import cv2, numpy as np, math, csv, json, os, time, sys
from typing import Optional
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from lib.pair.pair_algorithms import (
    detect as alg_detect,
    pair_scored as alg_pair_scored,
    pair_scored_symmetric as alg_pair_scored_symmetric,
    pair_scored_hungarian as alg_pair_scored_hungarian,
)
from lib.pair.pair_draw import (
    draw_blob_boxes as draw_blob_boxes_ext,
    draw_center as draw_center_ext,
    draw_pair_centers as draw_pair_centers_ext,
    draw_pair_lines as draw_pair_lines_ext,
    draw_pair_rays_toward_center as draw_pair_rays_toward_center_ext,
    set_video_seed as set_video_seed_ext,
    draw_stats_overlay as draw_stats_overlay_ext,
    draw_z_values as draw_z_values_ext,
    draw_xy_values as draw_xy_values_ext,
    draw_real_point as draw_real_point_ext,
)
from lib.capture.util_paths import ts_name, path_stem, export_paths_for
from lib.pair.preset_io import save_preset_file, load_preset_file
from lib.ui import build_gui as ui_build_gui, set_controls_enabled as ui_set_controls_enabled, reset_defaults_ui as ui_reset_defaults_ui
from lib.pair.pair_tracker import PairTracker

# ============ Persistent preset file ============
PRESET_PATH = "pair_preset.json"

# ============ Default numeric parameters (sliders) ============
DEFAULT_PARAMS = {
    # --- Binarization ---
    "threshold":     70,   # [0..255] Binary threshold applied after blur
    "blur":           1,   # Odd kernel size for Gaussian blur (1,3,5,7...). Smooths noise before threshold

    # --- Blob constraints ---
    "minArea":       20,   # Minimum area (px^2) for valid blob bounding box (w*h)
    "maxArea":     200,   # Maximum area (px^2) for valid blob
    "maxW":         100,   # Maximum width/height (px) for blob bounding box (rejects very large blobs)

    # --- Candidate gating for pairing ---
    "maxRadGap":    120,   # Max |rA - rC| (pixels) between two blobs' radial distances from center
    "maxDMR":        3,   # Max |θA - θC| (degrees) to consider blobs as potential pair

    # --- Colinearity constraint relative to center ---
    "maxCenterOff":  15,   # Scale (pixels). Pair-line distance to optical center is mapped to a 0..1 score via this scale

    # --- Pair scoring weights ---
    "w_theta":      0.35,  # Weight for angular similarity term
    "w_area":       0.35,  # Weight for area similarity term
    "w_center":     0.30,  # Weight for colinearity-to-center term

    # --- Acceptance threshold for a pair ---
    "Smin":         0.90,  # Score ≥ Smin is accepted (0..2 supported; typical 0.7..1.2)
    # --- Pairing method (UI dropdown) ---
    "pair_method":  "Hungarian",  # One of: Greedy, Symmetric, Hungarian
    
    # --- Tracking parameters ---
    "track_max_match_dist": 25.0,  # Max distance (px) for matching pairs across frames
    "track_max_misses":     10,    # Frames to wait before retiring a lost track
    
    # --- Background subtraction parameters ---
    "bg_alpha":           0.95,   # Running background update rate (0-1, higher = slower)
    "bg_static_thresh":   6,      # Pixel difference threshold for stationary detection
    "bg_min_static_ratio": 0.8,   # Minimum % of frame that must be static to include in averaging
    
    # --- Contrast enhancement ---
    "contrast":           100,    # Contrast multiplier (0-200, 100 = no change)
}

# ============ Default overlay toggles (checkboxes) ============
DEFAULT_OVERLAYS = {
    "show_blobs":   1,     # Draw green bounding boxes around detected blobs
    "show_center":  1,     # Draw yellow crosshair at the optical center
    "show_pair_center": 0, # Draw small circle at pair midpoint (for tracking visualization)
    "show_lines":   1,     # Draw white line between paired blobs
    "show_rays":    1,     # Extend pair line (white) in the AC direction toward/past the center
    "show_pair_points": 1, # Draw circles at pair endpoints (A and C points)
    "label_mode":   "Red/Blue",   # Label mode: "None", "Red/Blue", "Random"
    "show_text_labels": 1,         # Show #A/#C text labels on pairs
    "show_z_value": 0,     # Show Z value text labels on pairs (requires calibration)
    "show_xy_values": 0,   # Show X and Y value text labels on pairs (requires calibration)
    "show_real_point": 0,  # Show real point at B mm distance from center along ray (requires calibration)
    "show_current_stats": 0,  # Show current frame stats overlay
    "show_total_stats": 0,    # Show total stats overlay
}

DEFAULT_OVERLAY_TARGETS = {
    "enable_tracked": 1,
    "enable_binary":  0,
}
overlay_targets = DEFAULT_OVERLAY_TARGETS.copy()


# ============ Global state persisted to preset ============
params       = DEFAULT_PARAMS.copy()   # numeric tuning variables
overlays     = DEFAULT_OVERLAYS.copy() # overlay visibility toggles
video_path   = ""                      # last opened video path
last_video_path = ""                   # previous video path to detect new video loads
xCenter      = None                    # optical center x-coordinate (int pixels)
yCenter      = None                    # optical center y-coordinate (int pixels)
center_valid = False                   # True once user clicked in preview
showing_center_setup = False           # True when showing raw first frame for center setup

# ============ Calibration data for Z/B calculation ============
calibration_magic_constant: Optional[float] = None
calibration_magic_offset: Optional[float] = None
calibration_working_distance_mm: Optional[float] = None
calibration_pixels_per_mm: Optional[float] = None  # Pixels per mm for B to mm conversion
calibration_loaded = False

# ============ Live preview control ============
FPS_PREVIEW   = 15                     # Target preview FPS (smooth enough to tune)
DELAY_MS      = int(1000 / FPS_PREVIEW)
cap           = None                   # cv2.VideoCapture for preview loop
last_pair_method = None               # track changes to show overlay
tracker_preview = None                # persistent tracker for preview
total_pairs_count = 0                 # cumulative count of all pairs across all frames
max_pair_count = 0                    # maximum number of unique pairs seen in any single frame
background_image = None               # Averaged background image for subtraction (uint8 grayscale)
preview_playing = True                # Play/pause state for preview
video_total_frames = 0                # Total frames in current video (for trackbar)
trackbar_being_set = False            # Flag to prevent trackbar callback from triggering during programmatic updates
cached_frame = None                   # Cached frame when paused

# ============ Tkinter window refs ============
root          = None                   # main Tk window
widgets       = {}                     # store widget refs to enable/disable during export
gui_vars_numeric = {}  # StringVar/IntVar/DoubleVar for sliders
gui_vars_check   = {}  # IntVar for checkboxes
video_controls_window = None           # Separate window for video controls (trackbar + pause)
video_controls_widgets = {}            # Widgets in video controls window

# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------
def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v
def angdiff(a, b):    return abs(((a - b + 180) % 360) - 180)

 

# line_distance_to_point moved to pair_algorithms

# --------------------------------------------------------------------------------------
# Preset I/O (JSON)
# --------------------------------------------------------------------------------------
def save_preset():
    ok = save_preset_file(
        PRESET_PATH,
        params,
        overlays,
        overlay_targets,
        (xCenter, yCenter, center_valid),
        video_path,
    )
    if ok:
        print(f"[INFO] Preset saved → {PRESET_PATH}")
    else:
        print("[WARN] Preset save failed")

def load_preset():
    global params, overlays, xCenter, yCenter, center_valid, video_path, overlay_targets, last_video_path
    params, overlays, overlay_targets, center, video_path, ok = load_preset_file(
        PRESET_PATH, params, overlays, overlay_targets, video_path
    )
    xCenter, yCenter, center_valid = center
    # Initialize last_video_path to empty so first video load will be treated as new
    last_video_path = ""
    if ok:
        print(f"[INFO] Preset loaded ← {PRESET_PATH}")
    else:
        print("[INFO] No preset found, using defaults.")

# --------------------------------------------------------------------------------------
# Background subtraction and contrast enhancement
# --------------------------------------------------------------------------------------
def build_background_from_video(video_path: str) -> np.ndarray:
    """
    Build averaged background image from video using stationary pixel averaging.
    Returns uint8 grayscale background image.
    """
    global params
    tmpcap = cv2.VideoCapture(video_path)
    if not tmpcap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    
    bg_run = None
    bg_avg = None
    count = 0
    mask_sum = None
    
    alpha = params.get("bg_alpha", 0.95)
    static_thresh = params.get("bg_static_thresh", 6)
    min_static_ratio = params.get("bg_min_static_ratio", 0.8)
    
    print("[INFO] Building background model from video...")
    total_frames = int(tmpcap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_marks = set(int(total_frames * q / 10) for q in range(1, 10))
    frame_idx = 0
    
    while True:
        ret, frame = tmpcap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        if bg_run is None:
            bg_run = gray.copy()
            bg_avg = np.zeros_like(gray, dtype=np.float32)
            mask_sum = np.zeros_like(gray, dtype=np.float32)
            frame_idx += 1
            continue
        
        diff = cv2.absdiff(gray, bg_run)
        bg_run = alpha * bg_run + (1 - alpha) * gray
        
        stationary_mask = (diff < static_thresh).astype(np.float32)
        
        if stationary_mask.mean() > min_static_ratio:
            bg_avg += stationary_mask * gray
            mask_sum += stationary_mask
            count += 1
        
        if frame_idx in progress_marks:
            pct = 100 * frame_idx / max(1, total_frames)
            print(f"[INFO] Background analysis: {pct:.1f}% ({frame_idx}/{total_frames})")
        
        frame_idx += 1
    
    tmpcap.release()
    
    # Finalize averaged background
    mask_sum[mask_sum == 0] = 1.0
    bg_final = bg_avg / mask_sum
    bg_final8 = cv2.convertScaleAbs(bg_final)
    
    print(f"[INFO] Background model complete: {count} frames contributed to averaging.")
    return bg_final8

def apply_background_subtraction(frame_gray: np.ndarray) -> np.ndarray:
    """Apply background subtraction to a grayscale frame. Returns uint8 grayscale."""
    global background_image
    if background_image is None:
        return frame_gray
    diff = cv2.absdiff(frame_gray, background_image)
    return cv2.convertScaleAbs(diff)

def apply_contrast(frame_gray: np.ndarray) -> np.ndarray:
    """Apply contrast enhancement to a grayscale frame. Returns uint8 grayscale."""
    global params
    contrast_val = params.get("contrast", 100) / 100.0  # Convert 0-200 to 0.0-2.0
    if abs(contrast_val - 1.0) < 0.01:
        return frame_gray  # No change needed
    # Apply contrast: result = (pixel - 128) * contrast + 128
    result = frame_gray.astype(np.float32)
    result = (result - 128.0) * contrast_val + 128.0
    return np.clip(result, 0, 255).astype(np.uint8)

# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------
def set_centerxy(x, y):
    """Update optical center to (x,y) and mark as valid."""
    global xCenter, yCenter, center_valid, showing_center_setup
    xCenter, yCenter, center_valid = int(x), int(y), True
    showing_center_setup = False  # Center is set, switch to normal preview
    print(f"[INFO] Optical center -> ({xCenter},{yCenter})")

# polar_from_center moved to pair_algorithms

# --------------------------------------------------------------------------------------
# Detection, pairing, and drawing moved to modules: pair_algorithms, pair_draw
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Open / Export / Reset / Exit
# --------------------------------------------------------------------------------------
def prompt_optical_center_choice():
    """Show a custom dialog asking if user wants to choose a new optical center."""
    global root
    if root is None:
        # Fallback to simple messagebox if root doesn't exist yet
        return messagebox.askyesno(
            "Choose New Optical Center?",
            "Would you like to choose a new optical center for this video?\n\n"
            "Yes - choose new optical center\n"
            "No - the last optical center hasn't moved - the setup is the same"
        )
    
    dialog = tk.Toplevel(root)
    dialog.title("Choose New Optical Center?")
    dialog.geometry("600x150")
    dialog.resizable(False, False)
    dialog.transient(root)  # Make it modal relative to main window
    dialog.grab_set()  # Make it modal
    
    # Center the dialog on screen
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
    y = (dialog.winfo_screenheight() // 2) - (150 // 2)
    dialog.geometry(f"600x150+{x}+{y}")
    
    result = [None]  # Use list to store result from nested function
    
    # Message label
    msg_label = tk.Label(
        dialog,
        text="Would you like to choose a new optical center for this video?",
        font=("Arial", 10),
        wraplength=580,
        justify="center",
        pady=10
    )
    msg_label.pack()
    
    # Button frame
    button_frame = tk.Frame(dialog, pady=10)
    button_frame.pack()
    
    def choose_new():
        result[0] = True
        dialog.destroy()
    
    def keep_existing():
        result[0] = False
        dialog.destroy()
    
    # Buttons with exact labels requested by user
    btn_yes = ttk.Button(
        button_frame,
        text="Yes - choose new optical center",
        command=choose_new,
        width=35
    )
    btn_yes.pack(side="left", padx=5)
    
    btn_no = ttk.Button(
        button_frame,
        text="No - the last optical center hasn't moved - the setup is the same",
        command=keep_existing,
        width=55
    )
    btn_no.pack(side="left", padx=5)
    
    # Wait for dialog to close
    dialog.wait_window()
    
    return result[0]

def open_video():
    """Open a video file and start/restart preview."""
    global video_path, last_video_path, center_valid, xCenter, yCenter
    fp = filedialog.askopenfilename(
        title="Open Video",
        filetypes=[("Video", "*.mp4;*.avi;*.mov;*.mkv;*.m4v;*.mpg;*.mpeg"), ("All files", "*.*")]
    )
    if not fp:
        return
    
    # Normalize paths for comparison (handle Windows path case sensitivity)
    fp_normalized = os.path.normpath(os.path.abspath(fp))
    current_video_normalized = os.path.normpath(os.path.abspath(video_path)) if video_path else ""
    
    # Check if this is a new video (different from the currently loaded one)
    is_new_video = (current_video_normalized != "" and fp_normalized != current_video_normalized)
    
    print(f"[INFO] Opened video: {fp}")
    if is_new_video:
        print(f"[INFO] New video detected (previous: {video_path})")
    
    # If this is a new video and we have a valid center, prompt user
    if is_new_video and center_valid and xCenter is not None and yCenter is not None:
        # Show dialog asking if user wants to choose a new optical center
        response = prompt_optical_center_choice()
        
        if response:  # Yes - choose new optical center
            # Reset center validity so user can choose a new one
            center_valid = False
            print("[INFO] User chose to set a new optical center")
        else:  # No - keep existing center
            # Keep the existing center, just proceed
            print(f"[INFO] User chose to keep existing optical center -> ({xCenter},{yCenter})")
    
    # Update video path
    video_path = fp
    last_video_path = fp
    
    reopen_video()

def reopen_video():
    """(Re)create VideoCapture for preview loop and resize OpenCV windows to frame size."""
    global cap, background_image, root, widgets
    if cap is not None:
        cap.release()
        cap = None
    
    # Build background model from raw video
    print("[INFO] Processing video for background subtraction...")
    if root is not None and widgets:
        ui_set_controls_enabled(widgets, False)
        root.update_idletasks()
    try:
        background_image = build_background_from_video(video_path)
        print("[INFO] Background model ready.")
    except Exception as e:
        print(f"[ERR] Failed to build background model: {e}")
        background_image = None
    finally:
        if root is not None and widgets:
            ui_set_controls_enabled(widgets, True)
            root.update_idletasks()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERR] Could not open video: {video_path}")
        cap = None
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    print(f"[INFO] Video: {W}×{H}, fps≈{fps:.2f}")
    
    # Reset preview tracker when video (re)opens
    global tracker_preview, total_pairs_count, max_pair_count, center_valid, showing_center_setup, xCenter, yCenter
    tracker_preview = PairTracker(
        max_match_dist_px=float(params.get("track_max_match_dist", 25.0)),
        max_misses=int(params.get("track_max_misses", 10))
    )
    total_pairs_count = 0  # Reset total pairs count when video reopens
    max_pair_count = 0  # Reset max pair count when video reopens
    
    # Handle center setup based on validity
    if not center_valid:
        # If we have a last known center location from preset, use it as default but still show setup
        if xCenter is not None and yCenter is not None:
            # Center coordinates exist from preset, use them as default but allow user to change
            showing_center_setup = True
            print(f"[INFO] Showing center setup with preset default -> ({xCenter},{yCenter})")
        else:
            # No previous center exists, default to frame center and show setup
            xCenter, yCenter = W // 2, H // 2
            showing_center_setup = True
    else:
        # Center is already valid (user chose to keep existing), don't show setup
        showing_center_setup = False
        if xCenter is not None and yCenter is not None:
            print(f"[INFO] Using existing optical center -> ({xCenter},{yCenter})")
    
    # Set video seed for persistent colors based on filename
    set_video_seed_ext(video_path)
    
    # Ensure windows exist and resize to native video resolution (both same size)
    try:
        cv2.namedWindow("Tracked", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.namedWindow("Binary", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow("Tracked", W, H)
        cv2.resizeWindow("Binary", W, H)  # Same size as Tracked
    except Exception:
        pass  # Windows may already exist, ignore
    
    # Show raw first frame for center point placement (only if center not valid)
    if showing_center_setup:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if ret:
            # Display raw first frame (unprocessed) for center placement
            first_frame_display = first_frame.copy()  # Original BGR frame
            cv2.putText(first_frame_display, "Click to set optical center (Press Enter to confirm)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Show default/previous center location if available
            if xCenter is not None and yCenter is not None:
                cv2.circle(first_frame_display, (xCenter, yCenter), 10, (0, 255, 255), 2)
                cv2.line(first_frame_display, (xCenter - 15, yCenter), (xCenter + 15, yCenter), (0, 255, 255), 2)
                cv2.line(first_frame_display, (xCenter, yCenter - 15), (xCenter, yCenter + 15), (0, 255, 255), 2)
            cv2.imshow("Tracked", first_frame_display)
            cv2.setMouseCallback("Tracked", on_mouse_tracked)
            # Bring Tracked window to front to ensure it can receive keyboard input
            try:
                cv2.setWindowProperty("Tracked", cv2.WND_PROP_TOPMOST, 0)
            except:
                pass
            cv2.waitKey(1)  # Update display
    
    # Create or update video controls window
    setup_video_controls_window()


def optimize_optical_center():
    """Analyze all frames to find optimal optical center based on ray intersections."""
    global xCenter, yCenter, center_valid
    
    if not video_path or not os.path.exists(video_path):
        print("[WARN] No video loaded. Use 'Open Video' first.")
        return
    
    print("[INFO] Analyzing all frames to find optimal optical center...")
    ui_set_controls_enabled(widgets, False)
    root.update_idletasks()
    
    tmpcap = cv2.VideoCapture(video_path)
    if not tmpcap.isOpened():
        print(f"[ERR] Could not open for analysis: {video_path}")
        ui_set_controls_enabled(widgets, True)
        return
    
    W = int(tmpcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(tmpcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(tmpcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use current params for detection
    p = dict(params)
    # Start with current center or frame center
    test_cx = xCenter if center_valid and xCenter is not None else W // 2
    test_cy = yCenter if center_valid and yCenter is not None else H // 2
    
    # Collect all pair line segments (for ray intersection calculation)
    all_lines = []  # List of (x1, y1, x2, y2) pair endpoints
    
    idx = 0
    progress_marks = set(int(N * q / 10) for q in range(1, 10))
    
    while True:
        ret, frm = tmpcap.read()
        if not ret:
            break
        
        # Apply background subtraction and contrast enhancement for optimization
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        gray_bgsub = apply_background_subtraction(gray)
        gray_contrast = apply_contrast(gray_bgsub)
        
        blur = cv2.GaussianBlur(gray_contrast, (max(1, int(p["blur"])|1), max(1, int(p["blur"])|1)), 0)
        _, binary = cv2.threshold(blur, int(p["threshold"]), 255, cv2.THRESH_BINARY)
        
        blobs = alg_detect(binary, test_cx, test_cy, p)
        method = p.get("pair_method", "Hungarian")
        if method == "Greedy":
            pairs = alg_pair_scored(blobs, p, test_cx, test_cy, True)
        elif method == "Symmetric":
            pairs = alg_pair_scored_symmetric(blobs, p, test_cx, test_cy, True)
        else:
            pairs = alg_pair_scored_hungarian(blobs, p, test_cx, test_cy, True)
        
        # Collect pair endpoints
        for (pid, xi, yi, xj, yj, *_rest) in pairs:
            all_lines.append((xi, yi, xj, yj))
        
        if idx in progress_marks:
            pct = 100 * idx / max(1, N)
            print(f"[INFO] Processing: {pct:5.1f}% ({idx}/{N})")
        
        idx += 1
    
    tmpcap.release()
    
    if len(all_lines) < 2:
        print("[WARN] Not enough pairs found for optimization. Need at least 2 pairs.")
        ui_set_controls_enabled(widgets, True)
        return
    
    print(f"[INFO] Analyzing {len(all_lines)} pair lines...")
    
    # Use grid voting: create a grid and vote for best intersection region
    grid_size = 5  # 5px grid cells
    votes = {}
    max_votes = 0
    best_pos = (test_cx, test_cy)
    
    # For each pair of lines, find intersection and vote
    for i in range(len(all_lines)):
        x1a, y1a, x2a, y2a = all_lines[i]
        # Line A equation: through (x1a,y1a) and (x2a,y2a)
        dxa = x2a - x1a
        dya = y2a - y1a
        if abs(dxa) < 1e-6 and abs(dya) < 1e-6:
            continue
        # Normalize to unit vector
        la = math.sqrt(dxa*dxa + dya*dya)
        uxa, uya = dxa/la, dya/la
        
        for j in range(i+1, len(all_lines)):
            x1b, y1b, x2b, y2b = all_lines[j]
            dxb = x2b - x1b
            dyb = y2b - y1b
            if abs(dxb) < 1e-6 and abs(dyb) < 1e-6:
                continue
            lb = math.sqrt(dxb*dxb + dyb*dyb)
            uxb, uyb = dxb/lb, dyb/lb
            
            # Find intersection of two lines (infinite lines through endpoints)
            # Line A: (x1a, y1a) + t * (uxa, uya)
            # Line B: (x1b, y1b) + s * (uxb, uyb)
            # Solve for intersection
            
            denom = uxa * uyb - uya * uxb
            if abs(denom) < 1e-6:  # Parallel lines
                continue
            
            dx = x1b - x1a
            dy = y1b - y1a
            t = (dx * uyb - dy * uxb) / denom
            # s = (dx * uya - dy * uxa) / denom
            
            ix = x1a + t * uxa
            iy = y1a + t * uya
            
            # Only consider intersections within frame bounds (with some margin)
            if 0 <= ix < W and 0 <= iy < H:
                # Vote in grid cell
                gx = int(ix / grid_size)
                gy = int(iy / grid_size)
                key = (gx, gy)
                votes[key] = votes.get(key, 0) + 1
                
                if votes[key] > max_votes:
                    max_votes = votes[key]
                    # Use center of grid cell
                    best_pos = (gx * grid_size + grid_size // 2, gy * grid_size + grid_size // 2)
    
    if max_votes < 2:
        print("[WARN] Could not find clear intersection point. Using frame center.")
        new_cx, new_cy = W // 2, H // 2
    else:
        new_cx, new_cy = best_pos
        print(f"[INFO] Found optimal center at ({new_cx}, {new_cy}) with {max_votes} votes")
    
    set_centerxy(new_cx, new_cy)
    
    ui_set_controls_enabled(widgets, True)
    print("[INFO] Optical center optimized. Preview will update automatically.")


def export_video():
    """Resave entire video with overlays and UTF-8 CSV metadata (fast, non-real-time)."""
    if not video_path or not os.path.exists(video_path):
        print("[WARN] No video loaded. Use 'Open Video' first.")
        return

    # Disable controls during export to avoid races
    ui_set_controls_enabled(widgets, False)
    root.update_idletasks()

    paths = export_paths_for(video_path)
    out_dir = paths["dir"]
    out_color = paths["tracked_mp4"]
    out_bin   = paths["binary_mp4"]
    out_csv   = paths["pairs_csv"]

    tmpcap = cv2.VideoCapture(video_path)
    if not tmpcap.isOpened():
        print(f"[ERR] Could not open for export: {video_path}")
        ui_set_controls_enabled(widgets, True)
        return
    W = int(tmpcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(tmpcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(tmpcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_guess = tmpcap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_color = cv2.VideoWriter(out_color, fourcc, max(1.0, fps_guess), (W, H))
    w_bin   = cv2.VideoWriter(out_bin,   fourcc, max(1.0, fps_guess), (W, H))

    print(f"[INFO] Export started →\n  DIR: {out_dir}\n  {out_color}\n  {out_bin}\n  CSV: {out_csv}")

    # Snapshot of params/overlays for deterministic export
    p  = dict(params)
    ov = dict(overlays)

    # Default center if not set
    global center_valid, xCenter, yCenter
    if not center_valid or xCenter is None or yCenter is None:
        xCenter, yCenter, center_valid = W // 2, H // 2, True
        print(f"[INFO] Default center for export -> ({xCenter},{yCenter})")

    rows = []
    # progress checkpoints (~5% steps)
    progress_marks = set(int(N * q / 20) for q in range(1, 20))

    # Initialize tracker and stats for export
    tracker = PairTracker(
        max_match_dist_px=float(p.get("track_max_match_dist", 25.0)),
        max_misses=int(p.get("track_max_misses", 10))
    )
    export_total_pairs_count = 0

    idx = 0
    while True:
        ret, frm = tmpcap.read()
        if not ret:
            break

        # Apply background subtraction and contrast enhancement
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        gray_bgsub = apply_background_subtraction(gray)
        gray_contrast = apply_contrast(gray_bgsub)
        
        # Use contrast-enhanced bg-subtracted image for detection and display
        blur = cv2.GaussianBlur(gray_contrast, (max(1, int(p["blur"])|1), max(1, int(p["blur"])|1)), 0)  # ensure odd ksize
        _, binary = cv2.threshold(blur, int(p["threshold"]), 255, cv2.THRESH_BINARY)

        # Detection + pairing
        blobs = alg_detect(binary, xCenter, yCenter, p)
        method = p.get("pair_method", "Hungarian")
        if method == "Greedy":
            pairs_before_tracking = alg_pair_scored(blobs, p, xCenter, yCenter, center_valid)
        elif method == "Symmetric":
            pairs_before_tracking = alg_pair_scored_symmetric(blobs, p, xCenter, yCenter, center_valid)
        else:
            pairs_before_tracking = alg_pair_scored_hungarian(blobs, p, xCenter, yCenter, center_valid)
        # Stable IDs for export
        pairs = tracker.update(pairs_before_tracking)
        
        # Update total pairs count for export
        export_total_pairs_count += len(pairs_before_tracking)

        # Prepare two overlay frames - use contrast-enhanced bg-subtracted image
        color = cv2.cvtColor(gray_contrast, cv2.COLOR_GRAY2BGR)
        bin3  = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Apply overlays
        if overlays["show_blobs"]:
            if overlay_targets["enable_tracked"]:
                draw_blob_boxes_ext(color, blobs)
            if overlay_targets["enable_binary"]:
                draw_blob_boxes_ext(bin3, blobs)

        if overlays["show_center"]:
            if overlay_targets["enable_tracked"]:
                draw_center_ext(color, xCenter, yCenter)
            if overlay_targets["enable_binary"]:
                draw_center_ext(bin3, xCenter, yCenter)

        if overlays["show_pair_center"]:
            if overlay_targets["enable_tracked"]:
                draw_pair_centers_ext(color, pairs, overlays.get("label_mode", "Red/Blue"), video_path)
            if overlay_targets["enable_binary"]:
                draw_pair_centers_ext(bin3, pairs, overlays.get("label_mode", "Red/Blue"), video_path)

        if overlays["show_lines"]:
            if overlay_targets["enable_tracked"]:
                show_labels = overlays.get("show_text_labels", 1)
                show_points = overlays.get("show_pair_points", 1)
                draw_pair_lines_ext(color, pairs, show_labels, overlays.get("label_mode", "Red/Blue"), video_path, show_points)
            if overlay_targets["enable_binary"]:
                show_labels = overlays.get("show_text_labels", 1)
                show_points = overlays.get("show_pair_points", 1)
                draw_pair_lines_ext(bin3, pairs, show_labels, overlays.get("label_mode", "Red/Blue"), video_path, show_points)

        if overlays["show_rays"]:
            if overlay_targets["enable_tracked"]:
                draw_pair_rays_toward_center_ext(color, pairs, color.shape[1], xCenter, yCenter, overlays.get("label_mode", "Red/Blue"), video_path)
            if overlay_targets["enable_binary"]:
                draw_pair_rays_toward_center_ext(bin3, pairs, bin3.shape[1], xCenter, yCenter, overlays.get("label_mode", "Red/Blue"), video_path)

        # Z value overlay - show if enabled and working distance is available
        # The draw function will show "Z:??" if magic constants are not available
        if overlays.get("show_z_value", 0) and calibration_working_distance_mm and calibration_working_distance_mm > 0:
            if overlay_targets["enable_tracked"]:
                draw_z_values_ext(color, pairs, 
                                calibration_working_distance_mm,
                                calibration_magic_constant,
                                calibration_magic_offset,
                                overlays.get("label_mode", "Red/Blue"),
                                video_path)
            if overlay_targets["enable_binary"]:
                draw_z_values_ext(bin3, pairs,
                                calibration_working_distance_mm,
                                calibration_magic_constant,
                                calibration_magic_offset,
                                overlays.get("label_mode", "Red/Blue"),
                                video_path)

        # X, Y values overlay - show if enabled and required parameters are available
        if overlays.get("show_xy_values", 0):
            pixels_per_mm = get_pixels_per_mm()
            if calibration_working_distance_mm and calibration_working_distance_mm > 0 and pixels_per_mm and pixels_per_mm > 0:
                if overlay_targets["enable_tracked"]:
                    draw_xy_values_ext(color, pairs,
                                      xCenter, yCenter,
                                      calibration_working_distance_mm,
                                      pixels_per_mm,
                                      overlays.get("label_mode", "Red/Blue"),
                                      video_path)
                if overlay_targets["enable_binary"]:
                    draw_xy_values_ext(bin3, pairs,
                                      xCenter, yCenter,
                                      calibration_working_distance_mm,
                                      pixels_per_mm,
                                      overlays.get("label_mode", "Red/Blue"),
                                      video_path)

        # Real point overlay (B point) - show if enabled and working distance is available
        # B can be calculated from A and C radii, working distance check is kept for consistency
        if overlays.get("show_real_point", 0) and calibration_working_distance_mm and calibration_working_distance_mm > 0:
                if overlay_targets["enable_tracked"]:
                    draw_real_point_ext(color, pairs,
                                      xCenter, yCenter,
                                      calibration_working_distance_mm,
                                      calibration_pixels_per_mm,
                                      overlays.get("label_mode", "Red/Blue"),
                                      video_path)
                if overlay_targets["enable_binary"]:
                    draw_real_point_ext(bin3, pairs,
                                      xCenter, yCenter,
                                      calibration_working_distance_mm,
                                      calibration_pixels_per_mm,
                                      overlays.get("label_mode", "Red/Blue"),
                                      video_path)

        # Stats overlay
        if overlays.get("show_current_stats", 0) or overlays.get("show_total_stats", 0):
            export_total_tracks = tracker.next_id - 1
            if overlay_targets["enable_tracked"]:
                draw_stats_overlay_ext(
                    color, 
                    pairs_before_tracking, 
                    pairs, 
                    export_total_pairs_count, 
                    export_total_tracks,
                    bool(overlays.get("show_current_stats", 0)),
                    bool(overlays.get("show_total_stats", 0))
                )
            if overlay_targets["enable_binary"]:
                draw_stats_overlay_ext(
                    bin3, 
                    pairs_before_tracking, 
                    pairs, 
                    export_total_pairs_count, 
                    export_total_tracks,
                    bool(overlays.get("show_current_stats", 0)),
                    bool(overlays.get("show_total_stats", 0))
                )

        # Write frames
        w_color.write(color)
        w_bin.write(bin3)

        # Calculate Pair Count for this frame (unique track IDs)
        pair_count = len(set(tid for tid, *_ in pairs))
        
        # Log CSV rows (pid is stable track id)
        for pair in pairs:
            # Unpack pair (now includes areas)
            pid, xi, yi, xj, yj, th_i, r_i, th_j, r_j, score = pair[0:10]
            
            # Calculate Zprime, B, and Z if calibration is loaded
            zprime_val = None
            b_val = None
            z_val = None
            
            if calibration_loaded and calibration_working_distance_mm and calibration_working_distance_mm > 0:
                # r_i is Radius_A (inner), r_j is Radius_B (outer, which is C)
                r_a = r_i  # A = inner radius
                r_c = r_j  # C = outer radius
                
                if r_a > 0 and r_c > 0:
                    # Calculate Zprime = working_distance * (C-A)/(A+C)
                    zprime_val = calibration_working_distance_mm * (r_c - r_a) / (r_a + r_c)
                    
                    # Calculate B = (2*A*C)/(A+C)
                    b_val = (2 * r_a * r_c) / (r_a + r_c)
                    
                    # Calculate Z = Zprime * magic_constant + magic_offset
                    # Note: Z requires calibration JSON with magic_constant/offset
                    if calibration_magic_constant is not None and calibration_magic_offset is not None:
                        z_val = zprime_val * calibration_magic_constant + calibration_magic_offset
            
            # Build row - always include Z, B, X, Y columns (empty if constants not present)
            # Calculate Z and B if calibration constants are available
            z_mm_val = ""
            b_px_val = ""
            x_mm_val = ""
            y_mm_val = ""
            
            if calibration_loaded and calibration_magic_constant is not None and calibration_magic_offset is not None:
                # Z can only be calculated if magic_constant and magic_offset are available
                if z_val is not None:
                    z_mm_val = round(z_val, 4)
            
            if calibration_loaded and calibration_working_distance_mm and calibration_working_distance_mm > 0:
                # B can be calculated if working distance is available
                if b_val is not None:
                    b_px_val = round(b_val, 4)
                    
                    # Calculate X, Y from B and theta (polar to Cartesian conversion)
                    # Get pixels_per_mm for conversion
                    pixels_per_mm = get_pixels_per_mm()
                    if pixels_per_mm and pixels_per_mm > 0:
                        # Calculate B in mm
                        b_mm = b_px_val / pixels_per_mm
                        
                        # Calculate angle to pair midpoint (average of the two angles, or from midpoint coordinates)
                        # Use the midpoint angle from the pair
                        mx = 0.5 * (xi + xj)
                        my = 0.5 * (yi + yj)
                        dx = mx - xCenter
                        dy = my - yCenter
                        dist_to_midpoint = math.sqrt(dx*dx + dy*dy)
                        
                        if dist_to_midpoint > 1e-6:
                            # Calculate angle in radians (atan2 gives angle from positive x-axis)
                            # Note: In image coordinates, Y increases downward, so we negate Y
                            theta_rad = math.atan2(dy, dx)
                            
                            # Convert polar to Cartesian: X = r * cos(theta), Y = -r * sin(theta)
                            # Y is negated because image coordinates have Y increasing downward
                            x_mm_val = round(b_mm * math.cos(theta_rad), 4)
                            y_mm_val = round(-b_mm * math.sin(theta_rad), 4)
            
            row = [
                idx, pid, xi, yi, xj, yj, th_i, r_i, th_j, r_j, round(float(score), 4),
                pair_count,  # Pair Count column
                int(p["threshold"]), int(p["blur"]), int(p["minArea"]), int(p["maxArea"]), int(p["maxW"]),
                float(p["maxRadGap"]), float(p["maxDMR"]), float(p["maxCenterOff"]),
                float(p["w_theta"]), float(p["w_area"]), float(p["w_center"]), float(p["Smin"]),
                z_mm_val,  # Z_mm column (empty if constants not present)
                b_px_val,  # B_px column (empty if constants not present)
                x_mm_val,  # X_mm column (empty if pixels_per_mm not available)
                y_mm_val   # Y_mm column (empty if pixels_per_mm not available)
            ]
            
            rows.append(row)

        if idx in progress_marks:
            pct = 100 * idx / max(1, N)
            print(f"[INFO] {pct:5.1f}% ({idx}/{N})")

        idx += 1

    tmpcap.release()
    w_color.release()
    w_bin.release()

    # Save CSV (UTF-8)
    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # Build header - always include Z, B, X, Y columns (empty if constants not present)
            header = [
                "Frame_Number", "Track_ID", "Center_X", "Center_Y", "Right_X", "Right_Y",
                "Angle_A_deg", "A_px", "Angle_B_deg", "C_px", "Pair_Score",
                "Pair_Count",
                "Binary_Threshold", "Blur_Size", "Min_Area_px2", "Max_Area_px2", "Max_Width_px",
                "Max_Radial_Gap_px", "Max_Angle_Diff_deg", "Max_Center_Offset_px",
                "Weight_Angle", "Weight_Area", "Weight_Center", "Min_Score_Threshold",
                "Z_mm", "B_px", "X_mm", "Y_mm"
            ]
            
            w.writerow(header)
            w.writerows(rows)
        
        print(f"[INFO] Export complete. CSV → {out_csv}")
        if calibration_loaded:
            print(f"[INFO] Calibration used: magic_constant={calibration_magic_constant:.6f}, "
                  f"magic_offset={calibration_magic_offset:.6f} mm, "
                  f"working_distance={calibration_working_distance_mm:.2f} mm")
            print(f"[INFO] Z and B values calculated and included in CSV.")
        else:
            print(f"[INFO] Note: Z and B columns are empty (no calibration constants loaded).")
    except Exception as e:
        print(f"[WARN] CSV save failed: {e}")

    # Re-enable controls after export
    ui_set_controls_enabled(widgets, True)

def get_latest_image_calibration_file() -> Optional[str]:
    """
    Find the latest image calibration JSON file in the calibrations folder.
    Returns the path to the latest file, or None if no file is found.
    """
    calibrations_dir = Path("calibrations")
    
    if not calibrations_dir.exists():
        return None
    
    # Find image calibration files
    json_files = list(calibrations_dir.glob("image_calibration_*.json"))
    
    if not json_files:
        return None
    
    # Sort by modification time (newest first)
    json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    return str(json_files[0])


def get_pixels_per_mm() -> Optional[float]:
    """
    Get pixels_per_mm from calibration data.
    First checks video calibration, then falls back to latest image calibration.
    """
    global calibration_pixels_per_mm
    
    # First check if it's already loaded from video calibration
    if calibration_pixels_per_mm is not None and calibration_pixels_per_mm > 0:
        return calibration_pixels_per_mm
    
    # Try to load from latest image calibration file
    latest_image_cal = get_latest_image_calibration_file()
    if latest_image_cal:
        try:
            with open(latest_image_cal, 'r', encoding='utf-8') as f:
                cal_data = json.load(f)
                if "pixels_per_mm" in cal_data:
                    pixels_per_mm = float(cal_data["pixels_per_mm"])
                    if pixels_per_mm > 0:
                        return pixels_per_mm
        except Exception:
            pass
    
    return None


def get_latest_video_calibration_file() -> Optional[str]:
    """
    Find the latest video calibration JSON file in the calibrations folder.
    Returns the path to the latest file, or None if no file is found.
    """
    calibrations_dir = Path("calibrations")
    
    if not calibrations_dir.exists():
        return None
    
    # Find all video calibration JSON files (those with magic_constant and magic_offset)
    json_files = list(calibrations_dir.glob("video_calibration_*.json"))
    
    if not json_files:
        # Fallback: check any JSON file that might contain video calibration data
        json_files = list(calibrations_dir.glob("*.json"))
        # Filter for files that have video calibration structure
        valid_files = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    cal_data = json.load(f)
                    if "magic_constant" in cal_data and "magic_offset" in cal_data:
                        valid_files.append(json_file)
            except Exception:
                continue
        json_files = valid_files
    
    if not json_files:
        return None
    
    # Sort by modification time (newest first)
    json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    return str(json_files[0])


def load_calibration_file(file_path: str, silent: bool = False) -> bool:
    """
    Load calibration data from a JSON file.
    
    Args:
        file_path: Path to the calibration JSON file
        silent: If True, don't show messageboxes (only print to console)
    
    Returns:
        True if calibration was successfully loaded, False otherwise
    """
    global calibration_magic_constant, calibration_magic_offset, calibration_working_distance_mm, calibration_pixels_per_mm, calibration_loaded
    
    if not file_path or not os.path.exists(file_path):
        if not silent:
            messagebox.showerror("Error", f"File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            cal_data = json.load(f)
        
        if "magic_constant" in cal_data and "magic_offset" in cal_data:
            calibration_magic_constant = float(cal_data["magic_constant"])
            calibration_magic_offset = float(cal_data["magic_offset"])
            calibration_loaded = True
            
            # Try to get working distance from calibration file (top level first, then data points)
            if "working_distance_mm" in cal_data:
                calibration_working_distance_mm = float(cal_data["working_distance_mm"])
            elif "data_points" in cal_data and len(cal_data["data_points"]) > 0:
                # Use working distance from first data point (assuming same for all)
                calibration_working_distance_mm = float(cal_data["data_points"][0].get("working_distance_mm", 0))
            
            # Try to get pixels_per_mm from calibration file
            if "pixels_per_mm" in cal_data:
                calibration_pixels_per_mm = float(cal_data["pixels_per_mm"])
            else:
                calibration_pixels_per_mm = None
            
            if calibration_working_distance_mm is None or calibration_working_distance_mm <= 0:
                # Working distance not found
                if not silent:
                    print("[WARN] Working distance not found in calibration file.")
                    print(f"[INFO] Loaded calibration: magic_constant={calibration_magic_constant:.6f}, magic_offset={calibration_magic_offset:.6f}")
                    messagebox.showinfo(
                        "Calibration Loaded",
                        f"Magic Constant: {calibration_magic_constant:.6f}\n"
                        f"Magic Offset: {calibration_magic_offset:.6f} mm\n\n"
                        "Note: Working distance not found. Please ensure it's set for Z/B calculation."
                    )
                return True
            
            print(f"[INFO] Calibration loaded: magic_constant={calibration_magic_constant:.6f}, "
                  f"magic_offset={calibration_magic_offset:.6f} mm, "
                  f"working_distance={calibration_working_distance_mm:.2f} mm")
            if not silent:
                messagebox.showinfo(
                    "Calibration Loaded",
                    f"Magic Constant: {calibration_magic_constant:.6f}\n"
                    f"Magic Offset: {calibration_magic_offset:.6f} mm\n"
                    f"Working Distance: {calibration_working_distance_mm:.2f} mm\n\n"
                    "Z and B will be calculated for exported pairs."
                )
            return True
        else:
            if not silent:
                messagebox.showerror("Error", "Invalid calibration file: missing magic_constant or magic_offset")
            return False
    except Exception as e:
        if not silent:
            messagebox.showerror("Error", f"Failed to load calibration file: {e}")
        print(f"[ERROR] Failed to load calibration: {e}")
        return False


def load_calibration():
    """Load calibration data from JSON file for Z/B calculation (with file dialog)."""
    file_path = filedialog.askopenfilename(
        title="Load Video Calibration File",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    
    if file_path:
        load_calibration_file(file_path, silent=False)

def handle_reset():
    ui_reset_defaults_ui(
        params,
        overlays,
        DEFAULT_PARAMS,
        DEFAULT_OVERLAYS,
        gui_vars_numeric,
        gui_vars_check,
        widgets,
    )

def on_exit():
    """Graceful exit: save preset, close OpenCV windows, stop Tk mainloop."""
    try:
        save_preset()
    except:
        pass
    try:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    except:
        pass
    root.quit()

    # GUI moved to ui.py

# --------------------------------------------------------------------------------------
# Live preview loop
# --------------------------------------------------------------------------------------
def on_trackbar_changed(value):
    """Callback for video position trackbar in Tkinter window."""
    global cap, trackbar_being_set, cached_frame
    if trackbar_being_set:
        return
    try:
        if cap is not None and cap.isOpened():
            frame_pos = int(float(value))
            if frame_pos >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                # Update cached frame when seeking (so pause shows correct frame)
                ret, cached_frame = cap.read()
                if not ret:
                    cached_frame = None
                else:
                    # Reset position back since read() advanced it
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    except (TypeError, ValueError, AttributeError):
        pass
    except Exception:
        pass

def toggle_play_pause():
    """Toggle play/pause state for preview."""
    global preview_playing, cached_frame
    preview_playing = not preview_playing
    if not preview_playing:
        # When pausing, cache the current frame
        global cap
        if cap is not None and cap.isOpened():
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, cached_frame = cap.read()
            if not ret:
                cached_frame = None
    if video_controls_widgets and "btn_play_pause" in video_controls_widgets:
        video_controls_widgets["btn_play_pause"].config(text="▶ Play" if not preview_playing else "⏸ Pause")
    print(f"[INFO] Preview: {'Playing' if preview_playing else 'Paused'}")

def setup_video_controls_window():
    """Create or update the separate video controls window with trackbar and pause button."""
    global video_controls_window, video_controls_widgets, video_total_frames, cap
    
    if cap is None or not cap.isOpened():
        return
    
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_total_frames <= 0:
        return
    
    # Create window if it doesn't exist
    if video_controls_window is None or not video_controls_window.winfo_exists():
        video_controls_window = tk.Toplevel()
        video_controls_window.title("Video Controls")
        video_controls_window.geometry("500x100+60+200")
        video_controls_window.resizable(False, False)
        
        # Main frame
        main_frame = ttk.Frame(video_controls_window, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Trackbar frame
        trackbar_frame = ttk.Frame(main_frame)
        trackbar_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(trackbar_frame, text="Frame:").pack(side="left", padx=(0, 5))
        
        # Trackbar variable
        trackbar_var = tk.IntVar(value=0)
        video_controls_widgets["trackbar_var"] = trackbar_var
        
        # Create trackbar (scale widget)
        trackbar = ttk.Scale(
            trackbar_frame,
            from_=0,
            to=max(0, video_total_frames - 1),
            orient="horizontal",
            variable=trackbar_var,
            command=on_trackbar_changed,
            length=400
        )
        trackbar.pack(side="left", fill="x", expand=True)
        video_controls_widgets["trackbar"] = trackbar
        
        # Frame number label
        frame_label = ttk.Label(trackbar_frame, text="0 / 0", width=12)
        frame_label.pack(side="left", padx=(5, 0))
        video_controls_widgets["frame_label"] = frame_label
        
        # Update frame label when trackbar changes
        def update_frame_label(*args):
            current = trackbar_var.get()
            video_controls_widgets["frame_label"].config(text=f"{current} / {video_total_frames - 1}")
        trackbar_var.trace_add("write", update_frame_label)
        update_frame_label()
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x")
        
        # Play/Pause button
        play_pause_btn = ttk.Button(
            button_frame,
            text="⏸ Pause" if preview_playing else "▶ Play",
            command=toggle_play_pause,
            width=15
        )
        play_pause_btn.pack(side="left", padx=(0, 10))
        video_controls_widgets["btn_play_pause"] = play_pause_btn
    else:
        # Update existing window
        if "trackbar" in video_controls_widgets:
            video_controls_widgets["trackbar"].config(to=max(0, video_total_frames - 1))
        if "trackbar_var" in video_controls_widgets:
            video_controls_widgets["trackbar_var"].set(0)

def on_mouse_tracked(event, x, y, flags, userdata):
    """Mouse callback for the 'Tracked' window — click to set optical center."""
    if event == cv2.EVENT_LBUTTONDOWN:
        set_centerxy(x, y)

def preview_loop():
    """
    Tkinter-friendly preview loop:
      - Reads one frame
      - Applies blur + threshold
      - Detects + pairs
      - Renders overlays to 'Tracked' and 'Binary'
      - Schedules itself again after DELAY_MS
    """
    global cap, root, center_valid, xCenter, yCenter, last_pair_method, tracker_preview, total_pairs_count, max_pair_count, params, overlays, overlay_targets, video_path, preview_playing, video_total_frames, trackbar_being_set, showing_center_setup
    try:
        if cap is None or not cap.isOpened():
            # No video yet — reschedule soon
            root.after(DELAY_MS, preview_loop)
            return

        # If showing center setup, wait for user to click or press Enter
        global cached_frame
        if showing_center_setup and not center_valid:
            # Still waiting for center point - show raw first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, first_frame = cap.read()
            if ret:
                first_frame_display = first_frame.copy()
                cv2.putText(first_frame_display, "Click to set optical center (Press Enter to confirm)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if xCenter is not None and yCenter is not None:
                    # Show temporary crosshair at current/default position
                    cv2.circle(first_frame_display, (xCenter, yCenter), 10, (0, 255, 255), 2)
                    cv2.line(first_frame_display, (xCenter - 15, yCenter), (xCenter + 15, yCenter), (0, 255, 255), 2)
                    cv2.line(first_frame_display, (xCenter, yCenter - 15), (xCenter, yCenter + 15), (0, 255, 255), 2)
                cv2.imshow("Tracked", first_frame_display)
                # Show blank binary window
                h, w = first_frame.shape[:2]
                blank = np.zeros((h, w), dtype=np.uint8)
                cv2.imshow("Binary", blank)
                
                # Ensure Tracked window is active and can receive keyboard input
                # waitKey() only responds to keys when an OpenCV window has focus
                # This ensures Enter key detection only works when Tracked window is focused
                try:
                    # Check if Tracked window is visible
                    if cv2.getWindowProperty("Tracked", cv2.WND_PROP_VISIBLE) >= 1:
                        # Check for Enter key press (key code 13 or 10)
                        # Note: This only works when Tracked window has focus
                        key = cv2.waitKey(1) & 0xFF
                        if key == 13 or key == 10:  # Enter key
                            # Confirm current center location
                            if xCenter is not None and yCenter is not None:
                                set_centerxy(xCenter, yCenter)
                except cv2.error:
                    # Window might not exist yet, continue
                    pass
            root.after(DELAY_MS, preview_loop)
            return
        
        # Only advance frame if playing
        if preview_playing:
            ret, frm = cap.read()
            if not ret:
                # Loop to the start for continuous preview
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Reset stats when looping
                if tracker_preview is not None:
                    tracker_preview = PairTracker(
                        max_match_dist_px=float(params.get("track_max_match_dist", 25.0)),
                        max_misses=int(params.get("track_max_misses", 10))
                    )
                    total_pairs_count = 0
                    max_pair_count = 0
                ret, frm = cap.read()
                if not ret:
                    root.after(DELAY_MS, preview_loop)
                    return
            # Cache the frame for pause
            cached_frame = frm.copy()
        else:
            # Paused: use cached frame, don't advance
            if cached_frame is not None:
                frm = cached_frame
                ret = True
            else:
                # No cached frame yet, read current frame once
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frm = cap.read()
                if ret:
                    cached_frame = frm.copy()
                else:
                    root.after(DELAY_MS, preview_loop)
                    return
        
        # Update trackbar position in Tkinter window (only if playing, or initial update)
        if video_total_frames > 0 and video_controls_widgets:
            try:
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if "trackbar_var" in video_controls_widgets:
                    trackbar_var = video_controls_widgets["trackbar_var"]
                    # Update if playing, or if trackbar is out of sync (user might have dragged it)
                    if preview_playing or trackbar_var.get() != current_frame:
                        if trackbar_var.get() != current_frame:
                            trackbar_being_set = True
                            trackbar_var.set(current_frame)
                            trackbar_being_set = False
            except:
                pass

        # Apply background subtraction and contrast enhancement
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        gray_bgsub = apply_background_subtraction(gray)
        gray_contrast = apply_contrast(gray_bgsub)
        
        # Use contrast-enhanced bg-subtracted image for display and detection
        ksize = max(1, int(params["blur"]))
        if ksize % 2 == 0: ksize += 1
        blur = cv2.GaussianBlur(gray_contrast, (ksize, ksize), 0)
        _, binary = cv2.threshold(blur, int(params["threshold"]), 255, cv2.THRESH_BINARY)

        # Default center if user hasn't clicked yet
        if not center_valid or xCenter is None or yCenter is None:
            h, w = gray.shape[:2]
            set_centerxy(w // 2, h // 2)

        blobs = alg_detect(binary, xCenter, yCenter, params)
        method = params.get("pair_method", "Hungarian")
        # Show transient overlay when method changes
        if method != last_pair_method:
            try:
                cv2.displayOverlay("Tracked", f"Pairing method → {method}", 1500)
            except:
                pass
            last_pair_method = method
        if method == "Greedy":
            pairs_before_tracking = alg_pair_scored(blobs, params, xCenter, yCenter, center_valid)
        elif method == "Symmetric":
            pairs_before_tracking = alg_pair_scored_symmetric(blobs, params, xCenter, yCenter, center_valid)
        else:
            pairs_before_tracking = alg_pair_scored_hungarian(blobs, params, xCenter, yCenter, center_valid)
        # Stable IDs across frames (preview)
        if tracker_preview is None:
            tracker_preview = PairTracker(
                max_match_dist_px=float(params.get("track_max_match_dist", 25.0)),
                max_misses=int(params.get("track_max_misses", 10))
            )
            total_pairs_count = 0
        # Update tracker params if they changed
        tracker_preview.max_match_dist_px = float(params.get("track_max_match_dist", 25.0))
        tracker_preview.max_misses = int(params.get("track_max_misses", 10))
        pairs = tracker_preview.update(pairs_before_tracking)
        
        # Update total pairs count
        total_pairs_count += len(pairs_before_tracking)
        
        # Update max pair count (incrementing - tracks maximum seen in any single frame)
        current_pair_count = len(set(tid for tid, *_ in pairs))
        if current_pair_count > max_pair_count:
            max_pair_count = current_pair_count

        # Display contrast-enhanced, bg-subtracted grayscale in Tracked window
        color = cv2.cvtColor(gray_contrast, cv2.COLOR_GRAY2BGR)
        bin3  = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        if overlays["show_blobs"]:
            if overlay_targets["enable_tracked"]:
                draw_blob_boxes_ext(color, blobs)
            if overlay_targets["enable_binary"]:
                draw_blob_boxes_ext(bin3, blobs)

        if overlays["show_center"]:
            if overlay_targets["enable_tracked"]:
                draw_center_ext(color, xCenter, yCenter)
            if overlay_targets["enable_binary"]:
                draw_center_ext(bin3, xCenter, yCenter)

        if overlays["show_pair_center"]:
            if overlay_targets["enable_tracked"]:
                draw_pair_centers_ext(color, pairs, overlays.get("label_mode", "Red/Blue"), video_path)
            if overlay_targets["enable_binary"]:
                draw_pair_centers_ext(bin3, pairs, overlays.get("label_mode", "Red/Blue"), video_path)

        if overlays["show_lines"]:
            if overlay_targets["enable_tracked"]:
                show_labels = overlays.get("show_text_labels", 1)
                show_points = overlays.get("show_pair_points", 1)
                draw_pair_lines_ext(color, pairs, show_labels, overlays.get("label_mode", "Red/Blue"), video_path, show_points)
            if overlay_targets["enable_binary"]:
                show_labels = overlays.get("show_text_labels", 1)
                show_points = overlays.get("show_pair_points", 1)
                draw_pair_lines_ext(bin3, pairs, show_labels, overlays.get("label_mode", "Red/Blue"), video_path, show_points)

        if overlays["show_rays"]:
            if overlay_targets["enable_tracked"]:
                draw_pair_rays_toward_center_ext(color, pairs, color.shape[1], xCenter, yCenter, overlays.get("label_mode", "Red/Blue"), video_path)
            if overlay_targets["enable_binary"]:
                draw_pair_rays_toward_center_ext(bin3, pairs, bin3.shape[1], xCenter, yCenter, overlays.get("label_mode", "Red/Blue"), video_path)

        # Z value overlay - show if enabled and working distance is available
        # The draw function will show "Z:??" if magic constants are not available
        if overlays.get("show_z_value", 0) and calibration_working_distance_mm and calibration_working_distance_mm > 0:
            if overlay_targets["enable_tracked"]:
                draw_z_values_ext(color, pairs, 
                                calibration_working_distance_mm,
                                calibration_magic_constant,
                                calibration_magic_offset,
                                overlays.get("label_mode", "Red/Blue"),
                                video_path)
            if overlay_targets["enable_binary"]:
                draw_z_values_ext(bin3, pairs,
                                calibration_working_distance_mm,
                                calibration_magic_constant,
                                calibration_magic_offset,
                                overlays.get("label_mode", "Red/Blue"),
                                video_path)

        # X, Y values overlay - show if enabled and required parameters are available
        if overlays.get("show_xy_values", 0):
            pixels_per_mm = get_pixels_per_mm()
            if calibration_working_distance_mm and calibration_working_distance_mm > 0 and pixels_per_mm and pixels_per_mm > 0:
                if overlay_targets["enable_tracked"]:
                    draw_xy_values_ext(color, pairs,
                                      xCenter, yCenter,
                                      calibration_working_distance_mm,
                                      pixels_per_mm,
                                      overlays.get("label_mode", "Red/Blue"),
                                      video_path)
                if overlay_targets["enable_binary"]:
                    draw_xy_values_ext(bin3, pairs,
                                      xCenter, yCenter,
                                      calibration_working_distance_mm,
                                      pixels_per_mm,
                                      overlays.get("label_mode", "Red/Blue"),
                                      video_path)

        # Real point overlay (B point) - show if enabled and working distance is available
        # B can be calculated from A and C radii, working distance check is kept for consistency
        if overlays.get("show_real_point", 0) and calibration_working_distance_mm and calibration_working_distance_mm > 0:
                if overlay_targets["enable_tracked"]:
                    draw_real_point_ext(color, pairs,
                                      xCenter, yCenter,
                                      calibration_working_distance_mm,
                                      calibration_pixels_per_mm,
                                      overlays.get("label_mode", "Red/Blue"),
                                      video_path)
                if overlay_targets["enable_binary"]:
                    draw_real_point_ext(bin3, pairs,
                                      xCenter, yCenter,
                                      calibration_working_distance_mm,
                                      calibration_pixels_per_mm,
                                      overlays.get("label_mode", "Red/Blue"),
                                      video_path)

        # Stats overlay
        if overlays.get("show_current_stats", 0) or overlays.get("show_total_stats", 0):
            total_tracks = tracker_preview.next_id - 1 if tracker_preview else 0
            if overlay_targets["enable_tracked"]:
                draw_stats_overlay_ext(
                    color, 
                    pairs_before_tracking, 
                    pairs, 
                    total_pairs_count, 
                    total_tracks,
                    bool(overlays.get("show_current_stats", 0)),
                    bool(overlays.get("show_total_stats", 0))
                )
            if overlay_targets["enable_binary"]:
                draw_stats_overlay_ext(
                    bin3, 
                    pairs_before_tracking, 
                    pairs, 
                    total_pairs_count, 
                    total_tracks,
                    bool(overlays.get("show_current_stats", 0)),
                    bool(overlays.get("show_total_stats", 0))
                )

        # Display
        cv2.imshow("Tracked", color)
        cv2.imshow("Binary",  bin3)
        cv2.setMouseCallback("Tracked", on_mouse_tracked)

        # Keep OpenCV responsive
        cv2.waitKey(1)

    except Exception as e:
        print(f"[WARN] Preview error: {e}")

    # Schedule next frame
    root.after(DELAY_MS, preview_loop)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    load_preset()
    global root, widgets, gui_vars_numeric, gui_vars_check
    root, widgets, gui_vars_numeric, gui_vars_check = ui_build_gui(
        params, overlays, overlay_targets,
        open_video, export_video, optimize_optical_center, handle_reset, on_exit, toggle_play_pause,
        load_calibration
    )

    # Create windows early (needed before reopen_video can resize them)
    cv2.namedWindow("Tracked", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow("Binary",  cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    
    # Auto-load latest video calibration file
    latest_cal_file = get_latest_video_calibration_file()
    if latest_cal_file:
        print(f"[INFO] Auto-loading latest video calibration: {latest_cal_file}")
        if load_calibration_file(latest_cal_file, silent=True):
            print(f"[INFO] Successfully auto-loaded calibration from: {os.path.basename(latest_cal_file)}")
        else:
            print(f"[WARN] Failed to auto-load calibration from: {latest_cal_file}")
    else:
        print("[INFO] No video calibration file found in calibrations folder.")
    
    if video_path and os.path.exists(video_path):
        # Set last_video_path when loading from preset so we can detect new videos later
        global last_video_path
        last_video_path = video_path
        reopen_video()  # This will resize windows to native video resolution
    else:
        print("[INFO] No video in preset — use 'Open Video' in the GUI.")
        # Default size if no video (both same size)
        cv2.resizeWindow("Tracked", 960, 720)
        cv2.resizeWindow("Binary", 960, 720)


    print("[INFO] Live preview ready. Use the GUI for controls. ESC closes preview windows; Exit button quits.")

    # Start periodic preview
    root.after(DELAY_MS, preview_loop)

    # Bind ESC in root to close gracefully too
    root.bind("<Escape>", lambda e: on_exit())

    # Enter Tk mainloop (OpenCV windows continue via after() scheduling)
    root.mainloop()

    # Cleanup on window close (if not already cleaned)
    try:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    except:
        pass

if __name__ == "__main__":
    main()
