#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watershed Blob Detector with Pairing and Tracking

Pipeline:
  1. Import video → Grayscale
  2. Background subtraction (visualize)
  3. Binary thresholding (Otsu's or adjustable) (visualize)
  4. EDT for peaks/markers (visualize with heatmap) - adjustable distances, threshold
  5. Detect contours in mask → centroid xy, area, polar coordinates
  6. Pair blobs using Greedy/Symmetric/Hungarian algorithms
  7. Track pairs across frames with stable IDs
  8. Tracked window with GUI labeling (click to set optical center)
  9. Optimize optical center based on ray intersections
  10. Video and CSV export (matches detect_pairs.py format)
"""

import cv2
import numpy as np
import csv
import json
import os
import math
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy import ndimage
try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed as skimage_watershed
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from lib.capture.util_paths import export_paths_for
from lib.pair.pair_tracker import PairTracker
from lib.pair.pair_algorithms import (
    pair_scored,
    pair_scored_symmetric,
    pair_scored_hungarian,
    polar_from_center,
    line_distance_to_point,
)
from lib.pair.pair_draw import (
    draw_blob_boxes,
    draw_center,
    draw_pair_lines,
    draw_pair_rays_toward_center,
    set_video_seed,
    draw_pair_centers,
    draw_stats_overlay,
    draw_z_values,
    draw_xy_values,
    draw_real_point,
)
from lib.pair.preset_io import save_preset_file, load_preset_file

# ============ Default parameters ============
DEFAULT_PARAMS = {
    # Background subtraction
    "bg_alpha": 0.95,
    "bg_static_thresh": 6,
    "bg_min_static_ratio": 0.8,
    
    # Binary thresholding
    "threshold": 70,
    "blur": 1,
    "use_otsu": 0,
    "invert_threshold": 0,
    
    # EDT / Watershed parameters
    "ws_min_distance": 20,
    "ws_marker_threshold": 0.7,  # Threshold for markers (0-1, fraction of max EDT)
    "ws_edt_power": 1.0,  # Power/gamma adjustment for EDT (1.0 = linear, <1.0 = emphasize small distances, >1.0 = emphasize large distances)
    
    # Blob filtering
    "minArea": 20,
    "maxArea": 200,
    "maxW": 100,
    
    # Pairing parameters
    "maxRadGap": 120,      # Max radial distance difference between pair blobs
    "maxDMR": 3,           # Max angle difference (degrees)
    "maxCenterOff": 15,    # Max distance from center line
    "w_theta": 0.35,       # Weight for angle similarity
    "w_area": 0.35,        # Weight for area similarity
    "w_center": 0.30,      # Weight for center alignment
    "Smin": 0.90,          # Minimum pair score
    "pair_method": "Hungarian",  # Greedy, Symmetric, or Hungarian
    
    # Tracking parameters
    "track_max_match_dist": 25.0,  # Max distance for matching pairs across frames
    "track_max_misses": 10,        # Frames to wait before retiring track
    
    # Visualization
    "show_debug_windows": 0,  # Master switch for all debug windows
}

PRESET_PATH = os.path.join("apps", "detect_pairs_watershed_default.json")

# Debug window size (consistent for all debug windows)
DEBUG_WINDOW_SIZE = (640, 480)

# ============ Default overlay toggles (checkboxes) ============
DEFAULT_OVERLAYS = {
    "show_blobs": 1,           # Draw green bounding boxes around detected blobs
    "show_center": 1,           # Draw yellow crosshair at the optical center
    "show_pair_center": 0,     # Draw small circle at pair midpoint (for tracking visualization)
    "show_lines": 1,           # Draw white line between paired blobs
    "show_rays": 1,            # Extend pair line (white) in the AC direction toward/past the center
    "show_pair_points": 1,     # Draw circles at pair endpoints (A and C points)
    "label_mode": "Red/Blue",  # Label mode: "None", "Red/Blue", "Random"
    "show_text_labels": 1,    # Show #A/#C text labels on pairs
    "show_z_value": 0,         # Show Z value text labels on pairs (requires calibration)
    "show_xy_values": 0,       # Show X and Y value text labels on pairs (requires calibration)
    "show_real_point": 0,     # Show real point at B mm distance from center along ray (requires calibration)
    "show_current_stats": 0,   # Show current frame stats overlay
    "show_total_stats": 0,     # Show total stats overlay
}

DEFAULT_OVERLAY_TARGETS = {
    "enable_tracked": 1,
    "enable_binary": 0,
}

# ============ Global state ============
params = DEFAULT_PARAMS.copy()
overlays = DEFAULT_OVERLAYS.copy()
overlay_targets = DEFAULT_OVERLAY_TARGETS.copy()
video_path = ""
cap = None
background_image = None
root = None
widgets = {}
gui_vars_numeric = {}
gui_vars_check = {}
tracker = None  # PairTracker instance
xCenter = None
yCenter = None
center_valid = False

# ============ Utility functions ============
def build_background_from_video(video_path: str) -> np.ndarray:
    """Build averaged background image from video."""
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
    
    print("[INFO] Building background model...")
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
            print(f"[INFO] Background: {pct:.1f}%")
        
        frame_idx += 1
    
    tmpcap.release()
    
    mask_sum[mask_sum == 0] = 1.0
    bg_final = bg_avg / mask_sum
    bg_final8 = cv2.convertScaleAbs(bg_final)
    
    print(f"[INFO] Background model complete: {count} frames contributed.")
    return bg_final8

def apply_background_subtraction(frame_gray: np.ndarray) -> np.ndarray:
    """Apply background subtraction."""
    global background_image
    if background_image is None:
        return frame_gray
    diff = cv2.absdiff(frame_gray, background_image)
    return cv2.convertScaleAbs(diff)

def detect_blobs_watershed(binary: np.ndarray, params: Dict, cx: Optional[int] = None, cy: Optional[int] = None) -> Tuple[List[Dict], Dict]:
    """
    Detect blobs using watershed segmentation.
    
    Args:
        binary: Binary image
        params: Detection parameters
        cx, cy: Optical center coordinates (for polar coordinates)
    
    Returns:
        - List of blobs: [dict(xc, yc, area, box, theta, r), ...]
        - Debug images dict for visualization
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image required. Install with: pip install scikit-image")
    
    minA = int(params.get("minArea", 4))
    maxA = int(params.get("maxArea", 5000))
    maxW = int(params.get("maxW", 100))
    ws_min_distance = int(params.get("ws_min_distance", 20))
    ws_marker_threshold = float(params.get("ws_marker_threshold", 0.7))
    ws_edt_power = float(params.get("ws_edt_power", 1.0))
    
    debug_images = {}
    
    # Step 1: Ensure binary
    if len(binary.shape) == 3:
        gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary.copy()
    
    # Ensure 0/255 binary
    thresh = (gray > 127).astype(np.uint8) * 255
    debug_images['binary'] = thresh.copy()
    
    # Step 2: Compute Euclidean Distance Transform
    D = ndimage.distance_transform_edt(thresh)
    
    # Step 2.5: Apply power/gamma adjustment to EDT (for sensitivity control)
    # Power < 1.0: Emphasizes small distances (better for small blobs, preserves detail)
    # Power = 1.0: Linear (default, no change)
    # Power > 1.0: Emphasizes large distances (helps separate overlapping objects)
    if ws_edt_power != 1.0:
        D_max = D.max()
        if D_max > 0:
            # Normalize, apply power, then scale back
            D_normalized = D / D_max
            D_powered = np.power(D_normalized, ws_edt_power)
            D = D_powered * D_max
    
    # Normalize EDT for visualization and apply heatmap colormap
    D_normalized = cv2.normalize(D, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    D_heatmap = cv2.applyColorMap(D_normalized, cv2.COLORMAP_JET)
    debug_images['edt'] = D_heatmap
    
    # Step 3: Find local maxima (peaks) in EDT
    peak_coords = peak_local_max(D, min_distance=ws_min_distance, labels=thresh)
    localMax = np.zeros(D.shape, dtype=bool)
    if peak_coords.size > 0:
        localMax[tuple(peak_coords.T)] = True
    
    local_max_vis = (localMax.astype(np.uint8) * 255)
    debug_images['peaks'] = local_max_vis
    
    # Step 4: Create markers from peaks (threshold EDT)
    marker_threshold = ws_marker_threshold * D.max()
    sure_fg = (D >= marker_threshold).astype(np.uint8) * 255
    markers, num_labels = ndimage.label(sure_fg, structure=np.ones((3, 3)))
    
    # Colored markers visualization
    markers_colored = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    unique_markers = np.unique(markers)
    num_objects = len([m for m in unique_markers if m > 0])
    for marker_id in unique_markers:
        if marker_id == 0:
            continue
        mask = (markers == marker_id)
        hue = int(((marker_id - 1) * 180 / max(1, num_objects)) % 180)
        markers_colored[mask] = [hue, 255, 255]
    markers_colored = cv2.cvtColor(markers_colored, cv2.COLOR_HSV2BGR)
    markers_colored[markers == 0] = [0, 0, 0]
    debug_images['markers'] = markers_colored
    
    # Step 5: Apply watershed
    labels = skimage_watershed(-D, markers, mask=thresh)
    
    # Colored labels visualization
    labels_colored = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    unique_labels = np.unique(labels)
    num_segments = len([l for l in unique_labels if l > 0])
    for label_id in unique_labels:
        if label_id == 0:
            continue
        mask = (labels == label_id)
        hue = int(((label_id - 1) * 180 / max(1, num_segments)) % 180)
        labels_colored[mask] = [hue, 255, 255]
    labels_colored = cv2.cvtColor(labels_colored, cv2.COLOR_HSV2BGR)
    labels_colored[labels == 0] = [0, 0, 0]
    debug_images['labels'] = labels_colored
    
    # Step 6: Extract blobs from labels
    blobs = []
    for label in np.unique(labels):
        if label == 0:
            continue
        
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        if area < minA or area > maxA:
            continue
        if w > maxW or h > maxW:
            continue
        
        # Centroid
        M = cv2.moments(c)
        if M["m00"] != 0:
            xc = int(M["m10"] / M["m00"])
            yc = int(M["m01"] / M["m00"])
        else:
            xc = x + w // 2
            yc = y + h // 2
        
        # Calculate polar coordinates from optical center
        if cx is not None and cy is not None:
            th, r = polar_from_center(xc, yc, cx, cy)
        else:
            th, r = 0.0, 0.0  # Default if center not set
        
        blobs.append(dict(xc=xc, yc=yc, area=area, box=(x, y, w, h), theta=th, r=r))
    
    return blobs, debug_images

# ============ Optical center functions ============
def set_centerxy(x, y):
    """Update optical center to (x,y) and mark as valid."""
    global xCenter, yCenter, center_valid
    xCenter, yCenter, center_valid = int(x), int(y), True
    print(f"[INFO] Optical center -> ({xCenter},{yCenter})")

def on_mouse_tracked(event, x, y, flags, userdata):
    """Mouse callback for the 'Tracked' window — click to set optical center."""
    if event == cv2.EVENT_LBUTTONDOWN:
        set_centerxy(x, y)

def optimize_optical_center():
    """Analyze all frames to find optimal optical center based on ray intersections."""
    global xCenter, yCenter, center_valid
    
    if not video_path or not os.path.exists(video_path):
        messagebox.showerror("Error", "No video loaded")
        return
    
    print("[INFO] Analyzing all frames to find optimal optical center...")
    if root:
        for w in widgets.values():
            try:
                w.configure(state="disabled")
            except:
                pass
        root.update_idletasks()
    
    tmpcap = cv2.VideoCapture(video_path)
    if not tmpcap.isOpened():
        messagebox.showerror("Error", "Could not open video")
        if root:
            for w in widgets.values():
                try:
                    w.configure(state="normal")
                except:
                    pass
        return
    
    W = int(tmpcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(tmpcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(tmpcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    p = dict(params)
    test_cx = xCenter if center_valid and xCenter is not None else W // 2
    test_cy = yCenter if center_valid and yCenter is not None else H // 2
    
    all_lines = []
    idx = 0
    progress_marks = set(int(N * q / 10) for q in range(1, 10))
    
    while True:
        ret, frm = tmpcap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        bg_sub = apply_background_subtraction(gray)
        
        ksize = max(1, int(p["blur"]))
        if ksize % 2 == 0:
            ksize += 1
        blur = cv2.GaussianBlur(bg_sub, (ksize, ksize), 0)
        
        if p.get("use_otsu", 0):
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            thresh_type = cv2.THRESH_BINARY_INV if p.get("invert_threshold", 0) else cv2.THRESH_BINARY
            _, binary = cv2.threshold(blur, int(p["threshold"]), 255, thresh_type)
        
        blobs, _ = detect_blobs_watershed(binary, p, test_cx, test_cy)
        
        method = p.get("pair_method", "Hungarian")
        if method == "Greedy":
            pairs = pair_scored(blobs, p, test_cx, test_cy, True)
        elif method == "Symmetric":
            pairs = pair_scored_symmetric(blobs, p, test_cx, test_cy, True)
        else:
            pairs = pair_scored_hungarian(blobs, p, test_cx, test_cy, True)
        
        for (pid, xi, yi, xj, yj, *_rest) in pairs:
            all_lines.append((xi, yi, xj, yj))
        
        if idx in progress_marks:
            pct = 100 * idx / max(1, N)
            print(f"[INFO] Processing: {pct:5.1f}% ({idx}/{N})")
        
        idx += 1
    
    tmpcap.release()
    
    if len(all_lines) < 2:
        print("[WARN] Not enough pairs found for optimization.")
        if root:
            for w in widgets.values():
                try:
                    w.configure(state="normal")
                except:
                    pass
        return
    
    print(f"[INFO] Analyzing {len(all_lines)} pair lines...")
    
    grid_size = 5
    votes = {}
    max_votes = 0
    best_pos = (test_cx, test_cy)
    
    for i in range(len(all_lines)):
        x1a, y1a, x2a, y2a = all_lines[i]
        dxa = x2a - x1a
        dya = y2a - y1a
        if abs(dxa) < 1e-6 and abs(dya) < 1e-6:
            continue
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
            
            denom = uxa * uyb - uya * uxb
            if abs(denom) < 1e-6:
                continue
            
            dx = x1b - x1a
            dy = y1b - y1a
            t = (dx * uyb - dy * uxb) / denom
            
            ix = x1a + t * uxa
            iy = y1a + t * uya
            
            if 0 <= ix < W and 0 <= iy < H:
                gx = int(ix / grid_size)
                gy = int(iy / grid_size)
                key = (gx, gy)
                votes[key] = votes.get(key, 0) + 1
                
                if votes[key] > max_votes:
                    max_votes = votes[key]
                    best_pos = (gx * grid_size + grid_size // 2, gy * grid_size + grid_size // 2)
    
    if max_votes < 2:
        print("[WARN] Could not find clear intersection point.")
        new_cx, new_cy = W // 2, H // 2
    else:
        new_cx, new_cy = best_pos
        print(f"[INFO] Found optimal center at ({new_cx}, {new_cy}) with {max_votes} votes")
    
    set_centerxy(new_cx, new_cy)
    
    if root:
        for w in widgets.values():
            try:
                w.configure(state="normal")
            except:
                pass
    
    messagebox.showinfo("Optimization Complete", f"Optical center set to ({new_cx}, {new_cy})")

# ============ GUI ============
def build_gui():
    """Build simple GUI for watershed blob detection."""
    global root, widgets, gui_vars_numeric, gui_vars_check
    
    root = tk.Tk()
    root.title("Watershed Blob Detector")
    root.geometry("900x700+60+60")
    
    content = ttk.Frame(root, padding="10")
    content.pack(fill="both", expand=True)
    content.grid_columnconfigure(0, weight=1)
    content.grid_columnconfigure(1, weight=1)
    
    # Buttons (span both columns)
    btn_frame = ttk.Frame(content)
    btn_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    
    widgets["btn_open"] = ttk.Button(btn_frame, text="📂 Open Video", command=open_video)
    widgets["btn_open"].pack(side="left", padx=2)
    
    widgets["btn_optimize"] = ttk.Button(btn_frame, text="🎯 Optimize Center", command=optimize_optical_center)
    widgets["btn_optimize"].pack(side="left", padx=2)
    
    widgets["btn_export"] = ttk.Button(btn_frame, text="💾 Export", command=export_video)
    widgets["btn_export"].pack(side="left", padx=2)
    
    widgets["btn_save"] = ttk.Button(btn_frame, text="💾 Save Settings", command=save_settings)
    widgets["btn_save"].pack(side="left", padx=2)
    
    widgets["btn_load"] = ttk.Button(btn_frame, text="📂 Load Settings", command=load_settings)
    widgets["btn_load"].pack(side="left", padx=2)
    
    widgets["btn_exit"] = ttk.Button(btn_frame, text="🚪 Exit", command=on_exit)
    widgets["btn_exit"].pack(side="left", padx=2)
    
    # Parameters
    def create_slider(parent, row, label, key, from_, to_, is_int=True):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        var = tk.IntVar(value=int(params[key])) if is_int else tk.DoubleVar(value=float(params[key]))
        scale = ttk.Scale(parent, from_=from_, to=to_, orient=tk.HORIZONTAL, variable=var)
        scale.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        
        def update_val(v):
            val = int(float(v)) if is_int else float(v)
            params[key] = val
            if is_int:
                lbl_val.config(text=str(val))
            else:
                lbl_val.config(text=f"{val:.2f}")
        
        scale.config(command=lambda v, k=key: update_val(v))
        lbl_val = ttk.Label(parent, text=str(params[key]), width=8)
        lbl_val.grid(row=row, column=2, padx=4)
        
        widgets[f"scale_{key}"] = scale
        gui_vars_numeric[key] = var
        return row + 1
    
    def create_checkbox(parent, row, label, key):
        var = tk.IntVar(value=int(params.get(key, 0)))
        chk = ttk.Checkbutton(parent, text=label, variable=var,
                             command=lambda k=key, v=var: params.__setitem__(k, int(v.get())))
        chk.grid(row=row, column=0, columnspan=3, sticky="w", padx=4, pady=2)
        widgets[f"chk_{key}"] = chk
        gui_vars_check[key] = var
        return row + 1
    
    # Helper function to handle debug windows toggle
    def toggle_debug_windows():
        val = params.get("show_debug_windows", 0)
        debug_windows = ["Background Subtraction", "Binary", "EDT", "Markers", "Labels"]
        
        if val:
            # Create and resize all debug windows
            for win_name in debug_windows:
                try:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win_name, DEBUG_WINDOW_SIZE[0], DEBUG_WINDOW_SIZE[1])
                except:
                    pass
        else:
            # Close all debug windows
            for win_name in debug_windows:
                try:
                    cv2.destroyWindow(win_name)
                except:
                    pass
    
    # Create master debug checkbox
    def create_debug_checkbox(parent, row, label, key):
        var = tk.IntVar(value=int(params.get(key, 0)))
        chk = ttk.Checkbutton(parent, text=label, variable=var,
                             command=lambda k=key, v=var: (
                                 params.__setitem__(k, int(v.get())),
                                 toggle_debug_windows()
                             ))
        chk.grid(row=row, column=0, columnspan=3, sticky="w", padx=4, pady=2)
        widgets[f"chk_{key}"] = chk
        gui_vars_check[key] = var
        return row + 1
    
    # Left column
    left_col = ttk.Frame(content)
    left_col.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
    
    # Right column
    right_col = ttk.Frame(content)
    right_col.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
    
    # Background Subtraction (Left)
    frm_bg = ttk.LabelFrame(left_col, text="Background Subtraction", padding="5")
    frm_bg.pack(fill="x", pady=5)
    frm_bg.grid_columnconfigure(1, weight=1)
    row = 0
    row = create_slider(frm_bg, row, "Alpha", "bg_alpha", 0, 100, False)
    row = create_slider(frm_bg, row, "Static Threshold", "bg_static_thresh", 1, 20, True)
    
    # Binary Thresholding (Left)
    frm_bin = ttk.LabelFrame(left_col, text="Binary Thresholding", padding="5")
    frm_bin.pack(fill="x", pady=5)
    frm_bin.grid_columnconfigure(1, weight=1)
    row = 0
    row = create_slider(frm_bin, row, "Threshold", "threshold", 0, 255, True)
    row = create_slider(frm_bin, row, "Blur", "blur", 1, 25, True)
    row = create_checkbox(frm_bin, row, "Use Otsu", "use_otsu")
    row = create_checkbox(frm_bin, row, "Invert", "invert_threshold")
    
    # EDT / Watershed (Left)
    frm_ws = ttk.LabelFrame(left_col, text="EDT / Watershed", padding="5")
    frm_ws.pack(fill="x", pady=5)
    frm_ws.grid_columnconfigure(1, weight=1)
    row = 0
    row = create_slider(frm_ws, row, "Min Distance", "ws_min_distance", 5, 50, True)
    # Marker threshold slider (0.0-1.0, displayed as 0-100)
    ttk.Label(frm_ws, text="Marker Threshold").grid(row=row, column=0, sticky="w", padx=4, pady=2)
    var_mt = tk.IntVar(value=int(params["ws_marker_threshold"] * 100))
    scale_mt = ttk.Scale(frm_ws, from_=0, to=100, orient=tk.HORIZONTAL, variable=var_mt)
    scale_mt.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
    lbl_mt = ttk.Label(frm_ws, text=f"{params['ws_marker_threshold']:.2f}", width=8)
    lbl_mt.grid(row=row, column=2, padx=4)
    def update_mt(v):
        val = float(v) / 100.0
        params["ws_marker_threshold"] = val
        lbl_mt.config(text=f"{val:.2f}")
    scale_mt.config(command=update_mt)
    widgets["scale_ws_marker_threshold"] = scale_mt
    gui_vars_numeric["ws_marker_threshold"] = var_mt
    row += 1
    
    # EDT Power/Gamma slider (0.1-2.0, displayed as 10-200)
    # Lower values (<1.0) emphasize small distances (better for small blobs, preserves detail)
    # Higher values (>1.0) emphasize large distances (helps separate overlapping objects)
    ttk.Label(frm_ws, text="EDT Power (×100)").grid(row=row, column=0, sticky="w", padx=4, pady=2)
    var_ep = tk.IntVar(value=int(params["ws_edt_power"] * 100))
    scale_ep = ttk.Scale(frm_ws, from_=10, to=200, orient=tk.HORIZONTAL, variable=var_ep)
    scale_ep.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
    lbl_ep = ttk.Label(frm_ws, text=f"{params['ws_edt_power']:.2f}", width=8)
    lbl_ep.grid(row=row, column=2, padx=4)
    def update_ep(v):
        val = max(0.1, min(2.0, float(v) / 100.0))
        params["ws_edt_power"] = val
        lbl_ep.config(text=f"{val:.2f}")
    scale_ep.config(command=update_ep)
    widgets["scale_ws_edt_power"] = scale_ep
    gui_vars_numeric["ws_edt_power"] = var_ep
    row += 1
    
    # Blob Filtering (Left)
    frm_blob = ttk.LabelFrame(left_col, text="Blob Filtering", padding="5")
    frm_blob.pack(fill="x", pady=5)
    frm_blob.grid_columnconfigure(1, weight=1)
    row = 0
    row = create_slider(frm_blob, row, "Min Area", "minArea", 0, 200, True)
    row = create_slider(frm_blob, row, "Max Area", "maxArea", 100, 1000, True)
    row = create_slider(frm_blob, row, "Max Width", "maxW", 1, 200, True)
    
    # Debug windows checkbox (Left)
    frm_debug = ttk.LabelFrame(left_col, text="Debug", padding="5")
    frm_debug.pack(fill="x", pady=5)
    frm_debug.grid_columnconfigure(1, weight=1)
    row = 0
    row = create_debug_checkbox(frm_debug, row, "📹 Show Debug Videos", "show_debug_windows")
    
    # Pairing Parameters (Right)
    frm_pair = ttk.LabelFrame(right_col, text="Pairing Parameters", padding="5")
    frm_pair.pack(fill="x", pady=5)
    frm_pair.grid_columnconfigure(1, weight=1)
    row = 0
    row = create_slider(frm_pair, row, "Max Radial Gap", "maxRadGap", 0, 200, True)
    row = create_slider(frm_pair, row, "Max Angle Diff", "maxDMR", 0, 30, True)
    row = create_slider(frm_pair, row, "Max Center Offset", "maxCenterOff", 0, 200, True)
    
    # Pairing weights (0-100, stored as 0.0-1.0)
    ttk.Label(frm_pair, text="Weight: Angle").grid(row=row, column=0, sticky="w", padx=4, pady=2)
    var_wt = tk.IntVar(value=int(params["w_theta"] * 100))
    scale_wt = ttk.Scale(frm_pair, from_=0, to=100, orient=tk.HORIZONTAL, variable=var_wt)
    scale_wt.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
    lbl_wt = ttk.Label(frm_pair, text=f"{params['w_theta']:.2f}", width=8)
    lbl_wt.grid(row=row, column=2, padx=4)
    def update_wt(v):
        val = float(v) / 100.0
        params["w_theta"] = val
        lbl_wt.config(text=f"{val:.2f}")
    scale_wt.config(command=update_wt)
    widgets["scale_w_theta"] = scale_wt
    gui_vars_numeric["w_theta"] = var_wt
    row += 1
    
    ttk.Label(frm_pair, text="Weight: Area").grid(row=row, column=0, sticky="w", padx=4, pady=2)
    var_wa = tk.IntVar(value=int(params["w_area"] * 100))
    scale_wa = ttk.Scale(frm_pair, from_=0, to=100, orient=tk.HORIZONTAL, variable=var_wa)
    scale_wa.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
    lbl_wa = ttk.Label(frm_pair, text=f"{params['w_area']:.2f}", width=8)
    lbl_wa.grid(row=row, column=2, padx=4)
    def update_wa(v):
        val = float(v) / 100.0
        params["w_area"] = val
        lbl_wa.config(text=f"{val:.2f}")
    scale_wa.config(command=update_wa)
    widgets["scale_w_area"] = scale_wa
    gui_vars_numeric["w_area"] = var_wa
    row += 1
    
    ttk.Label(frm_pair, text="Weight: Center").grid(row=row, column=0, sticky="w", padx=4, pady=2)
    var_wc = tk.IntVar(value=int(params["w_center"] * 100))
    scale_wc = ttk.Scale(frm_pair, from_=0, to=100, orient=tk.HORIZONTAL, variable=var_wc)
    scale_wc.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
    lbl_wc = ttk.Label(frm_pair, text=f"{params['w_center']:.2f}", width=8)
    lbl_wc.grid(row=row, column=2, padx=4)
    def update_wc(v):
        val = float(v) / 100.0
        params["w_center"] = val
        lbl_wc.config(text=f"{val:.2f}")
    scale_wc.config(command=update_wc)
    widgets["scale_w_center"] = scale_wc
    gui_vars_numeric["w_center"] = var_wc
    row += 1
    
    # Min Score (0.1-2.0, displayed as 10-200)
    ttk.Label(frm_pair, text="Min Score (×100)").grid(row=row, column=0, sticky="w", padx=4, pady=2)
    var_smin = tk.IntVar(value=int(params["Smin"] * 100))
    scale_smin = ttk.Scale(frm_pair, from_=10, to=200, orient=tk.HORIZONTAL, variable=var_smin)
    scale_smin.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
    lbl_smin = ttk.Label(frm_pair, text=f"{params['Smin']:.2f}", width=8)
    lbl_smin.grid(row=row, column=2, padx=4)
    def update_smin(v):
        val = max(0.1, min(2.0, float(v) / 100.0))
        params["Smin"] = val
        lbl_smin.config(text=f"{val:.2f}")
    scale_smin.config(command=update_smin)
    widgets["scale_Smin"] = scale_smin
    gui_vars_numeric["Smin"] = var_smin
    row += 1
    
    # Pairing method dropdown
    ttk.Label(frm_pair, text="Method").grid(row=row, column=0, sticky="w", padx=4, pady=2)
    method_var = tk.StringVar(value=params.get("pair_method", "Hungarian"))
    methods = ["Greedy", "Symmetric", "Hungarian"]
    cmb_method = ttk.Combobox(frm_pair, values=methods, state="readonly", textvariable=method_var)
    cmb_method.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
    cmb_method.bind("<<ComboboxSelected>>", lambda e: params.__setitem__("pair_method", method_var.get()))
    widgets["cmb_pair_method"] = cmb_method
    row += 1
    
    # Tracking Parameters (Right)
    frm_track = ttk.LabelFrame(right_col, text="Tracking Parameters", padding="5")
    frm_track.pack(fill="x", pady=5)
    frm_track.grid_columnconfigure(1, weight=1)
    row = 0
    ttk.Label(frm_track, text="Max Match Distance").grid(row=row, column=0, sticky="w", padx=4, pady=2)
    var_tmd = tk.DoubleVar(value=float(params["track_max_match_dist"]))
    scale_tmd = ttk.Scale(frm_track, from_=5.0, to=100.0, orient=tk.HORIZONTAL, variable=var_tmd)
    scale_tmd.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
    lbl_tmd = ttk.Label(frm_track, text=f"{params['track_max_match_dist']:.1f}", width=8)
    lbl_tmd.grid(row=row, column=2, padx=4)
    def update_tmd(v):
        val = float(v)
        params["track_max_match_dist"] = val
        lbl_tmd.config(text=f"{val:.1f}")
    scale_tmd.config(command=update_tmd)
    widgets["scale_track_max_match_dist"] = scale_tmd
    gui_vars_numeric["track_max_match_dist"] = var_tmd
    row += 1
    row = create_slider(frm_track, row, "Max Misses", "track_max_misses", 1, 30, True)
    
    # Overlay Targets (Right column, after Tracking)
    frm_target = ttk.LabelFrame(right_col, text="Overlay Targets", padding="5")
    frm_target.pack(fill="x", pady=5)
    frm_target.grid_columnconfigure((0, 1), weight=1)
    
    def add_target_check(col, text, key):
        var = tk.IntVar(value=int(overlay_targets[key]))
        chk = ttk.Checkbutton(
            frm_target, text=text, variable=var, command=lambda k=key, v=var: overlay_targets.__setitem__(k, int(v.get()))
        )
        chk.grid(row=0, column=col, padx=4, pady=2, sticky="w")
        widgets[f"chk_{key}"] = chk
        gui_vars_check[key] = var
    
    add_target_check(0, "Enable Tracked", "enable_tracked")
    add_target_check(1, "Enable Binary", "enable_binary")
    
    # Overlays (Right column)
    frm_ov = ttk.LabelFrame(right_col, text="Overlays", padding="5")
    frm_ov.pack(fill="x", pady=5)
    for i in range(3):
        frm_ov.grid_columnconfigure(i, weight=1)
    
    def add_check(grid_r, grid_c, text, key):
        var = tk.IntVar(value=int(overlays[key]))
        chk = ttk.Checkbutton(
            frm_ov, text=text, variable=var, command=lambda k=key, v=var: overlays.__setitem__(k, int(v.get()))
        )
        chk.grid(row=grid_r, column=grid_c, padx=4, pady=2, sticky="w")
        gui_vars_check[key] = var
        widgets[f"chk_{key}"] = chk
    
    add_check(0, 0, "Blobs (Green)", "show_blobs")
    add_check(0, 1, "Optical Center (Yellow)", "show_center")
    add_check(0, 2, "Pair Midpoint", "show_pair_center")
    add_check(1, 0, "Pair Line", "show_lines")
    add_check(1, 1, "Pair Rays", "show_rays")
    add_check(1, 2, "Pair Points", "show_pair_points")
    
    # Pair Color dropdown
    ttk.Label(frm_ov, text="Pair Color:").grid(row=2, column=0, padx=4, pady=2, sticky="w")
    label_mode_var = tk.StringVar(value=overlays.get("label_mode", "Red/Blue"))
    current_mode = overlays.get("label_mode", "Red/Blue")
    display_mode = "White" if current_mode == "None" else current_mode
    label_mode_var.set(display_mode)
    cmb_label = ttk.Combobox(
        frm_ov,
        values=["White", "Red/Blue", "Random"],
        state="readonly",
        textvariable=label_mode_var,
        width=12,
    )
    cmb_label.grid(row=2, column=1, padx=4, pady=2, sticky="w")
    cmb_label.set(display_mode)
    
    def on_label_mode_change(*_):
        display_val = label_mode_var.get()
        overlays["label_mode"] = "None" if display_val == "White" else display_val
    
    cmb_label.bind("<<ComboboxSelected>>", on_label_mode_change)
    label_mode_var.trace_add("write", on_label_mode_change)
    widgets["cmb_label_mode"] = cmb_label
    
    # Text labels checkboxes
    add_check(3, 0, "#A/#C", "show_text_labels")
    add_check(3, 1, "Z value", "show_z_value")
    add_check(3, 2, "X/Y values", "show_xy_values")
    
    # Real point checkbox
    add_check(4, 0, "Real point", "show_real_point")
    
    # Preview Overlay section (Right column)
    frm_preview_ov = ttk.LabelFrame(right_col, text="Preview Overlay", padding="5")
    frm_preview_ov.pack(fill="x", pady=5)
    for i in range(2):
        frm_preview_ov.grid_columnconfigure(i, weight=1)
    
    def add_preview_check(grid_r, grid_c, text, key):
        var = tk.IntVar(value=int(overlays.get(key, 0)))
        chk = ttk.Checkbutton(
            frm_preview_ov, text=text, variable=var, command=lambda k=key, v=var: overlays.__setitem__(k, int(v.get()))
        )
        chk.grid(row=grid_r, column=grid_c, padx=4, pady=2, sticky="w")
        gui_vars_check[key] = var
        widgets[f"chk_{key}"] = chk
    
    add_preview_check(0, 0, "Current Stats", "show_current_stats")
    add_preview_check(0, 1, "Total Stats", "show_total_stats")
    
    # Help text (span both columns)
    help_text = ttk.Label(content, text="💡 Tip: Click in the 'Tracked' window to set optical center", 
                         font=("Arial", 9), foreground="gray")
    help_text.grid(row=1, column=0, columnspan=2, pady=5)
    
    # Load settings on startup
    load_settings()
    
    root.mainloop()

def open_video():
    """Open video file."""
    global video_path, cap, background_image, tracker, xCenter, yCenter, center_valid
    
    file_path = filedialog.askopenfilename(
        title="Open Video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if not file_path:
        return
    
    video_path = file_path
    print(f"[INFO] Opening video: {video_path}")
    
    # Build background
    try:
        background_image = build_background_from_video(video_path)
        print("[INFO] Background ready")
    except Exception as e:
        print(f"[ERR] Background failed: {e}")
        background_image = None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video: {video_path}")
        return
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video: {W}×{H}")
    
    # Initialize tracker
    tracker = PairTracker(
        max_match_dist_px=float(params.get("track_max_match_dist", 25.0)),
        max_misses=int(params.get("track_max_misses", 10))
    )
    
    # Set default center to frame center (user can click to change)
    if not center_valid or xCenter is None or yCenter is None:
        xCenter, yCenter = W // 2, H // 2
        center_valid = False  # Will be set when user clicks
        print(f"[INFO] Default center: ({xCenter},{yCenter}) - Click in Tracked window to set")
    
    # Set video seed for consistent colors
    set_video_seed(video_path)
    
    # Create tracked window (always visible)
    cv2.namedWindow("Tracked", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracked", W, H)
    cv2.setMouseCallback("Tracked", on_mouse_tracked)
    
    # Create debug windows only if enabled
    if params.get("show_debug_windows", 0):
        cv2.namedWindow("Background Subtraction", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Background Subtraction", DEBUG_WINDOW_SIZE[0], DEBUG_WINDOW_SIZE[1])
        cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Binary", DEBUG_WINDOW_SIZE[0], DEBUG_WINDOW_SIZE[1])
        cv2.namedWindow("EDT", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("EDT", DEBUG_WINDOW_SIZE[0], DEBUG_WINDOW_SIZE[1])
        cv2.namedWindow("Markers", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Markers", DEBUG_WINDOW_SIZE[0], DEBUG_WINDOW_SIZE[1])
        cv2.namedWindow("Labels", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Labels", DEBUG_WINDOW_SIZE[0], DEBUG_WINDOW_SIZE[1])
    
    # Start preview loop
    preview_loop()

def preview_loop():
    """Main preview loop."""
    global cap, params, overlays, tracker, xCenter, yCenter, center_valid
    
    if cap is None or not cap.isOpened():
        root.after(66, preview_loop)  # ~15 FPS
        return
    
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            root.after(66, preview_loop)
            return
    
    # Step 1: Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Background subtraction
    bg_sub = apply_background_subtraction(gray)
    show_debug = params.get("show_debug_windows", 0)
    if show_debug:
        # Ensure window exists and is properly sized
        try:
            cv2.namedWindow("Background Subtraction", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Background Subtraction", DEBUG_WINDOW_SIZE[0], DEBUG_WINDOW_SIZE[1])
        except:
            pass
        cv2.imshow("Background Subtraction", bg_sub)
    
    # Step 3: Binary thresholding
    ksize = max(1, int(params["blur"]))
    if ksize % 2 == 0:
        ksize += 1
    blur = cv2.GaussianBlur(bg_sub, (ksize, ksize), 0)
    
    if params.get("use_otsu", 0):
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        thresh_type = cv2.THRESH_BINARY_INV if params.get("invert_threshold", 0) else cv2.THRESH_BINARY
        _, binary = cv2.threshold(blur, int(params["threshold"]), 255, thresh_type)
    
    if show_debug:
        # Ensure window exists and is properly sized
        try:
            cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Binary", DEBUG_WINDOW_SIZE[0], DEBUG_WINDOW_SIZE[1])
        except:
            pass
        cv2.imshow("Binary", binary)
    
    # Step 4: Watershed detection
    try:
        # Default center if not set
        if not center_valid or xCenter is None or yCenter is None:
            h, w = gray.shape[:2]
            xCenter, yCenter = w // 2, h // 2
            center_valid = True
        
        blobs, debug_images = detect_blobs_watershed(binary, params, xCenter, yCenter)
        
        # Step 5: Pair blobs
        method = params.get("pair_method", "Hungarian")
        if method == "Greedy":
            pairs_before_tracking = pair_scored(blobs, params, xCenter, yCenter, center_valid)
        elif method == "Symmetric":
            pairs_before_tracking = pair_scored_symmetric(blobs, params, xCenter, yCenter, center_valid)
        else:
            pairs_before_tracking = pair_scored_hungarian(blobs, params, xCenter, yCenter, center_valid)
        
        # Step 6: Track pairs
        if tracker is None:
            tracker = PairTracker(
                max_match_dist_px=float(params.get("track_max_match_dist", 25.0)),
                max_misses=int(params.get("track_max_misses", 10))
            )
        tracker.max_match_dist_px = float(params.get("track_max_match_dist", 25.0))
        tracker.max_misses = int(params.get("track_max_misses", 10))
        pairs = tracker.update(pairs_before_tracking)
        
        # Show debug windows (only if enabled)
        if show_debug:
            # Ensure windows exist and are properly sized
            if "edt" in debug_images:
                try:
                    cv2.namedWindow("EDT", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("EDT", DEBUG_WINDOW_SIZE[0], DEBUG_WINDOW_SIZE[1])
                except:
                    pass
                cv2.imshow("EDT", debug_images["edt"])
            if "markers" in debug_images:
                try:
                    cv2.namedWindow("Markers", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Markers", DEBUG_WINDOW_SIZE[0], DEBUG_WINDOW_SIZE[1])
                except:
                    pass
                cv2.imshow("Markers", debug_images["markers"])
            if "labels" in debug_images:
                try:
                    cv2.namedWindow("Labels", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Labels", DEBUG_WINDOW_SIZE[0], DEBUG_WINDOW_SIZE[1])
                except:
                    pass
                cv2.imshow("Labels", debug_images["labels"])
        
        # Step 7: Draw on tracked window
        tracked = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        label_mode = overlays.get("label_mode", "Red/Blue")
        
        # Apply overlays based on settings
        if overlays.get("show_blobs", 1):
            draw_blob_boxes(tracked, blobs)
        
        if overlays.get("show_center", 1) and center_valid and xCenter is not None and yCenter is not None:
            draw_center(tracked, xCenter, yCenter)
        
        if overlays.get("show_pair_center", 0):
            draw_pair_centers(tracked, pairs, label_mode, video_path)
        
        if overlays.get("show_lines", 1):
            show_labels = overlays.get("show_text_labels", 1)
            show_points = overlays.get("show_pair_points", 1)
            draw_pair_lines(tracked, pairs, show_labels, label_mode, video_path, show_points)
        
        if overlays.get("show_rays", 1):
            draw_pair_rays_toward_center(tracked, pairs, tracked.shape[1], xCenter, yCenter, label_mode, video_path)
        
        # Stats overlay
        if overlays.get("show_current_stats", 0) or overlays.get("show_total_stats", 0):
            draw_stats_overlay(
                tracked, pairs_before_tracking, pairs,
                bool(overlays.get("show_current_stats", 0)),
                bool(overlays.get("show_total_stats", 0))
            )
        
        cv2.imshow("Tracked", tracked)
        cv2.setMouseCallback("Tracked", on_mouse_tracked)
        
    except Exception as e:
        print(f"[ERR] Detection error: {e}")
        tracked = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if center_valid and xCenter is not None and yCenter is not None:
            draw_center(tracked, xCenter, yCenter)
        cv2.imshow("Tracked", tracked)
        cv2.setMouseCallback("Tracked", on_mouse_tracked)
    
    cv2.waitKey(1)
    root.after(66, preview_loop)

def export_video():
    """Export video and CSV."""
    global video_path, cap, params, overlays, xCenter, yCenter, center_valid
    
    if not video_path or not os.path.exists(video_path):
        messagebox.showerror("Error", "No video loaded")
        return
    
    if not SKIMAGE_AVAILABLE:
        messagebox.showerror("Error", "scikit-image required for export")
        return
    
    if not center_valid or xCenter is None or yCenter is None:
        messagebox.showwarning("Warning", "Optical center not set. Using frame center.")
        tmpcap_test = cv2.VideoCapture(video_path)
        if tmpcap_test.isOpened():
            W = int(tmpcap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(tmpcap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
            xCenter, yCenter = W // 2, H // 2
            center_valid = True
        tmpcap_test.release()
    
    paths = export_paths_for(video_path)
    out_video = paths["tracked_mp4"]
    out_csv = paths["pairs_csv"]
    
    tmpcap = cv2.VideoCapture(video_path)
    if not tmpcap.isOpened():
        messagebox.showerror("Error", "Could not open video for export")
        return
    
    W = int(tmpcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(tmpcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(tmpcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = tmpcap.get(cv2.CAP_PROP_FPS) or 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, max(1.0, fps), (W, H))
    
    print(f"[INFO] Exporting to: {out_video}")
    
    # Initialize tracker for export
    export_tracker = PairTracker(
        max_match_dist_px=float(params.get("track_max_match_dist", 25.0)),
        max_misses=int(params.get("track_max_misses", 10))
    )
    
    rows = []
    idx = 0
    progress_marks = set(int(N * q / 20) for q in range(1, 20))
    p = dict(params)
    
    while True:
        ret, frame = tmpcap.read()
        if not ret:
            break
        
        # Process frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg_sub = apply_background_subtraction(gray)
        
        ksize = max(1, int(p["blur"]))
        if ksize % 2 == 0:
            ksize += 1
        blur = cv2.GaussianBlur(bg_sub, (ksize, ksize), 0)
        
        if p.get("use_otsu", 0):
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            thresh_type = cv2.THRESH_BINARY_INV if p.get("invert_threshold", 0) else cv2.THRESH_BINARY
            _, binary = cv2.threshold(blur, int(p["threshold"]), 255, thresh_type)
        
        try:
            blobs, _ = detect_blobs_watershed(binary, p, xCenter, yCenter)
        except Exception as e:
            print(f"[ERR] Frame {idx}: {e}")
            blobs = []
        
        # Pair blobs
        method = p.get("pair_method", "Hungarian")
        if method == "Greedy":
            pairs_before_tracking = pair_scored(blobs, p, xCenter, yCenter, center_valid)
        elif method == "Symmetric":
            pairs_before_tracking = pair_scored_symmetric(blobs, p, xCenter, yCenter, center_valid)
        else:
            pairs_before_tracking = pair_scored_hungarian(blobs, p, xCenter, yCenter, center_valid)
        
        # Track pairs
        pairs = export_tracker.update(pairs_before_tracking)
        
        # Draw on output frame
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        label_mode = overlays.get("label_mode", "Red/Blue")
        
        # Apply overlays based on settings
        if overlays.get("show_blobs", 1):
            draw_blob_boxes(output, blobs)
        
        if overlays.get("show_center", 1) and center_valid and xCenter is not None and yCenter is not None:
            draw_center(output, xCenter, yCenter)
        
        if overlays.get("show_pair_center", 0):
            draw_pair_centers(output, pairs, label_mode, video_path)
        
        if overlays.get("show_lines", 1):
            show_labels = overlays.get("show_text_labels", 1)
            show_points = overlays.get("show_pair_points", 1)
            draw_pair_lines(output, pairs, show_labels, label_mode, video_path, show_points)
        
        if overlays.get("show_rays", 1):
            draw_pair_rays_toward_center(output, pairs, output.shape[1], xCenter, yCenter, label_mode, video_path)
        
        # Stats overlay
        if overlays.get("show_current_stats", 0) or overlays.get("show_total_stats", 0):
            draw_stats_overlay(
                output, pairs_before_tracking, pairs,
                bool(overlays.get("show_current_stats", 0)),
                bool(overlays.get("show_total_stats", 0))
            )
        
        writer.write(output)
        
        # CSV row - match detect_pairs.py format
        pair_count = len(set(tid for tid, *_ in pairs))
        for pair in pairs:
            pid, xi, yi, xj, yj, th_i, r_i, th_j, r_j, score = pair[0:10]
            
            rows.append([
                idx, pid, xi, yi, xj, yj, th_i, r_i, th_j, r_j, round(float(score), 4),
                pair_count,
                int(p["threshold"]), int(p["blur"]), 
                int(p["minArea"]), int(p["maxArea"]), int(p["maxW"]),
                float(p["maxRadGap"]), float(p["maxDMR"]), float(p["maxCenterOff"]),
                float(p["w_theta"]), float(p["w_area"]), float(p["w_center"]), 
                float(p["Smin"]),
            ])
        
        if idx in progress_marks:
            pct = 100 * idx / max(1, N)
            print(f"[INFO] {pct:.1f}% ({idx}/{N})")
        
        idx += 1
    
    tmpcap.release()
    writer.release()
    
    # Save CSV
    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "Frame_Number", "Track_ID", "Center_X", "Center_Y", "Right_X", "Right_Y",
                "Angle_A_deg", "A_px", "Angle_B_deg", "C_px", "Pair_Score",
                "Pair_Count",
                "Binary_Threshold", "Blur_Size", "Min_Area_px2", "Max_Area_px2", "Max_Width_px",
                "Max_Radial_Gap_px", "Max_Angle_Diff_deg", "Max_Center_Offset_px",
                "Weight_Angle", "Weight_Area", "Weight_Center", "Min_Score_Threshold",
            ])
            w.writerows(rows)
        print(f"[INFO] CSV saved: {out_csv}")
    except Exception as e:
        print(f"[ERR] CSV save failed: {e}")
    
    messagebox.showinfo("Export Complete", f"Video: {out_video}\nCSV: {out_csv}")

def save_settings():
    """Save current settings to JSON file."""
    global params, overlays, overlay_targets, xCenter, yCenter, center_valid, video_path
    center = (xCenter if xCenter is not None else 0, yCenter if yCenter is not None else 0, center_valid)
    success = save_preset_file(PRESET_PATH, params, overlays, overlay_targets, center, video_path)
    if success:
        messagebox.showinfo("Settings Saved", f"Settings saved to:\n{PRESET_PATH}")
    else:
        messagebox.showerror("Error", f"Failed to save settings to:\n{PRESET_PATH}")

def load_settings():
    """Load settings from JSON file."""
    global params, overlays, overlay_targets, xCenter, yCenter, center_valid, video_path, widgets, gui_vars_numeric, gui_vars_check
    
    loaded_params, loaded_overlays, loaded_targets, center_tuple, loaded_video, ok, _ = load_preset_file(
        PRESET_PATH, params, overlays, overlay_targets, video_path
    )
    
    if not ok:
        messagebox.showwarning("No Settings", f"No saved settings found at:\n{PRESET_PATH}")
        return
    
    # Update global state
    params.update(loaded_params)
    overlays.update(loaded_overlays)
    overlay_targets.update(loaded_targets)
    xCenter, yCenter = center_tuple[0], center_tuple[1]
    center_valid = center_tuple[2]
    if loaded_video:
        video_path = loaded_video
    
    # Update GUI widgets
    for key, var in gui_vars_numeric.items():
        if key in params:
            if isinstance(var, tk.IntVar):
                var.set(int(params[key]))
            elif isinstance(var, tk.DoubleVar):
                var.set(float(params[key]))
    
    for key, var in gui_vars_check.items():
        if key in overlays:
            var.set(int(overlays[key]))
        elif key in overlay_targets:
            var.set(int(overlay_targets[key]))
    
    # Update label mode combobox if it exists
    if "cmb_label_mode" in widgets:
        current_mode = overlays.get("label_mode", "Red/Blue")
        display_mode = "White" if current_mode == "None" else current_mode
        widgets["cmb_label_mode"].set(display_mode)
    
    messagebox.showinfo("Settings Loaded", f"Settings loaded from:\n{PRESET_PATH}")

def on_exit():
    """Exit application and save settings."""
    global cap
    # Save settings on exit
    save_settings()
    try:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    except:
        pass
    root.quit()

if __name__ == "__main__":
    build_gui()
