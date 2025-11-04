#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Calibration Tool

This tool allows you to:
  • Input multiple CSV files from pair detection
  • Specify the mm height and working distance for each CSV
  • Calculate magic offset and magic constant
  • Automatically save calibration to calibrations folder

The calibration uses linear regression on:
  - Zprime values calculated from highest quality pairs: Zprime = working_distance * (C-A)/(A+C)
  - Z values: the calibrated mm height input for each CSV
  - Formula: Z = Zprime * magic_constant + magic_offset
"""

import csv
import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from lib.xyzcalc import (
    extract_working_distance,
    extract_pixels_per_mm,
    calculate_b_px,
    calculate_b_mm,
    calculate_xy_mm,
    calculate_b_xy_from_pair
)


def get_latest_calibration_file() -> Optional[str]:
    """
    Find the latest calibration JSON file in the calibrations folder.
    Returns the path to the latest file, or None if no file is found.
    """
    calibrations_dir = Path("calibrations")
    
    if not calibrations_dir.exists():
        return None
    
    # Find all JSON files in the calibrations directory
    json_files = list(calibrations_dir.glob("*.json"))
    
    if not json_files:
        return None
    
    # Sort by modification time (newest first)
    json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    return str(json_files[0])


class VideoEntry:
    """Container for a single video calibration entry."""
    def __init__(self, frame, row):
        self.frame = frame
        self.row = row
        self.csv_var = tk.StringVar()
        self.height_var = tk.StringVar()
        self.csv_path = ""
        self.mm_height: Optional[float] = None
        
        # Create widgets
        ttk.Label(frame, text=f"CSV {row + 1}:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        
        csv_label = ttk.Label(frame, text="No CSV selected", foreground="gray")
        csv_label.grid(row=0, column=1, sticky="w", padx=5)
        self.csv_label = csv_label
        
        csv_btn = ttk.Button(frame, text="📂 Browse", 
                            command=lambda: self.select_csv())
        csv_btn.grid(row=0, column=2, padx=5)
        
        ttk.Label(frame, text="Height (mm):").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        height_entry = ttk.Entry(frame, textvariable=self.height_var, width=15)
        height_entry.grid(row=1, column=1, sticky="w", padx=5, pady=(5, 0))
        self.height_entry = height_entry
        
        remove_btn = ttk.Button(frame, text="✖ Remove", 
                               command=lambda: self.remove())
        remove_btn.grid(row=1, column=2, padx=5, pady=(5, 0))
        self.remove_btn = remove_btn
        
        # Store reference to main app for removal
        self.app = None
    
    def set_app(self, app):
        """Set reference to main app."""
        self.app = app
    
    def select_csv(self):
        """Open file dialog to select CSV file."""
        csv_file = filedialog.askopenfilename(
            title=f"Select CSV File for Calibration {self.row + 1}",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="inputs_outputs" if os.path.exists("inputs_outputs") else "."
        )
        
        if csv_file:
            self.csv_path = csv_file
            csv_name = os.path.basename(csv_file)
            self.csv_label.config(text=csv_name, foreground="black")
            print(f"[INFO] CSV {self.row + 1}: Selected {csv_name}")
            # Update live metrics when CSV is loaded
            if self.app:
                # Load CSV data for visualization
                csv_name = os.path.basename(csv_file)
                self.app.load_csv_for_visualization(csv_file, csv_name)
                self.app.update_live_metrics()
                self.app.update_3d_visualization()
    
    def get_data(self) -> Optional[Tuple[str, float, float, float, float, Dict, Dict]]:
        """
        Get CSV path, mm height, working distance, average Zprime, and average B from highest quality pairs.
        Also returns metrics for input and chosen datasets.
        Returns None if invalid.
        Returns: (csv_path, mm_height, working_distance_mm, avg_zprime, avg_b, input_metrics, chosen_metrics)
        """
        if not self.csv_path or not os.path.exists(self.csv_path):
            return None
        
        if not self.app:
            return None
        
        try:
            mm_val = float(self.height_var.get())
            if mm_val <= 0:
                return None
            
            # Get working distance from app's global field
            working_dist_val = float(self.app.working_dist_var.get())
            if working_dist_val <= 0:
                return None
        except (ValueError, tk.TclError):
            return None
        
        # Load pairs from the selected CSV file
        if not os.path.exists(self.csv_path):
            return None
        
        try:
            pairs_data = []
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Try new A/C notation first, fall back to old notation for backward compatibility
                        r_a = float(row.get("A_px", row.get("Radius_A_px", 0)))  # Inner radius (A)
                        r_c = float(row.get("C_px", row.get("Radius_B_px", 0)))  # Outer radius (C)
                        score = float(row.get("Pair_Score", 0))
                        track_id = int(row.get("Track_ID", 0))
                        
                        if r_a > 0 and r_c > 0 and score > 0:
                            frame_num = int(row.get('Frame_Number', 0))
                            pairs_data.append({
                                "r_a": r_a,
                                "r_c": r_c,
                                "score": score,
                                "track_id": track_id,
                                "frame": frame_num  # Store frame number for visualization
                            })
                    except (ValueError, KeyError):
                        continue
            
            if len(pairs_data) == 0:
                return None
            
            # STEP 1: First, isolate near-mean pairs (Z-based filtering)
            # Calculate Zprime, Zdoubleprime for ALL pairs first
            all_pairs_zprimes = []
            all_pairs_zdoubleprimes = []
            all_pairs_z_values = []  # Store (pair_idx, zprime, zdoubleprime, z_mm)
            
            for idx, p in enumerate(pairs_data):
                r_a = p["r_a"]
                r_c = p["r_c"]
                if r_a + r_c > 0:
                    zprime = working_dist_val * (r_c - r_a) / (r_a + r_c)
                    zdoubleprime = (r_c - r_a) / (r_a + r_c)
                    all_pairs_zprimes.append(zprime)
                    all_pairs_zdoubleprimes.append(zdoubleprime)
                    all_pairs_z_values.append((idx, zprime, zdoubleprime, None))
            
            # Try to load Z_mm from CSV if available
            if hasattr(self, 'app') and self.app:
                try:
                    with open(self.csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        z_mm_map = {}
                        for row in reader:
                            try:
                                track_id = int(row.get('Track_ID', 0))
                                z_mm_str = row.get('Z_mm', '').strip()
                                if z_mm_str and track_id not in z_mm_map:
                                    z_mm_val = float(z_mm_str)
                                    if z_mm_val > 0:
                                        z_mm_map[track_id] = z_mm_val
                            except (ValueError, KeyError):
                                continue
                    
                    for i, (idx, zp, zdp, _) in enumerate(all_pairs_z_values):
                        pair_track_id = pairs_data[idx].get("track_id", 0)
                        if pair_track_id in z_mm_map:
                            all_pairs_z_values[i] = (idx, zp, zdp, z_mm_map[pair_track_id])
                except Exception:
                    pass
            
            # Apply Z-based filtering if enabled (isolate near-mean pairs)
            near_mean_pairs_indices = list(range(len(pairs_data)))
            z_filter_stats = None
            
            if hasattr(self, 'app') and self.app and self.app.enable_z_filter_var.get():
                filter_type = self.app.z_filter_type_var.get()
                threshold_std = self.app.z_filter_threshold_var.get()
                
                # Select which Z values to use for filtering
                z_values_for_filter = []
                if filter_type == "Zprime":
                    z_values_for_filter = [zv[1] for zv in all_pairs_z_values]
                elif filter_type == "Zdoubleprime":
                    z_values_for_filter = [zv[2] for zv in all_pairs_z_values]
                elif filter_type == "Z_mm":
                    z_values_for_filter = [zv[3] for zv in all_pairs_z_values if zv[3] is not None]
                
                if len(z_values_for_filter) > 0 and len(z_values_for_filter) == len(all_pairs_z_values):
                    # Calculate mean and std of Z values
                    mean_z = np.mean(z_values_for_filter)
                    std_z = np.std(z_values_for_filter)
                    
                    if std_z > 0:
                        # Filter pairs: keep those within threshold_std standard deviations
                        filtered_indices = []
                        for i, zv in enumerate(all_pairs_z_values):
                            idx, zp, zdp, zmm = zv
                            
                            z_to_check = None
                            if filter_type == "Zprime":
                                z_to_check = zp
                            elif filter_type == "Zdoubleprime":
                                z_to_check = zdp
                            elif filter_type == "Z_mm":
                                z_to_check = zmm
                            
                            if z_to_check is not None:
                                z_score = abs(z_to_check - mean_z) / std_z
                                if z_score <= threshold_std:
                                    filtered_indices.append(idx)
                        
                        if len(filtered_indices) > 0:
                            near_mean_pairs_indices = filtered_indices
                            z_filter_stats = {
                                "filter_type": filter_type,
                                "threshold_std": float(threshold_std),
                                "mean_z": float(mean_z),
                                "std_z": float(std_z),
                                "pairs_before": len(pairs_data),
                                "pairs_after": len(filtered_indices),
                                "omitted": len(pairs_data) - len(filtered_indices)
                            }
            
            # STEP 2: Then, choose high quality pairs from near-mean pairs
            min_score_threshold = 0.9
            if hasattr(self, 'app') and self.app:
                min_score_threshold = self.app.min_score_var.get()
            
            # Filter by quality score
            quality_pairs = []
            if hasattr(self, 'app') and self.app and self.app.enable_quality_filter_var.get():
                quality_pairs = [pairs_data[i] for i in near_mean_pairs_indices 
                               if pairs_data[i]["score"] >= min_score_threshold]
            else:
                quality_pairs = [pairs_data[i] for i in near_mean_pairs_indices]
            
            # Check if we have enough good data - no fallbacks!
            if len(quality_pairs) < 10:
                # Warning will be shown when get_data returns insufficient data
                pass  # Let the caller handle the warning
            
            # Calculate Zprime, Zdoubleprime, and B for each quality pair
            # Zprime = working_distance * (C-A)/(A+C)
            # Zdoubleprime = (C-A)/(A+C)  (working_distance = 1)
            # B = (2*A*C)/(A+C)
            zprimes = []
            zdoubleprimes = []
            b_values = []
            pair_z_values = []  # Store Z values with their corresponding pair indices
            
            for idx, p in enumerate(quality_pairs):
                r_a = p["r_a"]  # A is the inner radius (smaller)
                r_c = p["r_c"]  # C is the outer radius (larger)
                if r_a + r_c > 0:
                    zprime = working_dist_val * (r_c - r_a) / (r_a + r_c)
                    zdoubleprime = (r_c - r_a) / (r_a + r_c)  # working_distance = 1
                    zprimes.append(zprime)
                    zdoubleprimes.append(zdoubleprime)
                    # Calculate B = (2*A*C)/(A+C)
                    b_val = (2 * r_a * r_c) / (r_a + r_c)
                    b_values.append(b_val)
                    pair_z_values.append((idx, zprime, zdoubleprime, None))  # Z_mm will be filled if available
            
            if len(quality_pairs) == 0:
                # Not enough data - return None to trigger warning
                return None
            
            if len(quality_pairs) < 10:
                # Warning: insufficient good data
                if hasattr(self, 'app') and self.app:
                    # Warning will be shown in calculate() method
                    pass
            
            # Final quality pairs are already filtered
            final_quality_pairs = quality_pairs
            
            # Store calibration pairs for visualization
            if hasattr(self, 'app') and self.app:
                csv_name = os.path.basename(self.csv_path)
                # Track which frames are calibration pairs
                calibration_frames_map = {}  # {track_id: set of frames}
                
                # Collect frames for all calibration pairs
                for p in final_quality_pairs:
                    track_id = p.get("track_id", 0)
                    frame = p.get("frame", 0)
                    if track_id > 0 and frame > 0:
                        if track_id not in calibration_frames_map:
                            calibration_frames_map[track_id] = set()
                        calibration_frames_map[track_id].add(frame)
                
                if csv_name not in self.app.viz_calibration_pairs:
                    self.app.viz_calibration_pairs[csv_name] = {}
                self.app.viz_calibration_pairs[csv_name] = calibration_frames_map
                
                # Load full CSV data for visualization if not already loaded
                if csv_name not in self.app.viz_data:
                    self.app.load_csv_for_visualization(self.csv_path, csv_name)
            
            if len(zprimes) == 0:
                return None
            
            avg_zprime = np.mean(zprimes)
            avg_b = np.mean(b_values)
            
            # Calculate metrics for chosen/quality pairs
            chosen_metrics = {
                "count": len(final_quality_pairs),
                "zprime": {
                    "mean": float(np.mean(zprimes)),
                    "std": float(np.std(zprimes)),
                    "min": float(np.min(zprimes)),
                    "max": float(np.max(zprimes))
                },
                "b": {
                    "mean": float(np.mean(b_values)),
                    "std": float(np.std(b_values)),
                    "min": float(np.min(b_values)),
                    "max": float(np.max(b_values))
                },
                "score": {
                    "mean": float(np.mean([p["score"] for p in final_quality_pairs])),
                    "std": float(np.std([p["score"] for p in final_quality_pairs])),
                    "min": float(np.min([p["score"] for p in final_quality_pairs])),
                    "max": float(np.max([p["score"] for p in final_quality_pairs]))
                }
            }
            
            # Add Z filtering statistics to chosen_metrics if available
            if z_filter_stats:
                chosen_metrics["z_filter"] = z_filter_stats
            
            # Calculate metrics for all input pairs
            all_zprimes = []
            all_b_values = []
            all_scores = []
            for p in pairs_data:
                r_a = p["r_a"]
                r_c = p["r_c"]
                if r_a + r_c > 0:
                    zprime = working_dist_val * (r_c - r_a) / (r_a + r_c)
                    all_zprimes.append(zprime)
                    b_val = (2 * r_a * r_c) / (r_a + r_c)
                    all_b_values.append(b_val)
                    all_scores.append(p["score"])
            
            input_metrics = {
                "count": len(pairs_data),
                "zprime": {
                    "mean": float(np.mean(all_zprimes)) if all_zprimes else 0.0,
                    "std": float(np.std(all_zprimes)) if all_zprimes else 0.0,
                    "min": float(np.min(all_zprimes)) if all_zprimes else 0.0,
                    "max": float(np.max(all_zprimes)) if all_zprimes else 0.0
                },
                "b": {
                    "mean": float(np.mean(all_b_values)) if all_b_values else 0.0,
                    "std": float(np.std(all_b_values)) if all_b_values else 0.0,
                    "min": float(np.min(all_b_values)) if all_b_values else 0.0,
                    "max": float(np.max(all_b_values)) if all_b_values else 0.0
                },
                "score": {
                    "mean": float(np.mean(all_scores)) if all_scores else 0.0,
                    "std": float(np.std(all_scores)) if all_scores else 0.0,
                    "min": float(np.min(all_scores)) if all_scores else 0.0,
                    "max": float(np.max(all_scores)) if all_scores else 0.0
                }
            }
            
            return (self.csv_path, mm_val, working_dist_val, avg_zprime, avg_b, input_metrics, chosen_metrics, z_filter_stats)
        
        except Exception as e:
            print(f"[ERROR] Failed to read {self.csv_path}: {e}")
            return None
    
    def remove(self):
        """Remove this entry from the GUI."""
        if self.app and len(self.app.video_entries) > 2:
            self.app.remove_video_entry(self)


class VideoCalibrationApp:
    """Main application for video calibration."""
    
    def __init__(self, root):
        self.root = root
        self.video_entries: List[VideoEntry] = []
        self.entries_frame = None
        self.canvas = None
        self.scrollbar = None
        self.scrollable_frame = None
        
        # Global working distance
        self.working_dist_var = tk.StringVar()
        
        self.setup_gui()
        # Add first two entries by default
        self.add_video_entry()
        self.add_video_entry()
        
        # Auto-load working distance from latest calibration file
        self._auto_load_from_latest_calibration()
    
    def setup_gui(self):
        """Create the GUI layout with three columns."""
        from lib.gui import apply_standard_theme, format_window_title, get_standard_size
        
        width, height = get_standard_size("xlarge")
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(1600, 700)
        self.root.title(format_window_title("Video Calibration Tool"))
        apply_standard_theme(self.root)
        
        # Create three-column layout using PanedWindow
        main_container = ttk.PanedWindow(self.root, orient="horizontal")
        main_container.pack(fill="both", expand=True)
        
        # LEFT COLUMN: Controls
        left_pane = ttk.Frame(main_container)
        main_container.add(left_pane, weight=2)
        
        # Create scrollable container for left column
        left_canvas = tk.Canvas(left_pane, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_pane, orient="vertical", command=left_canvas.yview)
        left_frame = ttk.Frame(left_canvas, padding="15")
        
        left_frame.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        
        left_window = left_canvas.create_window((0, 0), window=left_frame, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")
        
        def update_left_window_width(event):
            canvas_width = event.width
            left_canvas.itemconfig(left_window, width=canvas_width)
        left_canvas.bind('<Configure>', update_left_window_width)
        
        def on_mousewheel_left(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        left_canvas.bind("<MouseWheel>", on_mousewheel_left)
        
        # MIDDLE COLUMN: Data Metrics
        middle_pane = ttk.Frame(main_container)
        main_container.add(middle_pane, weight=1)
        self.setup_metrics_column(middle_pane)
        
        # RIGHT COLUMN: 3D Visualization
        right_pane = ttk.Frame(main_container)
        main_container.add(right_pane, weight=3)
        
        # Setup 3D plot in right pane
        self.setup_3d_visualization(right_pane)
        
        # Store reference for later updates
        self.main_frame = left_frame
        
        # Magic offset and constant storage
        self.magic_offset: Optional[float] = None
        self.magic_constant: Optional[float] = None
        self.calibration_data: Optional[Dict] = None
        
        # 3D visualization data
        self.viz_data = {}  # {csv_name: {track_id: [(frame, x, y, z), ...]}}
        self.viz_calibration_pairs = {}  # {csv_name: {track_id: set of frames that are calibration pairs}}
        
        # Coordinate unit tracking for axis labels
        self.viz_x_unit = "mm"
        self.viz_y_unit = "mm"
        self.viz_z_unit = "mm"
        
        # Instructions
        instructions = ttk.Label(
            self.main_frame,
            text="1. Enter working distance (mm) - applies to all CSVs\n"
                 "2. Select CSV files from pair detection\n"
                 "3. Enter mm height for each CSV\n"
                 "4. Adjust filters to see pairs in 3D view\n"
                 "5. Click Calculate to compute magic offset and magic constant\n"
                 "6. Save the calibration data",
            justify="left"
        )
        instructions.pack(pady=(0, 15))
        
        # Global working distance frame
        working_dist_frame = ttk.LabelFrame(self.main_frame, text="Working Distance (Global)", padding="10")
        working_dist_frame.pack(pady=(0, 15), fill="x")
        
        ttk.Label(working_dist_frame, text="Working Distance (mm):").pack(side="left", padx=(0, 10))
        working_dist_entry = ttk.Entry(working_dist_frame, textvariable=self.working_dist_var, width=15)
        working_dist_entry.pack(side="left", padx=(0, 10))
        self.working_dist_entry = working_dist_entry
        
        # Load calibration button
        load_cal_btn = ttk.Button(working_dist_frame, text="📋 Load from Latest Cal", 
                                 command=self.load_from_latest_calibration)
        load_cal_btn.pack(side="left", padx=5)
        
        # Manual load button
        load_manual_btn = ttk.Button(working_dist_frame, text="📂 Load from File", 
                                    command=self.load_calibration_from_file)
        load_manual_btn.pack(side="left", padx=5)
        
        # Filtering controls frame
        filter_frame = ttk.LabelFrame(self.main_frame, text="Filtering Parameters", padding="10")
        filter_frame.pack(pady=(0, 15), fill="x")
        
        # Z filtering section
        z_filter_subframe = ttk.LabelFrame(filter_frame, text="Z-Based Filtering (Near-Mean)", padding="5")
        z_filter_subframe.pack(fill="x", pady=(0, 10))
        
        # Enable Z filtering checkbox
        self.enable_z_filter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(z_filter_subframe, text="Enable Z-based filtering (isolate near-mean pairs first)", 
                       variable=self.enable_z_filter_var, command=self.on_filter_changed).pack(anchor="w")
        
        # Z type selection
        z_type_frame = ttk.Frame(z_filter_subframe)
        z_type_frame.pack(fill="x", pady=(5, 0))
        ttk.Label(z_type_frame, text="Filter by:").pack(side="left", padx=(0, 10))
        self.z_filter_type_var = tk.StringVar(value="Zprime")
        z_type_menu = ttk.OptionMenu(z_type_frame, self.z_filter_type_var, "Zprime", "Zprime", "Zdoubleprime", "Z_mm",
                                     command=lambda x: self.on_filter_changed())
        z_type_menu.pack(side="left")
        
        # Threshold (standard deviations) - slider
        threshold_frame = ttk.Frame(z_filter_subframe)
        threshold_frame.pack(fill="x", pady=(5, 0))
        ttk.Label(threshold_frame, text="Std Dev from Mean:").pack(side="left", padx=(0, 10))
        self.z_filter_threshold_var = tk.DoubleVar(value=2.0)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.1, to=5.0, orient="horizontal",
                                    variable=self.z_filter_threshold_var, length=200, command=self.on_filter_changed)
        threshold_scale.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.z_filter_threshold_label = ttk.Label(threshold_frame, text="2.0σ")
        self.z_filter_threshold_label.pack(side="left")
        threshold_scale.configure(command=lambda v: self.update_z_threshold_label())
        
        # Quality filtering section
        quality_filter_subframe = ttk.LabelFrame(filter_frame, text="Quality Filtering (Pair Score)", padding="5")
        quality_filter_subframe.pack(fill="x", pady=(0, 0))
        
        # Enable quality filtering checkbox
        self.enable_quality_filter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(quality_filter_subframe, text="Enable quality filtering (choose high quality pairs)", 
                       variable=self.enable_quality_filter_var, command=self.on_filter_changed).pack(anchor="w")
        
        # Min S value slider
        min_s_frame = ttk.Frame(quality_filter_subframe)
        min_s_frame.pack(fill="x", pady=(5, 0))
        ttk.Label(min_s_frame, text="Min Pair Score (S):").pack(side="left", padx=(0, 10))
        self.min_score_var = tk.DoubleVar(value=0.9)
        min_s_scale = ttk.Scale(min_s_frame, from_=0.0, to=1.0, orient="horizontal",
                               variable=self.min_score_var, length=200, command=self.on_filter_changed)
        min_s_scale.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.min_score_label = ttk.Label(min_s_frame, text="0.90")
        self.min_score_label.pack(side="left")
        min_s_scale.configure(command=lambda v: self.update_min_score_label())
        
        # Add CSV button
        add_btn = ttk.Button(self.main_frame, text="➕ Add CSV", command=self.add_video_entry)
        add_btn.pack(pady=5)
        
        # Scrollable frame for CSV entries (fixed height container)
        video_entries_container = ttk.LabelFrame(self.main_frame, text="Calibration CSVs", padding="10")
        video_entries_container.pack(fill="x", pady=10)
        
        # Use a fixed-height canvas for video entries (can scroll if many videos)
        canvas_frame = ttk.Frame(video_entries_container)
        canvas_frame.pack(fill="both", expand=True)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white", height=200)
        self.scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Update canvas width when scrollable_frame changes
        def update_canvas_width(event):
            canvas_width = event.width
            self.canvas.itemconfig(canvas_window, width=canvas_width)
        self.canvas.bind('<Configure>', update_canvas_width)
        
        # Bind mousewheel to video entries canvas
        def on_mousewheel_videos(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.canvas.bind("<MouseWheel>", on_mousewheel_videos)
        self.scrollable_frame.bind("<MouseWheel>", on_mousewheel_videos)
        
        # Calculate button
        calc_btn = ttk.Button(self.main_frame, text="Calculate", command=self.calculate)
        calc_btn.pack(pady=10)
        
        # Result label (with max width to prevent excessive expansion)
        result_container = ttk.Frame(self.main_frame)
        result_container.pack(fill="x", pady=10)
        
        self.result_label = ttk.Label(
            result_container,
            text="Calibration: Not calculated",
            justify="left",
            wraplength=750
        )
        self.result_label.pack(fill="x")
    
    def setup_metrics_column(self, parent_frame):
        """Setup the data metrics column."""
        # Metrics display frame (scrollable, fixed height)
        metrics_frame = ttk.LabelFrame(parent_frame, text="Data Metrics", padding="10")
        metrics_frame.pack(fill="both", expand=True, padx=5, pady=5)
        metrics_frame.grid_rowconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(0, weight=1)
        
        # Create scrollable text widget for metrics
        self.metrics_text = tk.Text(metrics_frame, wrap="none", font=("Courier", 8), state="disabled", width=50)
        self.metrics_text_scrollbar = ttk.Scrollbar(metrics_frame, orient="vertical", command=self.metrics_text.yview)
        self.metrics_text.configure(yscrollcommand=self.metrics_text_scrollbar.set)
        
        self.metrics_text.grid(row=0, column=0, sticky="nsew")
        self.metrics_text_scrollbar.grid(row=0, column=1, sticky="ns")
    
    def setup_3d_visualization(self, parent_frame):
        """Setup 3D visualization in the right pane."""
        # Create figure for 3D plot
        self.viz_fig = Figure(figsize=(10, 8), dpi=100)
        self.viz_ax = self.viz_fig.add_subplot(111, projection='3d')
        
        # Create canvas
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, parent_frame)
        self.viz_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.viz_canvas, parent_frame)
        toolbar.update()
        
        # Initial empty plot
        self.viz_ax.set_xlabel('X (mm)')
        self.viz_ax.set_ylabel('Y (mm)')
        self.viz_ax.set_zlabel('Z (mm)')
        self.viz_ax.set_title('Calibration Pairs Visualization')
        self.viz_canvas.draw()
    
    def update_3d_visualization(self):
        """Update 3D visualization with current filtered pairs."""
        if not hasattr(self, 'viz_ax'):
            return
            
        self.viz_ax.clear()
        
        if not self.viz_data:
            self.viz_ax.text(0.5, 0.5, 0.5, "Load CSVs to see calibration pairs", 
                           transform=self.viz_ax.transAxes, ha="center")
            self.viz_canvas.draw()
            return
        
        # Color map for different CSVs
        csv_names = list(self.viz_data.keys())
        colors = cm.tab20(np.linspace(0, 1, max(len(csv_names), 1)))
        csv_color_map = {name: colors[i % len(colors)] for i, name in enumerate(csv_names)}
        
        # Plot all trajectories and highlight calibration pairs
        for csv_name in csv_names:
            data = self.viz_data[csv_name]
            calibration_frames = self.viz_calibration_pairs.get(csv_name, {})
            csv_color = csv_color_map[csv_name]
            
            for track_id, points in data.items():
                if not points:
                    continue
                
                frames = [p[0] for p in points]
                xs = [p[1] for p in points]
                ys = [p[2] for p in points]
                zs = [p[3] for p in points]
                
                # Check if this track has calibration pairs
                has_cal_pairs = track_id in calibration_frames and len(calibration_frames[track_id]) > 0
                
                if has_cal_pairs:
                    # Plot full trajectory as background (light/gray)
                    self.viz_ax.plot(xs, ys, zs, color=csv_color, alpha=0.15, linewidth=0.5, linestyle='--')
                    
                    # Highlight calibration pairs (bright, thick trail)
                    cal_frames = calibration_frames[track_id]
                    cal_indices = sorted([i for i, f in enumerate(frames) if f in cal_frames])
                    
                    if cal_indices:
                        # Create continuous trail segments (handle gaps in frame sequence)
                        segments = []
                        segment_start = cal_indices[0]
                        for i in range(len(cal_indices)):
                            if i == len(cal_indices) - 1 or cal_indices[i+1] - cal_indices[i] > 1:
                                # End of segment
                                segment_end = cal_indices[i]
                                segments.append((segment_start, segment_end))
                                if i < len(cal_indices) - 1:
                                    segment_start = cal_indices[i+1]
                        
                        # Plot each segment as a continuous trail
                        for seg_idx, (seg_start, seg_end) in enumerate(segments):
                            seg_xs = xs[seg_start:seg_end+1]
                            seg_ys = ys[seg_start:seg_end+1]
                            seg_zs = zs[seg_start:seg_end+1]
                            
                            # Plot calibration trail segment (bright, thick)
                            label = f"{csv_name} - Track {track_id}" if seg_idx == 0 else ""
                            self.viz_ax.plot(seg_xs, seg_ys, seg_zs, 
                                           color=csv_color, alpha=0.9, linewidth=4,
                                           label=label, zorder=5)
                        
                        # Mark calibration points with larger markers
                        cal_xs = [xs[i] for i in cal_indices]
                        cal_ys = [ys[i] for i in cal_indices]
                        cal_zs = [zs[i] for i in cal_indices]
                        self.viz_ax.scatter(cal_xs, cal_ys, cal_zs, 
                                          color=csv_color, s=80, marker='o', 
                                          edgecolors='black', linewidths=2, alpha=0.9, zorder=10)
                else:
                    # No calibration pairs for this track, plot very faintly
                    self.viz_ax.plot(xs, ys, zs, color=csv_color, alpha=0.08, linewidth=0.3, linestyle='--')
        
        # Set labels and title based on detected coordinate units
        x_label = f'X ({self.viz_x_unit})' if self.viz_x_unit else 'X'
        y_label = f'Y ({self.viz_y_unit})' if self.viz_y_unit else 'Y'
        z_label = f'Z ({self.viz_z_unit})' if self.viz_z_unit else 'Z'
        self.viz_ax.set_xlabel(x_label)
        self.viz_ax.set_ylabel(y_label)
        self.viz_ax.set_zlabel(z_label)
        self.viz_ax.set_title('Calibration Pairs Visualization\n(Bright trails = Selected pairs)')
        
        # Update chart limits based on all data points
        # Collect all x, y, z values from all tracks
        all_xs = []
        all_ys = []
        all_zs = []
        for csv_name in csv_names:
            data = self.viz_data[csv_name]
            for track_id, points in data.items():
                if points:
                    for p in points:
                        all_xs.append(p[1])  # x coordinate
                        all_ys.append(p[2])  # y coordinate
                        all_zs.append(p[3])  # z coordinate
        
        # Set axis limits if we have data
        if all_xs and all_ys and all_zs:
            x_min, x_max = min(all_xs), max(all_xs)
            y_min, y_max = min(all_ys), max(all_ys)
            z_min, z_max = min(all_zs), max(all_zs)
            
            # Add small padding to limits (5% of range)
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            x_padding = x_range * 0.05 if x_range > 0 else 1.0
            y_padding = y_range * 0.05 if y_range > 0 else 1.0
            z_padding = z_range * 0.05 if z_range > 0 else 1.0
            
            self.viz_ax.set_xlim(x_min - x_padding, x_max + x_padding)
            self.viz_ax.set_ylim(y_min - y_padding, y_max + y_padding)
            self.viz_ax.set_zlim(z_min - z_padding, z_max + z_padding)
        
        # Add legend if not too many tracks
        handles, labels = self.viz_ax.get_legend_handles_labels()
        if handles and len(handles) <= 20:
            self.viz_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        self.viz_ax.grid(True, alpha=0.3)
        self.viz_canvas.draw()
    
    def load_csv_for_visualization(self, csv_path: str, csv_name: str):
        """Load CSV data for 3D visualization."""
        if not os.path.exists(csv_path):
            return
        
        try:
            data = {}  # {track_id: [(frame, x, y, z), ...]}
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Check which columns are available
                has_x_mm = 'X_mm' in reader.fieldnames
                has_y_mm = 'Y_mm' in reader.fieldnames
                has_center_x = 'Center_X' in reader.fieldnames
                has_center_y = 'Center_Y' in reader.fieldnames
                has_z_mm = 'Z_mm' in reader.fieldnames
                has_zprime_mm = 'Zprime_mm' in reader.fieldnames
                has_zdoubleprime = 'Zdoubleprime' in reader.fieldnames
                
                # Read all rows to check data availability
                all_rows = list(reader)
                
                # Determine coordinate units based on available data
                if all_rows:
                    sample_row = all_rows[0]
                    # Check if X_mm/Y_mm have actual data
                    has_x_mm_data = bool(sample_row.get('X_mm', '').strip())
                    has_y_mm_data = bool(sample_row.get('Y_mm', '').strip())
                    
                    # XY labels: "X (mm), Y (mm)" if X_mm/Y_mm has data, otherwise "X (px), Y (px)"
                    if has_x_mm_data and has_y_mm_data:
                        self.viz_x_unit = "mm"
                        self.viz_y_unit = "mm"
                    else:
                        self.viz_x_unit = "px"
                        self.viz_y_unit = "px"
                    
                    # Set Z unit based on available columns with actual data
                    # Priority: Z_mm (from pair_detect or after calibration) > Zprime > Zdoubleprime
                    has_z_mm_data = bool(sample_row.get('Z_mm', '').strip())
                    has_zprime_data = bool(sample_row.get('Zprime_mm', '').strip())
                    has_zdoubleprime_data = bool(sample_row.get('Zdoubleprime', '').strip())
                    
                    if has_z_mm_data:
                        # Z_mm exists (typically calculated by pair_detect when calibration loaded)
                        self.viz_z_unit = "mm"
                    elif has_zprime_data:
                        self.viz_z_unit = ""  # "Z" label (no unit indicator for Zprime)
                    elif has_zdoubleprime_data:
                        self.viz_z_unit = ""  # "Z" label (no unit indicator for Zdoubleprime)
                    else:
                        self.viz_z_unit = "mm"  # Default fallback
                
                # Process rows
                for row in all_rows:
                    try:
                        frame = int(row['Frame_Number'])
                        track_id = int(row['Track_ID'])
                        
                        # Get X, Y coordinates
                        x_str = ''
                        y_str = ''
                        if has_x_mm and has_y_mm:
                            x_str = row.get('X_mm', '').strip()
                            y_str = row.get('Y_mm', '').strip()
                        if (not x_str or not y_str) and has_center_x and has_center_y:
                            x_str = row.get('Center_X', '').strip()
                            y_str = row.get('Center_Y', '').strip()
                        
                        if not x_str or not y_str:
                            continue
                        
                        x = float(x_str)
                        y = float(y_str)
                        
                        # Get Z coordinate (priority: Z_mm from pair_detect > calculated Z_mm > Zprime > Zdoubleprime)
                        z = 0.0
                        z_str = ''
                        # First check if Z_mm exists (may have been calculated by pair_detect)
                        if has_z_mm:
                            z_str = row.get('Z_mm', '').strip()
                            if z_str:
                                z = float(z_str)
                                # Use Z_mm if available (pre-calculated by pair_detect)
                        if not z_str and has_zprime_mm:
                            z_str = row.get('Zprime_mm', '').strip()
                            if z_str:
                                z = float(z_str)
                        if not z_str and has_zdoubleprime:
                            z_str = row.get('Zdoubleprime', '').strip()
                            if z_str:
                                z = float(z_str)
                        
                        if track_id not in data:
                            data[track_id] = []
                        data[track_id].append((frame, x, y, z))
                        
                    except (ValueError, KeyError):
                        continue
            
            self.viz_data[csv_name] = data
            
        except Exception as e:
            print(f"[WARN] Failed to load CSV for visualization: {e}")
    
    def update_z_threshold_label(self):
        """Update Z threshold label."""
        val = self.z_filter_threshold_var.get()
        self.z_filter_threshold_label.config(text=f"{val:.2f}σ")
        self.on_filter_changed()
    
    def update_min_score_label(self):
        """Update min score label."""
        val = self.min_score_var.get()
        self.min_score_label.config(text=f"{val:.2f}")
        self.on_filter_changed()
    
    def on_filter_changed(self, value=None):
        """Called when any filter parameter changes - update live metrics."""
        # Only update if we have data loaded
        if hasattr(self, 'video_entries') and len(self.video_entries) > 0:
            # Check if at least one CSV is loaded
            has_data = False
            for entry in self.video_entries:
                if entry.csv_path:
                    has_data = True
                    break
            
            if has_data:
                # Recalculate and update metrics without running full calibration
                # This also updates 3D visualization
                self.update_live_metrics()
    
    def update_live_metrics(self):
        """Update metrics display in real-time as filter parameters change."""
        # Only update if metrics_text widget exists (GUI fully initialized)
        if not hasattr(self, 'metrics_text') or self.metrics_text is None:
            return
        
        # Collect metrics for all loaded CSVs (this also updates calibration pairs)
        all_input_metrics = []
        all_chosen_metrics = []
        all_z_filter_stats = []
        
        # Clear existing calibration pairs - they will be regenerated
        self.viz_calibration_pairs.clear()
        
        for entry in self.video_entries:
            result = entry.get_data()
            if result:
                csv_path, mm_height, working_dist, avg_zprime, avg_b, input_metrics, chosen_metrics, z_filter_stats = result
                all_input_metrics.append((os.path.basename(csv_path), input_metrics))
                all_chosen_metrics.append((os.path.basename(csv_path), chosen_metrics))
                if z_filter_stats:
                    all_z_filter_stats.append((os.path.basename(csv_path), z_filter_stats))
        
        # Calculate totals
        total_input = sum(m["count"] for _, m in all_input_metrics)
        total_chosen = sum(m["count"] for _, m in all_chosen_metrics)
        
        # Update the metrics display
        self.display_metrics(all_input_metrics, all_chosen_metrics, total_input, total_chosen, all_z_filter_stats)
        
        # Update 3D visualization with new calibration pairs
        self.update_3d_visualization()
    
    def _auto_load_from_latest_calibration(self):
        """Automatically load working distance from the latest calibration file."""
        # Only auto-load if working distance field is empty
        current_value = self.working_dist_var.get().strip()
        if current_value:
            return  # Don't overwrite existing value
        
        latest_cal_file = get_latest_calibration_file()
        if latest_cal_file:
            if self._load_working_distance_from_json_silent(latest_cal_file):
                print(f"[INFO] Auto-loaded working distance from latest calibration: {latest_cal_file}")
    
    def _load_working_distance_from_json_silent(self, file_path: str) -> bool:
        """
        Load working distance from JSON file silently (no messageboxes).
        Returns True if working distance was found and loaded.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cal_data = json.load(f)
            
            working_dist = extract_working_distance(cal_data)
            
            if working_dist is not None and working_dist > 0:
                self.working_dist_var.set(str(working_dist))
                return True
            return False
                
        except Exception:
            return False
    
    def load_from_latest_calibration(self):
        """Load working distance from the latest calibration file."""
        latest_cal_file = get_latest_calibration_file()
        if not latest_cal_file:
            messagebox.showwarning("Warning", "No calibration files found in the calibrations folder.")
            return
        
        try:
            with open(latest_cal_file, 'r', encoding='utf-8') as f:
                cal_data = json.load(f)
            
            working_dist = extract_working_distance(cal_data)
            
            if working_dist is not None and working_dist > 0:
                self.working_dist_var.set(str(working_dist))
                messagebox.showinfo("Success", f"Loaded working distance: {working_dist:.2f} mm\nfrom: {os.path.basename(latest_cal_file)}")
            else:
                messagebox.showwarning("Warning", "No valid working distance found in calibration file.\nPlease enter working distance manually.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration file: {e}")
    
    def load_calibration_from_file(self):
        """Load working distance from a selected image calibration JSON file."""
        file_path = filedialog.askopenfilename(
            title="Load Image Calibration File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path or not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cal_data = json.load(f)
            
            working_dist = extract_working_distance(cal_data)
            
            if working_dist is not None and working_dist > 0:
                self.working_dist_var.set(str(working_dist))
                messagebox.showinfo("Success", f"Loaded working distance: {working_dist:.2f} mm")
            else:
                messagebox.showwarning("Warning", "No valid working distance found in calibration file.\nPlease enter working distance manually.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration file: {e}")
    
    def add_video_entry(self):
        """Add a new CSV entry row."""
        entry_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text=f"CSV {len(self.video_entries) + 1}",
            padding="10"
        )
        entry_frame.pack(fill="x", padx=10, pady=5)
        entry_frame.grid_columnconfigure(1, weight=1)
        
        entry = VideoEntry(entry_frame, len(self.video_entries))
        entry.set_app(self)
        self.video_entries.append(entry)
        
        # Update canvas scroll region
        self.root.update_idletasks()
        if self.canvas is not None:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def remove_video_entry(self, entry: VideoEntry):
        """Remove a CSV entry."""
        if len(self.video_entries) <= 2:
            messagebox.showwarning(
                "Warning",
                "You need at least 2 CSVs for calibration.\n"
                "Cannot remove this entry."
            )
            return
        
        if entry in self.video_entries:
            entry.frame.destroy()
            self.video_entries.remove(entry)
            # Renumber remaining entries
            for i, e in enumerate(self.video_entries):
                e.row = i
                e.frame.config(text=f"CSV {i + 1}")
            # Update canvas scroll region
            self.root.update_idletasks()
            if self.canvas is not None:
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def calculate(self):
        """Calculate magic offset and magic constant using linear regression on Zprime values."""
        # Collect valid data points with metrics
        data_points = []
        all_input_metrics = []
        all_chosen_metrics = []
        
        all_z_filter_stats = []
        for entry in self.video_entries:
            result = entry.get_data()
            if result:
                csv_path, mm_height, working_dist, avg_zprime, avg_b, input_metrics, chosen_metrics, z_filter_stats = result
                data_points.append((csv_path, mm_height, working_dist, avg_zprime, avg_b))
                all_input_metrics.append((os.path.basename(csv_path), input_metrics))
                all_chosen_metrics.append((os.path.basename(csv_path), chosen_metrics))
                if z_filter_stats:
                    all_z_filter_stats.append((os.path.basename(csv_path), z_filter_stats))
                
                filter_info = ""
                if z_filter_stats:
                    filter_info = f", Z filter: {z_filter_stats['pairs_before']} → {z_filter_stats['pairs_after']} pairs (omitted {z_filter_stats['omitted']})"
                
                print(f"[INFO] CSV: {os.path.basename(csv_path)}, "
                      f"Height: {mm_height:.2f} mm, "
                      f"Working Dist: {working_dist:.2f} mm, "
                      f"Avg Zprime: {avg_zprime:.4f}, "
                      f"Avg B: {avg_b:.4f} px{filter_info}")
        
        if len(data_points) < 2:
            messagebox.showerror(
                "Error",
                f"Need at least 2 valid CSVs with:\n"
                "- Selected CSV file\n"
                "- Valid mm height > 0\n"
                "- Valid global working distance > 0\n"
                f"Currently have: {len(data_points)} valid entries"
            )
            return
        
        # Check for insufficient data warnings
        insufficient_data_warnings = []
        for entry in self.video_entries:
            result = entry.get_data()
            if result:
                csv_path, mm_height, working_dist, avg_zprime, avg_b, input_metrics, chosen_metrics, z_filter_stats = result
                if chosen_metrics["count"] < 10:
                    insufficient_data_warnings.append(
                        f"{os.path.basename(csv_path)}: Only {chosen_metrics['count']} pairs after filtering "
                        f"(need at least 10 for good calibration)"
                    )
        
        if insufficient_data_warnings:
            warning_msg = "WARNING: Insufficient data for good calibration:\n\n" + "\n".join(insufficient_data_warnings)
            warning_msg += "\n\nPlease adjust filter thresholds or check your input data."
            messagebox.showwarning("Insufficient Data", warning_msg)
        
        # Extract Zprime values, B values, and Z (calibrated mm height) values
        zprimes = np.array([z for _, _, _, z, _ in data_points])
        b_values = np.array([b for _, _, _, _, b in data_points])
        z_values = np.array([h for _, h, _, _, _ in data_points])  # Z = calibrated mm height input
        
        # Linear regression: Z = Zprime * magic_constant + magic_offset
        # Using np.polyfit (degree 1) or manual calculation
        # np.polyfit returns [slope, intercept] for degree 1
        # We fit: Z = slope * Zprime + intercept
        coeffs = np.polyfit(zprimes, z_values, 1)
        self.magic_constant = coeffs[0]  # slope
        self.magic_offset = coeffs[1]    # intercept
        
        # Calculate R² for quality assessment
        predicted_z = self.magic_constant * zprimes + self.magic_offset
        ss_res = np.sum((z_values - predicted_z) ** 2)
        ss_tot = np.sum((z_values - np.mean(z_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate average working distance from data points
        working_distances = [wd for _, _, wd, _, _ in data_points]
        avg_working_distance_mm = np.mean(working_distances) if len(working_distances) > 0 else None
        
        # Calculate aggregate metrics
        total_input_count = sum(m["count"] for _, m in all_input_metrics)
        total_chosen_count = sum(m["count"] for _, m in all_chosen_metrics)
        
        # Create dictionary for looking up input metrics by video name (for percentage calculation)
        input_video_dict = {name: m for name, m in all_input_metrics}
        
        self.calibration_data = {
            "magic_constant": float(self.magic_constant),
            "magic_offset": float(self.magic_offset),
            "r_squared": float(r_squared),
            "data_points": [
                {
                    "csv_path": csv_path,
                    "z_mm": float(h),  # Z = calibrated input value
                    "working_distance_mm": float(wd),
                    "avg_zprime": float(z),
                    "avg_b": float(b)
                }
                for csv_path, h, wd, z, b in data_points
            ],
            "formula": "Z = Zprime * magic_constant + magic_offset",
            "description": "Z is the calibrated mm height input, Zprime is calculated from pair data",
            "zprime_formula": "Zprime = working_distance * (C-A)/(A+C)",
            "b_formula": "B = (2*A*C)/(A+C)",
            "metrics": {
                "input_dataset": {
                    "total_pairs": total_input_count,
                    "per_video": [
                        {
                            "video": name,
                            "count": m["count"],
                            "zprime_mean": m["zprime"]["mean"],
                            "zprime_std": m["zprime"]["std"],
                            "zprime_min": m["zprime"]["min"],
                            "zprime_max": m["zprime"]["max"],
                            "b_mean": m["b"]["mean"],
                            "b_std": m["b"]["std"],
                            "b_min": m["b"]["min"],
                            "b_max": m["b"]["max"],
                            "score_mean": m["score"]["mean"],
                            "score_std": m["score"]["std"],
                            "score_min": m["score"]["min"],
                            "score_max": m["score"]["max"]
                        }
                        for name, m in all_input_metrics
                    ]
                },
                "chosen_dataset": {
                    "total_pairs": total_chosen_count,
                    "percent_of_input": float(100 * total_chosen_count / max(1, total_input_count)),
                    "per_video": [
                        {
                            "video": name,
                            "count": m["count"],
                            "percent_of_input": float(100 * m["count"] / max(1, input_video_dict.get(name, {}).get("count", 0))) if name in input_video_dict else 0.0,
                            "zprime_mean": m["zprime"]["mean"],
                            "zprime_std": m["zprime"]["std"],
                            "zprime_min": m["zprime"]["min"],
                            "zprime_max": m["zprime"]["max"],
                            "b_mean": m["b"]["mean"],
                            "b_std": m["b"]["std"],
                            "b_min": m["b"]["min"],
                            "b_max": m["b"]["max"],
                            "score_mean": m["score"]["mean"],
                            "score_std": m["score"]["std"],
                            "score_min": m["score"]["min"],
                            "score_max": m["score"]["max"]
                        }
                        for name, m in all_chosen_metrics
                    ]
                }
            }
        }
        
        # Add average working distance to calibration data if available
        if avg_working_distance_mm is not None:
            self.calibration_data["working_distance_mm"] = float(avg_working_distance_mm)
        
        # Calculate average B for display
        avg_b_all = np.mean(b_values) if len(b_values) > 0 else 0
        
        # Build result text with Z filtering info
        z_filter_summary = ""
        if all_z_filter_stats:
            total_omitted = sum(stats['omitted'] for _, stats in all_z_filter_stats)
            z_filter_summary = f"\nZ Filtering: Omitted {total_omitted} outlier pairs across all videos"
        
        # Update result label
        result_text = (
            f"Calibration Complete!\n\n"
            f"Magic Constant: {self.magic_constant:.6f}\n"
            f"Magic Offset: {self.magic_offset:.6f} mm\n"
            f"R² (quality): {r_squared:.4f}\n"
            f"Avg B: {avg_b_all:.4f} px\n\n"
            f"Formula: Z = Zprime * {self.magic_constant:.6f} + {self.magic_offset:.6f}\n"
            f"where:\n"
            f"  Z = calibrated mm height (input)\n"
            f"  Zprime = working_distance * (C-A)/(A+C)\n"
            f"  B = (2*A*C)/(A+C)\n\n"
            f"Input Dataset: {total_input_count} pairs | "
            f"Chosen Dataset: {total_chosen_count} pairs ({100*total_chosen_count/max(1,total_input_count):.1f}%)"
            f"{z_filter_summary}"
        )
        self.result_label.config(text=result_text)
        
        # Display detailed metrics
        self.display_metrics(all_input_metrics, all_chosen_metrics, total_input_count, total_chosen_count, all_z_filter_stats)
        
        print(f"[INFO] Calibration calculated:")
        print(f"  Formula: Z = Zprime * magic_constant + magic_offset")
        print(f"  Magic Constant: {self.magic_constant:.6f}")
        print(f"  Magic Offset: {self.magic_offset:.6f} mm")
        print(f"  R²: {r_squared:.4f}")
        print(f"  Input Dataset: {total_input_count} pairs")
        print(f"  Chosen Dataset: {total_chosen_count} pairs ({100*total_chosen_count/max(1,total_input_count):.1f}%)")
        
        # Automatically save to calibrations folder
        self._auto_save_calibration()
        
        # Ask user if they want to save calculated mm values to CSV files
        response = messagebox.askyesno(
            "Save Calculated Values?",
            "Calibration complete!\n\n"
            "Would you like to save the calculated values (Z_mm, B_px, B_mm, X_mm, Y_mm) "
            "back to the CSV files?\n\n"
            "Yes - Update CSV files with all calculated mm values\n"
            "No - Keep CSV files unchanged"
        )
        
        if response:
            # Update CSV files with all calculated mm values
            # Each CSV is reloaded after update, so visualization data is automatically updated
            self._update_csv_files_with_mm_values()
            # Also convert visualization data to Z_mm as a backup (in case some CSVs weren't updated)
            # This ensures all visualization data uses the latest calibration constants
            self._convert_visualization_to_zmm()
            # Update 3D visualization with calibrated Z_mm values
            self.update_3d_visualization()
        else:
            # Still update visualization for display, but don't save to CSV
            self._convert_visualization_to_zmm()
            self.update_3d_visualization()
    
    def display_metrics(self, input_metrics_list, chosen_metrics_list, total_input, total_chosen, z_filter_stats_list=None):
        """Display detailed metrics for input and chosen datasets."""
        self.metrics_text.config(state="normal")
        self.metrics_text.delete(1.0, tk.END)
        
        # Calculate percentages for each video
        input_video_dict = {name: m for name, m in input_metrics_list}
        z_filter_dict = {name: stats for name, stats in (z_filter_stats_list or [])}
        
        # Header
        self.metrics_text.insert(tk.END, "="*80 + "\n")
        self.metrics_text.insert(tk.END, "DATASET METRICS\n")
        self.metrics_text.insert(tk.END, "="*80 + "\n\n")
        
        # Input dataset totals
        self.metrics_text.insert(tk.END, f"INPUT DATASET (Total: {total_input} pairs)\n")
        self.metrics_text.insert(tk.END, "-"*80 + "\n")
        
        # Per-video input metrics
        for name, m in input_metrics_list:
            self.metrics_text.insert(tk.END, f"\n  {name}:\n")
            self.metrics_text.insert(tk.END, f"    Pairs: {m['count']}\n")
            self.metrics_text.insert(tk.END, f"    Zprime: mean={m['zprime']['mean']:.4f}, std={m['zprime']['std']:.4f}, "
                                           f"range=[{m['zprime']['min']:.4f}, {m['zprime']['max']:.4f}]\n")
            self.metrics_text.insert(tk.END, f"    B:      mean={m['b']['mean']:.4f}, std={m['b']['std']:.4f}, "
                                           f"range=[{m['b']['min']:.4f}, {m['b']['max']:.4f}]\n")
            self.metrics_text.insert(tk.END, f"    Score:  mean={m['score']['mean']:.4f}, std={m['score']['std']:.4f}, "
                                           f"range=[{m['score']['min']:.4f}, {m['score']['max']:.4f}]\n")
        
        # Chosen dataset totals
        overall_percent = 100 * total_chosen / max(1, total_input)
        self.metrics_text.insert(tk.END, f"\n{'='*80}\n")
        self.metrics_text.insert(tk.END, f"CHOSEN DATASET (Total: {total_chosen} pairs, {overall_percent:.1f}% of input)\n")
        self.metrics_text.insert(tk.END, "-"*80 + "\n")
        
        # Per-video chosen metrics with percentages
        for name, m in chosen_metrics_list:
            input_count = input_video_dict.get(name, {}).get("count", 0)
            video_percent = 100 * m['count'] / max(1, input_count) if input_count > 0 else 0.0
            self.metrics_text.insert(tk.END, f"\n  {name}:\n")
            self.metrics_text.insert(tk.END, f"    Pairs: {m['count']} ({video_percent:.1f}% of input)\n")
            self.metrics_text.insert(tk.END, f"    Zprime: mean={m['zprime']['mean']:.4f}, std={m['zprime']['std']:.4f}, "
                                           f"range=[{m['zprime']['min']:.4f}, {m['zprime']['max']:.4f}]\n")
            self.metrics_text.insert(tk.END, f"    B:      mean={m['b']['mean']:.4f}, std={m['b']['std']:.4f}, "
                                           f"range=[{m['b']['min']:.4f}, {m['b']['max']:.4f}]\n")
            self.metrics_text.insert(tk.END, f"    Score:  mean={m['score']['mean']:.4f}, std={m['score']['std']:.4f}, "
                                           f"range=[{m['score']['min']:.4f}, {m['score']['max']:.4f}]\n")
            
            # Add Z filtering info if available
            if name in z_filter_dict:
                zf = z_filter_dict[name]
                self.metrics_text.insert(tk.END, f"\n    Z Filtering ({zf['filter_type']}):\n")
                self.metrics_text.insert(tk.END, f"      Target Z (mean): {zf['mean_z']:.4f}\n")
                self.metrics_text.insert(tk.END, f"      Std Dev: {zf['std_z']:.4f}\n")
                self.metrics_text.insert(tk.END, f"      Threshold: ±{zf['threshold_std']:.2f}σ\n")
                self.metrics_text.insert(tk.END, f"      Pairs: {zf['pairs_before']} → {zf['pairs_after']} (omitted {zf['omitted']})\n")
        
        self.metrics_text.config(state="disabled")
    
    def _convert_visualization_to_zmm(self):
        """
        Convert all visualization data to Z_mm using calibration constants.
        This is called after calibration is complete to update the 3D plot with calibrated Z values.
        """
        if self.magic_constant is None or self.magic_offset is None:
            return  # Calibration not complete yet
        
        # Get working distance
        try:
            working_dist_val = float(self.working_dist_var.get())
            if working_dist_val <= 0:
                return
        except (ValueError, tk.TclError):
            return
        
        # Find CSV paths for all loaded visualization data
        csv_path_map = {}  # {csv_name: csv_path}
        for entry in self.video_entries:
            if entry.csv_path and os.path.exists(entry.csv_path):
                csv_name = os.path.basename(entry.csv_path)
                csv_path_map[csv_name] = entry.csv_path
        
        # Convert each CSV's visualization data
        for csv_name, csv_path in csv_path_map.items():
            if csv_name not in self.viz_data:
                continue
            
            # Load CSV to get A and C values for calculating Zprime
            try:
                track_data_map = {}  # {track_id: {frame: (r_a, r_c)}}
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            track_id = int(row.get('Track_ID', 0))
                            frame = int(row.get('Frame_Number', 0))
                            r_a = float(row.get("A_px", row.get("Radius_A_px", 0)))
                            r_c = float(row.get("C_px", row.get("Radius_B_px", 0)))
                            
                            if track_id > 0 and frame > 0 and r_a > 0 and r_c > 0:
                                if track_id not in track_data_map:
                                    track_data_map[track_id] = {}
                                track_data_map[track_id][frame] = (r_a, r_c)
                        except (ValueError, KeyError):
                            continue
                
                # Update visualization data with Z_mm values
                data = self.viz_data[csv_name]
                for track_id, points in data.items():
                    if track_id not in track_data_map:
                        continue
                    
                    frame_map = track_data_map[track_id]
                    updated_points = []
                    
                    for frame, x, y, z_old in points:
                        if frame in frame_map:
                            r_a, r_c = frame_map[frame]
                            if r_a + r_c > 0:
                                # Calculate Zprime = working_distance * (C-A)/(A+C)
                                zprime = working_dist_val * (r_c - r_a) / (r_a + r_c)
                                # Convert to Z_mm = Zprime * magic_constant + magic_offset
                                z_mm = zprime * self.magic_constant + self.magic_offset
                                updated_points.append((frame, x, y, z_mm))
                            else:
                                updated_points.append((frame, x, y, z_old))
                        else:
                            # Frame not found, keep old Z value
                            updated_points.append((frame, x, y, z_old))
                    
                    data[track_id] = updated_points
                
                # Update Z unit label to mm
                self.viz_z_unit = "mm"
                
            except Exception as e:
                print(f"[WARN] Failed to convert {csv_name} to Z_mm: {e}")
                continue
    
    def _update_csv_files_with_mm_values(self):
        """
        Update CSV files on disk with calculated mm values (Z_mm, B_px, B_mm, X_mm, Y_mm).
        Uses the current calibration constants and tries to load optical center and pixels_per_mm
        from preset and calibration files.
        """
        if self.magic_constant is None or self.magic_offset is None:
            return  # Calibration not complete yet
        
        # Get working distance
        try:
            working_dist_val = float(self.working_dist_var.get())
            if working_dist_val <= 0:
                return
        except (ValueError, tk.TclError):
            return
        
        # Try to get pixels_per_mm from preset file, if not there, load from image calibration and save it to preset
        pixels_per_mm = None
        
        # Update each CSV file
        for entry in self.video_entries:
            if not entry.csv_path or not os.path.exists(entry.csv_path):
                continue
            
            csv_path = entry.csv_path
            csv_dir = os.path.dirname(csv_path)
            csv_basename = os.path.basename(csv_path)
            csv_name_no_ext = os.path.splitext(csv_basename)[0]
            
            # Try to load optical center and pixels_per_mm from preset file in same directory
            x_center = None
            y_center = None
            preset_path = os.path.join(csv_dir, "pair_detect_preset.json")
            preset_needs_save = False
            if os.path.exists(preset_path):
                try:
                    with open(preset_path, 'r', encoding='utf-8') as f:
                        preset_data = json.load(f)
                    if "center" in preset_data:
                        center_data = preset_data["center"]
                        if center_data.get("valid") and center_data.get("x") is not None and center_data.get("y") is not None:
                            x_center = float(center_data["x"])
                            y_center = float(center_data["y"])
                    
                    # Get pixels_per_mm from preset calibration data
                    if "calibration" in preset_data and preset_data["calibration"]:
                        calib_data = preset_data["calibration"]
                        if calib_data.get("pixels_per_mm") is not None:
                            pixels_per_mm = float(calib_data["pixels_per_mm"])
                except Exception as e:
                    print(f"[WARN] Could not load optical center or pixels_per_mm from preset: {e}")
            
            # If pixels_per_mm not in preset, try to load from image calibration and save to preset
            if pixels_per_mm is None or pixels_per_mm <= 0:
                try:
                    calibrations_dir = Path("calibrations")
                    if calibrations_dir.exists():
                        image_cal_files = list(calibrations_dir.glob("image_calibration_*.json"))
                        if image_cal_files:
                            image_cal_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                            latest_image_cal = image_cal_files[0]
                            with open(latest_image_cal, 'r', encoding='utf-8') as f:
                                image_cal_data = json.load(f)
                            pixels_per_mm = extract_pixels_per_mm(image_cal_data)
                            
                            # Save pixels_per_mm to preset file if we found it
                            if pixels_per_mm is not None and pixels_per_mm > 0:
                                if preset_path and os.path.exists(preset_path):
                                    try:
                                        with open(preset_path, 'r', encoding='utf-8') as f:
                                            preset_data = json.load(f)
                                        if "calibration" not in preset_data:
                                            preset_data["calibration"] = {}
                                        preset_data["calibration"]["pixels_per_mm"] = float(pixels_per_mm)
                                        with open(preset_path, 'w', encoding='utf-8') as f:
                                            json.dump(preset_data, f, indent=2)
                                        print(f"[INFO] Saved pixels_per_mm ({pixels_per_mm:.4f}) to preset file")
                                    except Exception as e:
                                        print(f"[WARN] Could not save pixels_per_mm to preset: {e}")
                except Exception as e:
                    print(f"[WARN] Could not load pixels_per_mm from image calibration: {e}")
            
            # If no preset, try to estimate from CSV data (use frame center as fallback)
            if x_center is None or y_center is None:
                # Estimate from first row's data - use a reasonable default
                # We'll calculate relative to (0,0) or use a default center if needed
                # For now, we'll skip X_mm/Y_mm if center is not available
                pass
            
            try:
                # Read CSV
                rows = []
                fieldnames = None
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    fieldnames = list(reader.fieldnames) if reader.fieldnames else []
                    rows = list(reader)
                
                if not fieldnames or not rows:
                    continue
                
                # Ensure all required columns exist
                new_columns = []
                if 'Z_mm' not in fieldnames:
                    new_columns.append('Z_mm')
                if 'B_px' not in fieldnames:
                    new_columns.append('B_px')
                if 'B_mm' not in fieldnames and pixels_per_mm is not None:
                    new_columns.append('B_mm')
                if 'X_mm' not in fieldnames and x_center is not None and y_center is not None and pixels_per_mm is not None:
                    new_columns.append('X_mm')
                if 'Y_mm' not in fieldnames and x_center is not None and y_center is not None and pixels_per_mm is not None:
                    new_columns.append('Y_mm')
                
                if new_columns:
                    fieldnames = fieldnames + new_columns
                
                # Calculate values for each row
                # Only fill empty values by default (preserve existing values)
                updated_count = 0
                overwritten_count = 0
                for row in rows:
                    try:
                        # Get A and C values
                        r_a = float(row.get("A_px", row.get("Radius_A_px", 0)))
                        r_c = float(row.get("C_px", row.get("Radius_B_px", 0)))
                        
                        if r_a > 0 and r_c > 0 and (r_a + r_c) > 0:
                            # Calculate Zprime and Z_mm
                            # Only overwrite if empty (preserve old calibration values)
                            existing_z_mm = row.get('Z_mm', '').strip()
                            if not existing_z_mm:
                                zprime = working_dist_val * (r_c - r_a) / (r_a + r_c)
                                z_mm = zprime * self.magic_constant + self.magic_offset
                                row['Z_mm'] = f"{z_mm:.4f}"
                                updated_count += 1
                            else:
                                # Overwrite existing value after new calibration (user confirmed via popup)
                                zprime = working_dist_val * (r_c - r_a) / (r_a + r_c)
                                z_mm = zprime * self.magic_constant + self.magic_offset
                                row['Z_mm'] = f"{z_mm:.4f}"
                                overwritten_count += 1
                            
                            # Calculate B_px (overwrite after calibration, or fill if empty)
                            existing_b_px = row.get('B_px', '').strip()
                            b_px = calculate_b_px(r_a, r_c)
                            if b_px is not None:
                                # After calibration, always overwrite mm values
                                row['B_px'] = f"{b_px:.4f}"
                                
                                # Calculate B_mm if pixels_per_mm is available
                                if pixels_per_mm is not None:
                                    existing_b_mm = row.get('B_mm', '').strip()
                                    b_mm = calculate_b_mm(b_px, pixels_per_mm)
                                    if b_mm is not None:
                                        # After calibration, always overwrite mm values
                                        row['B_mm'] = f"{b_mm:.4f}"
                                        
                                        # Calculate X_mm and Y_mm if center is available
                                        if x_center is not None and y_center is not None:
                                            # Get pair midpoint from CSV
                                            try:
                                                midpoint_x_str = row.get("Center_X", "").strip()
                                                midpoint_y_str = row.get("Center_Y", "").strip()
                                                
                                                if midpoint_x_str and midpoint_y_str:
                                                    midpoint_x = float(midpoint_x_str)
                                                    midpoint_y = float(midpoint_y_str)
                                                    
                                                    # Calculate X_mm and Y_mm using midpoint and optical center
                                                    x_mm, y_mm = calculate_xy_mm(
                                                        midpoint_x, midpoint_y,
                                                        x_center, y_center,
                                                        b_mm
                                                    )
                                                    
                                                    # After calibration, always overwrite mm values
                                                    existing_x_mm = row.get('X_mm', '').strip()
                                                    existing_y_mm = row.get('Y_mm', '').strip()
                                                    
                                                    if x_mm is not None:
                                                        row['X_mm'] = f"{x_mm:.4f}"
                                                        if existing_x_mm and existing_x_mm != row['X_mm']:
                                                            overwritten_count += 1
                                                    if y_mm is not None:
                                                        row['Y_mm'] = f"{y_mm:.4f}"
                                                        if existing_y_mm and existing_y_mm != row['Y_mm']:
                                                            overwritten_count += 1
                                            except (ValueError, KeyError):
                                                pass
                                        
                                        # Track overwrites for B_mm
                                        if existing_b_mm and existing_b_mm != row['B_mm']:
                                            overwritten_count += 1
                                # Track overwrites for B_px
                                if existing_b_px and existing_b_px != row['B_px']:
                                    overwritten_count += 1
                            else:
                                # Only set empty if column doesn't exist or is empty
                                if 'B_px' in fieldnames and not row.get('B_px', '').strip():
                                    row['B_px'] = ""
                                if 'B_mm' in fieldnames and not row.get('B_mm', '').strip():
                                    row['B_mm'] = ""
                                if 'X_mm' in fieldnames and not row.get('X_mm', '').strip():
                                    row['X_mm'] = ""
                                if 'Y_mm' in fieldnames and not row.get('Y_mm', '').strip():
                                    row['Y_mm'] = ""
                            
                            if not existing_z_mm:
                                updated_count += 1
                        else:
                            # No valid pair data, leave all values empty
                            row['Z_mm'] = ""
                            if 'B_px' in fieldnames:
                                row['B_px'] = ""
                            if 'B_mm' in fieldnames:
                                row['B_mm'] = ""
                            if 'X_mm' in fieldnames:
                                row['X_mm'] = ""
                            if 'Y_mm' in fieldnames:
                                row['Y_mm'] = ""
                    except (ValueError, KeyError) as e:
                        # Leave all values empty on error
                        row['Z_mm'] = ""
                        if 'B_px' in fieldnames:
                            row['B_px'] = ""
                        if 'B_mm' in fieldnames:
                            row['B_mm'] = ""
                        if 'X_mm' in fieldnames:
                            row['X_mm'] = ""
                        if 'Y_mm' in fieldnames:
                            row['Y_mm'] = ""
                        continue
                
                # Write updated CSV back to file
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                
                values_list = []
                if 'Z_mm' in fieldnames:
                    values_list.append("Z_mm")
                if 'B_px' in fieldnames:
                    values_list.append("B_px")
                if 'B_mm' in fieldnames:
                    values_list.append("B_mm")
                if 'X_mm' in fieldnames:
                    values_list.append("X_mm")
                if 'Y_mm' in fieldnames:
                    values_list.append("Y_mm")
                
                update_msg = f"[INFO] Updated CSV {csv_basename} with {', '.join(values_list)}"
                if updated_count > 0:
                    update_msg += f" (filled {updated_count} empty values"
                if overwritten_count > 0:
                    if updated_count > 0:
                        update_msg += ", "
                    else:
                        update_msg += " ("
                    update_msg += f"overwritten {overwritten_count} existing values after calibration"
                if updated_count > 0 or overwritten_count > 0:
                    update_msg += ")"
                print(update_msg)
                
                # Show popup if values were overwritten
                if overwritten_count > 0:
                    messagebox.showinfo(
                        "CSV Updated",
                        f"Updated CSV file: {csv_basename}\n\n"
                        f"Filled {updated_count} empty values\n"
                        f"Overwritten {overwritten_count} existing mm values with new calibration"
                    )
                
                # Reload visualization data to use the new values
                self._reload_csv_for_visualization(csv_path, csv_basename)
                
                # Look for associated JSON metadata file and update it
                json_candidates = [
                    os.path.join(csv_dir, csv_name_no_ext + ".json"),
                    os.path.join(csv_dir, csv_name_no_ext + "_metadata.json"),
                    os.path.join(csv_dir, "metadata.json"),
                ]
                
                for json_path in json_candidates:
                    if os.path.exists(json_path):
                        try:
                            # Read existing JSON
                            with open(json_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                            
                            # Update JSON with calibration info
                            if 'calibration' not in json_data:
                                json_data['calibration'] = {}
                            
                            json_data['calibration']['magic_constant'] = float(self.magic_constant)
                            json_data['calibration']['magic_offset'] = float(self.magic_offset)
                            json_data['calibration']['working_distance_mm'] = float(working_dist_val)
                            json_data['calibration']['formula'] = "Z_mm = Zprime * magic_constant + magic_offset"
                            
                            if pixels_per_mm is not None:
                                json_data['calibration']['pixels_per_mm'] = float(pixels_per_mm)
                            
                            # Add calibration timestamp
                            json_data['calibration']['calibrated_at'] = datetime.now().isoformat()
                            
                            # Update if calibration data exists
                            if self.calibration_data:
                                if 'r_squared' in self.calibration_data:
                                    json_data['calibration']['r_squared'] = float(self.calibration_data['r_squared'])
                            
                            # Write updated JSON back
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(json_data, f, indent=2)
                            
                            print(f"[INFO] Updated associated JSON metadata: {os.path.basename(json_path)}")
                            break  # Only update first found JSON file
                        except Exception as e:
                            print(f"[WARN] Failed to update JSON {json_path}: {e}")
                            continue
                
            except Exception as e:
                print(f"[WARN] Failed to update CSV {csv_basename}: {e}")
                continue
    
    def _reload_csv_for_visualization(self, csv_path: str, csv_name: str):
        """Reload CSV data for visualization after Z_mm has been updated."""
        # Reload the CSV to pick up new Z_mm values
        self.load_csv_for_visualization(csv_path, csv_name)
    
    def _auto_save_calibration(self):
        """Automatically save calibration data to the calibrations folder."""
        if self.calibration_data is None:
            return
        
        # Create calibrations folder if it doesn't exist
        calibrations_dir = Path("calibrations")
        calibrations_dir.mkdir(exist_ok=True)
        
        # Generate timestamped filename combining all CSV filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_names = []
        if self.calibration_data.get("data_points"):
            for point in self.calibration_data["data_points"]:
                csv_path = point.get("csv_path", "")
                if csv_path:
                    # Extract CSV filename without extension
                    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
                    csv_names.append(csv_name)
        
        # Combine CSV names with underscores
        if csv_names:
            combined_names = "_".join(csv_names)
            # Limit filename length to avoid filesystem issues
            if len(combined_names) > 100:
                combined_names = combined_names[:100]
            prefix = combined_names + "_"
        else:
            prefix = ""
        
        filename = f"{prefix}video_calibration_{timestamp}.json"
        file_path = calibrations_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.calibration_data, f, indent=2)
            print(f"[INFO] Calibration automatically saved to: {file_path}")
            
            # Update result label to show save status
            current_text = self.result_label.cget("text")
            self.result_label.config(
                text=current_text + f"\n\n✅ Saved to: {filename}"
            )
        except Exception as e:
            print(f"[ERROR] Failed to auto-save calibration: {e}")
    
def main():
    """Main entry point."""
    root = tk.Tk()
    app = VideoCalibrationApp(root)
    print("[INFO] Video Calibration Tool started.")
    root.mainloop()


if __name__ == "__main__":
    main()

