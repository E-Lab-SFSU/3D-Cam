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
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from lib.xyzcalc import (
    extract_working_distance,
    extract_pixels_per_mm,
    calculate_b_px,
    calculate_b_mm,
    calculate_xy_mm,
)
from lib.calibration import VideoCalibrator
from lib.visualizing.calibration_viz import CalibrationVisualizer
from lib.gui.calibration_gui import VideoEntry, display_metrics
from lib.util import find_latest_calibration_file


# VideoEntry is now imported from lib.gui.calibration_gui


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
        
        # Initialize calibrator and visualizer first (before they're used)
        self.calibrator = VideoCalibrator()
        self.visualizer = CalibrationVisualizer(self.root)
        
        # Expose visualizer data for VideoEntry compatibility
        self.viz_data = self.visualizer.viz_data
        self.viz_calibration_pairs = self.visualizer.viz_calibration_pairs
        
        width, height = get_standard_size("large")
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(width, height)
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
        from lib.gui import STANDARD_PADDING
        left_frame = ttk.Frame(left_canvas, padding=STANDARD_PADDING["large"])
        
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
        
        # Add button to open 3D plot window in middle column
        viz_button_frame = ttk.LabelFrame(middle_pane, text="3D Visualization", padding="10")
        viz_button_frame.pack(fill="x", pady=5)
        ttk.Button(viz_button_frame, text="Open 3D Plot Window", command=self.visualizer.open_3d_plot_window).pack(pady=5, fill="x")
        
        # Store reference for later updates
        self.main_frame = left_frame
        
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
    
    def load_csv_for_visualization(self, csv_path: str, csv_name: str):
        """Load CSV data for 3D visualization."""
        self.visualizer.load_csv_for_visualization(csv_path, csv_name)
    
    def set_calibration_pairs(self, csv_name: str, calibration_frames):
        """Set calibration pairs for a CSV file."""
        self.visualizer.set_calibration_pairs(csv_name, calibration_frames)
    
    def update_3d_visualization(self):
        """Update 3D visualization with current filtered pairs."""
        self.visualizer.update_3d_visualization()
    
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
        
        latest_cal_file = find_latest_calibration_file()
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
        latest_cal_file = find_latest_calibration_file()
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
        
        entry = VideoEntry(entry_frame, len(self.video_entries), self)
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
        
        # Use VideoCalibrator to perform the calculation
        try:
            result = self.calibrator.calculate_calibration(
                data_points,
                all_input_metrics,
                all_chosen_metrics,
                all_z_filter_stats if all_z_filter_stats else None
            )
            
            # Store results in app for compatibility
            self.magic_constant = result["magic_constant"]
            self.magic_offset = result["magic_offset"]
            self.calibration_data = result["calibration_data"]
            
            # Update result label
            result_text = (
                f"Calibration Complete!\n\n"
                f"Magic Constant: {self.magic_constant:.6f}\n"
                f"Magic Offset: {self.magic_offset:.6f} mm\n"
                f"R² (quality): {result['r_squared']:.4f}\n"
                f"Avg B: {result['avg_b']:.4f} px\n\n"
                f"Formula: Z = Zprime * {self.magic_constant:.6f} + {self.magic_offset:.6f}\n"
                f"where:\n"
                f"  Z = calibrated mm height (input)\n"
                f"  Zprime = working_distance * (C-A)/(A+C)\n"
                f"  B = (2*A*C)/(A+C)\n\n"
                f"Input Dataset: {result['total_input_count']} pairs | "
                f"Chosen Dataset: {result['total_chosen_count']} pairs ({100*result['total_chosen_count']/max(1,result['total_input_count']):.1f}%)"
                f"{result['z_filter_summary']}"
            )
            self.result_label.config(text=result_text)
            
            # Display detailed metrics using imported function
            display_metrics(self.metrics_text, all_input_metrics, all_chosen_metrics, 
                          result['total_input_count'], result['total_chosen_count'], 
                          all_z_filter_stats if all_z_filter_stats else None)
            
            print(f"[INFO] Calibration calculated:")
            print(f"  Formula: Z = Zprime * magic_constant + magic_offset")
            print(f"  Magic Constant: {self.magic_constant:.6f}")
            print(f"  Magic Offset: {self.magic_offset:.6f} mm")
            print(f"  R²: {result['r_squared']:.4f}")
            print(f"  Input Dataset: {result['total_input_count']} pairs")
            print(f"  Chosen Dataset: {result['total_chosen_count']} pairs ({100*result['total_chosen_count']/max(1,result['total_input_count']):.1f}%)")
            
            # Automatically save to calibrations folder
            saved_path = self.calibrator.auto_save_calibration()
            if saved_path:
                # Update result label to show save status
                current_text = self.result_label.cget("text")
                self.result_label.config(
                    text=current_text + f"\n\n✅ Saved to: {saved_path.name}"
                )
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        
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
        display_metrics(self.metrics_text, input_metrics_list, chosen_metrics_list, 
                       total_input, total_chosen, z_filter_stats_list)
    
    def _convert_visualization_to_zmm(self):
        """
        Convert all visualization data to Z_mm using calibration constants.
        This is called after calibration is complete to update the 3D plot with calibrated Z values.
        """
        if self.calibrator.magic_constant is None or self.calibrator.magic_offset is None:
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
                                z_mm = zprime * self.calibrator.magic_constant + self.calibrator.magic_offset
                                updated_points.append((frame, x, y, z_mm))
                            else:
                                updated_points.append((frame, x, y, z_old))
                        else:
                            # Frame not found, keep old Z value
                            updated_points.append((frame, x, y, z_old))
                    
                    data[track_id] = updated_points
                
                # Update Z unit label to mm
                self.visualizer.viz_z_unit = "mm"
                
            except Exception as e:
                print(f"[WARN] Failed to convert {csv_name} to Z_mm: {e}")
                continue
    
    def _update_csv_files_with_mm_values(self):
        """
        Update CSV files on disk with calculated mm values (Z_mm, B_px, B_mm, X_mm, Y_mm).
        Uses the current calibration constants and tries to load optical center and pixels_per_mm
        from preset and calibration files.
        """
        if self.calibrator.magic_constant is None or self.calibrator.magic_offset is None:
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
                                z_mm = zprime * self.calibrator.magic_constant + self.calibrator.magic_offset
                                row['Z_mm'] = f"{z_mm:.4f}"
                                updated_count += 1
                            else:
                                # Overwrite existing value after new calibration (user confirmed via popup)
                                zprime = working_dist_val * (r_c - r_a) / (r_a + r_c)
                                z_mm = zprime * self.calibrator.magic_constant + self.calibrator.magic_offset
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
                            
                            json_data['calibration']['magic_constant'] = float(self.calibrator.magic_constant)
                            json_data['calibration']['magic_offset'] = float(self.calibrator.magic_offset)
                            json_data['calibration']['working_distance_mm'] = float(working_dist_val)
                            json_data['calibration']['formula'] = "Z_mm = Zprime * magic_constant + magic_offset"
                            
                            if pixels_per_mm is not None:
                                json_data['calibration']['pixels_per_mm'] = float(pixels_per_mm)
                            
                            # Add calibration timestamp
                            json_data['calibration']['calibrated_at'] = datetime.now().isoformat()
                            
                            # Update if calibration data exists
                            if self.calibrator.calibration_data:
                                if 'r_squared' in self.calibrator.calibration_data:
                                    json_data['calibration']['r_squared'] = float(self.calibrator.calibration_data['r_squared'])
                            
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
        # This is now handled by calibrator.auto_save_calibration() in calculate()
        pass
    
def main():
    """Main entry point."""
    root = tk.Tk()
    app = VideoCalibrationApp(root)
    print("[INFO] Video Calibration Tool started.")
    root.mainloop()


if __name__ == "__main__":
    main()

