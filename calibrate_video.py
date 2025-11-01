#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Calibration Tool

This tool allows you to:
  • Input multiple pair_detect_output folders
  • Specify the mm height and working distance for each video
  • Calculate magic offset and magic constant
  • Save the calibration parameters to a file

The calibration uses linear regression on:
  - Zprime values calculated from highest quality pairs: Zprime = working_distance * (C-A)/(A+C)
  - Z values: the calibrated mm height input for each video
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
                            pairs_data.append({
                                "r_a": r_a,
                                "r_c": r_c,
                                "score": score,
                                "track_id": track_id
                            })
                    except (ValueError, KeyError):
                        continue
            
            if len(pairs_data) == 0:
                return None
            
            # Filter for highest quality pairs based on Pair_Score (S) only
            # B is NOT used in filtering - it's calculated later for metrics only
            # High S (score) is desirable for better calibration quality and R²
            # We accept ALL pairs meeting the threshold - many high-quality points is good
            scores = [p["score"] for p in pairs_data]
            if len(scores) > 0:
                # Primary filter: Accept ALL pairs with score >= 0.9
                # Many high-quality points improves calibration statistics and R²
                score_threshold = 0.9
                quality_pairs = [p for p in pairs_data if p["score"] >= score_threshold]
                
                # Fallback 1: If we don't have enough pairs (< 10), use score >= 0.85
                # Still maintains high quality while ensuring sufficient data
                if len(quality_pairs) < 10:
                    score_threshold = 0.85
                    quality_pairs = [p for p in pairs_data if p["score"] >= score_threshold]
                
                # Fallback 2: If still not enough, use score >= 0.8
                if len(quality_pairs) < 10:
                    score_threshold = 0.8
                    quality_pairs = [p for p in pairs_data if p["score"] >= score_threshold]
                
                # Final fallback: If still no pairs, use all pairs (for robustness)
                if len(quality_pairs) == 0:
                    quality_pairs = pairs_data  # Fallback to all pairs
            
            # Calculate Zprime and B for each quality pair
            # Zprime = working_distance * (C-A)/(A+C)
            # B = (2*A*C)/(A+C)
            zprimes = []
            b_values = []
            for p in quality_pairs:
                r_a = p["r_a"]  # A is the inner radius (smaller)
                r_c = p["r_c"]  # C is the outer radius (larger)
                if r_a + r_c > 0:
                    zprime = working_dist_val * (r_c - r_a) / (r_a + r_c)
                    zprimes.append(zprime)
                    # Calculate B = (2*A*C)/(A+C)
                    b_val = (2 * r_a * r_c) / (r_a + r_c)
                    b_values.append(b_val)
            
            if len(zprimes) == 0:
                return None
            
            avg_zprime = np.mean(zprimes)
            avg_b = np.mean(b_values)
            
            # Calculate metrics for chosen/quality pairs
            chosen_metrics = {
                "count": len(quality_pairs),
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
                    "mean": float(np.mean([p["score"] for p in quality_pairs])),
                    "std": float(np.std([p["score"] for p in quality_pairs])),
                    "min": float(np.min([p["score"] for p in quality_pairs])),
                    "max": float(np.max([p["score"] for p in quality_pairs]))
                }
            }
            
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
            
            return (self.csv_path, mm_val, working_dist_val, avg_zprime, avg_b, input_metrics, chosen_metrics)
        
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
        """Create the GUI layout."""
        self.root.title("Video Calibration Tool")
        self.root.geometry("850x700")
        self.root.minsize(700, 500)
        
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except:
            pass
        
        # Create scrollable container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True)
        
        # Canvas and scrollbar for main content
        main_canvas = tk.Canvas(main_container, highlightthickness=0)
        main_scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=main_canvas.yview)
        main_frame = ttk.Frame(main_canvas, padding="15")
        
        main_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_window = main_canvas.create_window((0, 0), window=main_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=main_scrollbar.set)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        main_scrollbar.pack(side="right", fill="y")
        
        # Update main canvas window width when canvas resizes
        def update_main_window_width(event):
            canvas_width = event.width
            main_canvas.itemconfig(main_window, width=canvas_width)
        main_canvas.bind('<Configure>', update_main_window_width)
        
        # Bind mousewheel to main canvas (for scrolling main content)
        def on_mousewheel_main(event):
            main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        main_canvas.bind("<MouseWheel>", on_mousewheel_main)
        
        # Store reference for later updates
        self.main_canvas = main_canvas
        self.main_frame = main_frame
        
        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="1. Enter working distance (mm) - applies to all CSVs\n"
                 "2. Select CSV files from pair detection\n"
                 "3. Enter mm height for each CSV\n"
                 "4. Click Calculate to compute magic offset and magic constant\n"
                 "5. Save the calibration data",
            justify="left"
        )
        instructions.pack(pady=(0, 15))
        
        # Global working distance frame
        working_dist_frame = ttk.LabelFrame(main_frame, text="Working Distance (Global)", padding="10")
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
        
        # Add CSV button
        add_btn = ttk.Button(main_frame, text="➕ Add CSV", command=self.add_video_entry)
        add_btn.pack(pady=5)
        
        # Scrollable frame for CSV entries (fixed height container)
        video_entries_container = ttk.LabelFrame(main_frame, text="Calibration CSVs", padding="10")
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
        calc_btn = ttk.Button(main_frame, text="Calculate", command=self.calculate)
        calc_btn.pack(pady=10)
        
        # Result label (with max width to prevent excessive expansion)
        result_container = ttk.Frame(main_frame)
        result_container.pack(fill="x", pady=10)
        
        self.result_label = ttk.Label(
            result_container,
            text="Calibration: Not calculated",
            justify="left",
            wraplength=750
        )
        self.result_label.pack(fill="x")
        
        # Metrics display frame (scrollable, fixed height)
        metrics_frame = ttk.LabelFrame(main_frame, text="Data Metrics", padding="10")
        metrics_frame.pack(fill="both", expand=False, pady=5)
        metrics_frame.grid_rowconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(0, weight=1)
        
        # Create scrollable text widget for metrics (fixed height)
        self.metrics_text = tk.Text(metrics_frame, wrap="none", font=("Courier", 8), state="disabled", height=12, width=90)
        self.metrics_text_scrollbar = ttk.Scrollbar(metrics_frame, orient="vertical", command=self.metrics_text.yview)
        self.metrics_text.configure(yscrollcommand=self.metrics_text_scrollbar.set)
        
        self.metrics_text.grid(row=0, column=0, sticky="nsew")
        self.metrics_text_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Save button
        save_btn = ttk.Button(main_frame, text="💾 Save Calibration", command=self.save_calibration)
        save_btn.pack(pady=5)
        
        # Magic offset and constant storage
        self.magic_offset: Optional[float] = None
        self.magic_constant: Optional[float] = None
        self.calibration_data: Optional[Dict] = None
    
    def _extract_working_distance(self, cal_data: dict) -> Optional[float]:
        """
        Extract working distance from calibration data dictionary.
        Checks multiple possible locations.
        Returns the working distance value or None if not found.
        """
        # 1. Top level
        if "working_distance_mm" in cal_data:
            try:
                return float(cal_data["working_distance_mm"])
            except (ValueError, TypeError):
                pass
        
        # 2. From data_points array (first entry)
        if "data_points" in cal_data and isinstance(cal_data["data_points"], list):
            if len(cal_data["data_points"]) > 0:
                first_point = cal_data["data_points"][0]
                if isinstance(first_point, dict) and "working_distance_mm" in first_point:
                    try:
                        return float(first_point["working_distance_mm"])
                    except (ValueError, TypeError):
                        pass
        
        # 3. From camera_parameters if it exists
        if "camera_parameters" in cal_data:
            cam_params = cal_data["camera_parameters"]
            if isinstance(cam_params, dict) and "working_distance_mm" in cam_params:
                try:
                    return float(cam_params["working_distance_mm"])
                except (ValueError, TypeError):
                    pass
        
        return None
    
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
            
            working_dist = self._extract_working_distance(cal_data)
            
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
            
            working_dist = self._extract_working_distance(cal_data)
            
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
            
            working_dist = self._extract_working_distance(cal_data)
            
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
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        # Update main canvas scroll region
        if hasattr(self, 'main_canvas'):
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
    
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
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            # Update main canvas scroll region
            if hasattr(self, 'main_canvas'):
                self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
    
    def calculate(self):
        """Calculate magic offset and magic constant using linear regression on Zprime values."""
        # Collect valid data points with metrics
        data_points = []
        all_input_metrics = []
        all_chosen_metrics = []
        
        for entry in self.video_entries:
            result = entry.get_data()
            if result:
                csv_path, mm_height, working_dist, avg_zprime, avg_b, input_metrics, chosen_metrics = result
                data_points.append((csv_path, mm_height, working_dist, avg_zprime, avg_b))
                all_input_metrics.append((os.path.basename(csv_path), input_metrics))
                all_chosen_metrics.append((os.path.basename(csv_path), chosen_metrics))
                print(f"[INFO] CSV: {os.path.basename(csv_path)}, "
                      f"Height: {mm_height:.2f} mm, "
                      f"Working Dist: {working_dist:.2f} mm, "
                      f"Avg Zprime: {avg_zprime:.4f}, "
                      f"Avg B: {avg_b:.4f} px")
        
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
        )
        self.result_label.config(text=result_text)
        
        # Display detailed metrics
        self.display_metrics(all_input_metrics, all_chosen_metrics, total_input_count, total_chosen_count)
        
        print(f"[INFO] Calibration calculated:")
        print(f"  Formula: Z = Zprime * magic_constant + magic_offset")
        print(f"  Magic Constant: {self.magic_constant:.6f}")
        print(f"  Magic Offset: {self.magic_offset:.6f} mm")
        print(f"  R²: {r_squared:.4f}")
        print(f"  Input Dataset: {total_input_count} pairs")
        print(f"  Chosen Dataset: {total_chosen_count} pairs ({100*total_chosen_count/max(1,total_input_count):.1f}%)")
        
        # Automatically save to calibrations folder
        self._auto_save_calibration()
    
    def display_metrics(self, input_metrics_list, chosen_metrics_list, total_input, total_chosen):
        """Display detailed metrics for input and chosen datasets."""
        self.metrics_text.config(state="normal")
        self.metrics_text.delete(1.0, tk.END)
        
        # Calculate percentages for each video
        input_video_dict = {name: m for name, m in input_metrics_list}
        
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
        
        self.metrics_text.config(state="disabled")
        
        # Update main canvas scroll region after adding metrics
        if hasattr(self, 'main_canvas'):
            self.root.update_idletasks()
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
    
    def _auto_save_calibration(self):
        """Automatically save calibration data to the calibrations folder."""
        if self.calibration_data is None:
            return
        
        # Create calibrations folder if it doesn't exist
        calibrations_dir = Path("calibrations")
        calibrations_dir.mkdir(exist_ok=True)
        
        # Generate timestamped filename with prefix from first data point
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = ""
        if self.calibration_data.get("data_points"):
            first_csv = self.calibration_data["data_points"][0].get("csv_path", "")
            if first_csv:
                # Extract CSV filename without extension
                csv_name = os.path.splitext(os.path.basename(first_csv))[0]
                prefix = csv_name + "_"
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
    
    def save_calibration(self):
        """Save calibration data to file."""
        if self.calibration_data is None:
            messagebox.showwarning("Warning", "Please calculate the calibration first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Calibration Data",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.calibration_data, f, indent=2)
            messagebox.showinfo("Success", f"Calibration saved to:\n{file_path}")
            print(f"[INFO] Calibration saved to: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save calibration: {e}")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = VideoCalibrationApp(root)
    print("[INFO] Video Calibration Tool started.")
    root.mainloop()


if __name__ == "__main__":
    main()

