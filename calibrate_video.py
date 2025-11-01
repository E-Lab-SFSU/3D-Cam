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
        self.folder_var = tk.StringVar()
        self.height_var = tk.StringVar()
        self.folder_path = ""
        self.mm_height: Optional[float] = None
        
        # Create widgets
        ttk.Label(frame, text=f"Video {row + 1}:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        
        folder_label = ttk.Label(frame, text="No folder selected", foreground="gray")
        folder_label.grid(row=0, column=1, sticky="w", padx=5)
        self.folder_label = folder_label
        
        folder_btn = ttk.Button(frame, text="📂 Browse", 
                               command=lambda: self.select_folder())
        folder_btn.grid(row=0, column=2, padx=5)
        
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
    
    def select_folder(self):
        """Open folder dialog to select pair_detect_output folder."""
        folder = filedialog.askdirectory(
            title=f"Select pair_detect_output Folder for Video {self.row + 1}",
            initialdir="pair_detect_output" if os.path.exists("pair_detect_output") else "."
        )
        
        if folder:
            pairs_csv = os.path.join(folder, "pairs.csv")
            if not os.path.exists(pairs_csv):
                messagebox.showerror(
                    "Error",
                    f"pairs.csv not found in:\n{folder}\n\n"
                    "Please select a valid pair_detect_output folder."
                )
                return
            
            self.folder_path = folder
            folder_name = os.path.basename(folder)
            self.folder_label.config(text=folder_name, foreground="black")
            print(f"[INFO] Video {self.row + 1}: Selected folder {folder}")
    
    def get_data(self) -> Optional[Tuple[str, float, float, float, float]]:
        """
        Get folder path, mm height, working distance, average Zprime, and average B from highest quality pairs.
        Returns None if invalid.
        Returns: (folder_path, mm_height, working_distance_mm, avg_zprime, avg_b)
        """
        if not self.folder_path or not os.path.exists(self.folder_path):
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
        
        # Load pairs from pairs.csv and calculate Zprime for highest quality pairs
        pairs_csv = os.path.join(self.folder_path, "pairs.csv")
        if not os.path.exists(pairs_csv):
            return None
        
        try:
            pairs_data = []
            with open(pairs_csv, 'r', encoding='utf-8') as f:
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
            
            # Filter for highest quality pairs (top 20% by score, or pairs with score > 0.9)
            scores = [p["score"] for p in pairs_data]
            if len(scores) > 0:
                # Use top 20% by score, but at least use score > 0.8 threshold
                score_threshold = max(0.8, np.percentile(scores, 80))
                quality_pairs = [p for p in pairs_data if p["score"] >= score_threshold]
                
                # If we don't have enough, use top 50%
                if len(quality_pairs) < 10:
                    score_threshold = np.percentile(scores, 50)
                    quality_pairs = [p for p in pairs_data if p["score"] >= score_threshold]
                
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
            return (self.folder_path, mm_val, working_dist_val, avg_zprime, avg_b)
        
        except Exception as e:
            print(f"[ERROR] Failed to read {pairs_csv}: {e}")
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
        self.root.geometry("700x600")
        
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except:
            pass
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill="both", expand=True)
        
        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="1. Enter working distance (mm) - applies to all videos\n"
                 "2. Select pair_detect_output folders\n"
                 "3. Enter mm height for each video\n"
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
        
        # Add video button
        add_btn = ttk.Button(main_frame, text="➕ Add Video", command=self.add_video_entry)
        add_btn.pack(pady=5)
        
        # Scrollable frame for video entries
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill="both", expand=True, pady=10)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white")
        self.scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel
        def on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Calculate button
        calc_btn = ttk.Button(main_frame, text="Calculate", command=self.calculate)
        calc_btn.pack(pady=10)
        
        # Result label
        self.result_label = ttk.Label(
            main_frame,
            text="Calibration: Not calculated",
            justify="left"
        )
        self.result_label.pack(pady=10)
        
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
        """Add a new video entry row."""
        entry_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text=f"Video {len(self.video_entries) + 1}",
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
    
    def remove_video_entry(self, entry: VideoEntry):
        """Remove a video entry."""
        if len(self.video_entries) <= 2:
            messagebox.showwarning(
                "Warning",
                "You need at least 2 videos for calibration.\n"
                "Cannot remove this entry."
            )
            return
        
        if entry in self.video_entries:
            entry.frame.destroy()
            self.video_entries.remove(entry)
            # Renumber remaining entries
            for i, e in enumerate(self.video_entries):
                e.row = i
                e.frame.config(text=f"Video {i + 1}")
            # Update canvas scroll region
            self.root.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def calculate(self):
        """Calculate magic offset and magic constant using linear regression on Zprime values."""
        # Collect valid data points
        data_points = []
        for entry in self.video_entries:
            result = entry.get_data()
            if result:
                folder, mm_height, working_dist, avg_zprime, avg_b = result
                data_points.append((folder, mm_height, working_dist, avg_zprime, avg_b))
                print(f"[INFO] Video: {os.path.basename(folder)}, "
                      f"Height: {mm_height:.2f} mm, "
                      f"Working Dist: {working_dist:.2f} mm, "
                      f"Avg Zprime: {avg_zprime:.4f}, "
                      f"Avg B: {avg_b:.4f} px")
        
        if len(data_points) < 2:
            messagebox.showerror(
                "Error",
                f"Need at least 2 valid videos with:\n"
                "- Selected pair_detect_output folder\n"
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
        
        self.calibration_data = {
            "magic_constant": float(self.magic_constant),
            "magic_offset": float(self.magic_offset),
            "r_squared": float(r_squared),
            "data_points": [
                {
                    "folder": folder,
                    "z_mm": float(h),  # Z = calibrated input value
                    "working_distance_mm": float(wd),
                    "avg_zprime": float(z),
                    "avg_b": float(b)
                }
                for folder, h, wd, z, b in data_points
            ],
            "formula": "Z = Zprime * magic_constant + magic_offset",
            "description": "Z is the calibrated mm height input, Zprime is calculated from pair data",
            "zprime_formula": "Zprime = working_distance * (C-A)/(A+C)",
            "b_formula": "B = (2*A*C)/(A+C)"
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
            f"  B = (2*A*C)/(A+C)"
        )
        self.result_label.config(text=result_text)
        
        print(f"[INFO] Calibration calculated:")
        print(f"  Formula: Z = Zprime * magic_constant + magic_offset")
        print(f"  Magic Constant: {self.magic_constant:.6f}")
        print(f"  Magic Offset: {self.magic_offset:.6f} mm")
        print(f"  R²: {r_squared:.4f}")
        
        # Automatically save to calibrations folder
        self._auto_save_calibration()
    
    def _auto_save_calibration(self):
        """Automatically save calibration data to the calibrations folder."""
        if self.calibration_data is None:
            return
        
        # Create calibrations folder if it doesn't exist
        calibrations_dir = Path("calibrations")
        calibrations_dir.mkdir(exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_calibration_{timestamp}.json"
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

