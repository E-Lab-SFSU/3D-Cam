#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Pair Visualization Tool
--------------------------
Interactive 3D visualization of X, Y, Z pair trajectories over time.

Features:
  • Load CSV files from pair_detect.py exports
  • Interactive 3D view with mouse drag to pan/rotate/zoom
  • Time slider to scrub through frames
  • Show trajectories with color-coded tracks
  • Toggle trails/history display
  • Select specific track IDs to visualize
"""

import tkinter as tk
from tkinter import ttk, messagebox
import csv
from pathlib import Path

from lib.visualizing import Base3DVisualizer


class Pair3DVisualizer(Base3DVisualizer):
    """3D pair trajectory visualizer extending Base3DVisualizer."""
    
    def __init__(self, root):
        # Initialize coordinate column selection
        self.use_smoothed_mm = tk.StringVar(value="original")  # "original" or "smoothed"
        
        # Initialize base class - wider window for 3-column layout
        super().__init__(root, "3D Pair Trajectory Visualizer", "1400x800")
        
        # Setup coordinate column selection dropdown
        self.setup_coordinate_selection()
    
    def setup_coordinate_selection(self):
        """Add dropdown to select between original and smoothed mm columns."""
        coord_frame = ttk.LabelFrame(self.custom_section, text="Coordinate Selection", padding="10")
        coord_frame.pack(fill="x", pady=5)
        
        ttk.Label(coord_frame, text="Use:").pack(side="left", padx=(0, 5))
        
        coord_combo = ttk.Combobox(
            coord_frame,
            textvariable=self.use_smoothed_mm,
            values=("original", "smoothed"),
            state="readonly",
            width=15
        )
        coord_combo.pack(side="left", padx=(0, 5))
        coord_combo.bind("<<ComboboxSelected>>", self.on_coordinate_type_changed)
        
        ttk.Label(coord_frame, text="mm columns").pack(side="left")
    
    def on_coordinate_type_changed(self, event=None):
        """Handle coordinate type selection change - reload CSV with new column selection."""
        if self.csv_path:
            # Reload the CSV file with the new column selection
            self.load_csv_file(self.csv_path)
    
    def load_csv_file(self, file_path: str):
        """Override to support smoothed mm columns selection."""
        self.csv_path = file_path
        self.data = {}
        self.frame_data = {}
        self.max_frame = 0
        self.track_ids = []
        
        # Determine which column suffix to use
        use_smoothed = self.use_smoothed_mm.get() == "smoothed"
        x_col = 'X_mm_smoothed' if use_smoothed else 'X_mm'
        y_col = 'Y_mm_smoothed' if use_smoothed else 'Y_mm'
        z_col = 'Z_mm_smoothed' if use_smoothed else 'Z_mm'
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Check for coordinate columns
                has_x_mm = x_col in reader.fieldnames
                has_y_mm = y_col in reader.fieldnames
                has_center_x = 'Center_X' in reader.fieldnames
                has_center_y = 'Center_Y' in reader.fieldnames
                has_z_mm = z_col in reader.fieldnames
                has_zprime_mm = 'Zprime_mm' in reader.fieldnames
                has_zdoubleprime = 'Zdoubleprime' in reader.fieldnames
                has_a_px = 'A_px' in reader.fieldnames
                has_c_px = 'C_px' in reader.fieldnames
                
                # Also check original columns as fallback
                has_original_x_mm = 'X_mm' in reader.fieldnames
                has_original_y_mm = 'Y_mm' in reader.fieldnames
                has_original_z_mm = 'Z_mm' in reader.fieldnames
                
                # If smoothed columns are requested but not available, warn and fall back to original
                if use_smoothed:
                    if not has_x_mm or not has_y_mm:
                        messagebox.showwarning(
                            "Smoothed Columns Not Available",
                            f"CSV file does not contain smoothed mm columns ({x_col}, {y_col}).\n"
                            "Falling back to original mm columns."
                        )
                        use_smoothed = False
                        x_col = 'X_mm'
                        y_col = 'Y_mm'
                        z_col = 'Z_mm'
                        has_x_mm = has_original_x_mm
                        has_y_mm = has_original_y_mm
                        has_z_mm = has_original_z_mm
                        self.use_smoothed_mm.set("original")
                
                # Need at least some form of X/Y data
                if not (has_x_mm or has_center_x) or not (has_y_mm or has_center_y):
                    messagebox.showerror("Error", "CSV file must contain X and Y coordinate columns")
                    return
                
                # Determine which columns to use - prefer calibrated columns
                can_calc_zprime = has_a_px and has_c_px
                
                # Read all rows into a list so we can peek at first row for unit detection
                all_rows = list(reader)
                
                # Set coordinate units based on available columns with actual data
                if all_rows:
                    sample_row = all_rows[0]
                    has_x_mm_data = bool(sample_row.get(x_col, '').strip())
                    has_y_mm_data = bool(sample_row.get(y_col, '').strip())
                    # XY labels: "X (mm), Y (mm)" if X_mm/Y_mm has data, otherwise "X (px), Y (px)"
                    if has_x_mm_data and has_y_mm_data:
                        self.x_unit = "mm"
                        self.y_unit = "mm"
                    else:
                        self.x_unit = "px"
                        self.y_unit = "px"
                    
                    # Set Z unit based on available columns with actual data
                    has_z_mm_data = bool(sample_row.get(z_col, '').strip())
                    has_zprime_data = bool(sample_row.get('Zprime_mm', '').strip())
                    has_zdoubleprime_data = bool(sample_row.get('Zdoubleprime', '').strip())
                    
                    if has_z_mm_data:
                        self.z_unit = "mm"
                    elif has_zprime_data:
                        self.z_unit = ""  # "Z" label (no unit indicator for Zprime)
                    elif has_zdoubleprime_data or can_calc_zprime:
                        self.z_unit = ""  # "Z" label (no unit indicator for Zdoubleprime)
                    else:
                        self.z_unit = "mm"  # Default fallback
                else:
                    # No data, use defaults
                    self.x_unit = "px"
                    self.y_unit = "px"
                    self.z_unit = "mm"
                
                for row in all_rows:
                    try:
                        frame = int(row['Frame_Number'])
                        track_id = int(row['Track_ID'])
                        
                        # Get X and Y coordinates - fallback order: selected X_mm/Y_mm -> Center_X/Center_Y
                        x_str = ''
                        y_str = ''
                        if has_x_mm and has_y_mm:
                            x_str = row.get(x_col, '').strip()
                            y_str = row.get(y_col, '').strip()
                        if (not x_str or not y_str) and has_center_x and has_center_y:
                            x_str = row.get('Center_X', '').strip()
                            y_str = row.get('Center_Y', '').strip()
                        
                        if not x_str or not y_str:
                            continue
                        
                        x = float(x_str)
                        y = float(y_str)
                        
                        # Get Z coordinate - prefer selected column, fall back to calculated values
                        z = 0.0
                        z_str = ''
                        
                        # Try selected Z_mm column first
                        if has_z_mm:
                            z_str = row.get(z_col, '').strip()
                            if z_str:
                                z = float(z_str)
                        
                        # Try Zprime_mm next
                        if not z_str and has_zprime_mm:
                            z_str = row.get('Zprime_mm', '').strip()
                            if z_str:
                                z = float(z_str)
                        
                        # Try Zdoubleprime next (always calculated, working_distance = 1)
                        if not z_str and has_zdoubleprime:
                            z_str = row.get('Zdoubleprime', '').strip()
                            if z_str:
                                z = float(z_str)
                        
                        # Calculate Zprime from A and C as last resort
                        if not z_str and can_calc_zprime:
                            try:
                                a_px = float(row.get('A_px', '0').strip() or '0')
                                c_px = float(row.get('C_px', '0').strip() or '0')
                                if a_px > 0 and c_px > 0:
                                    # Calculate Zdoubleprime (working_distance = 1)
                                    z = (c_px - a_px) / (a_px + c_px)
                            except (ValueError, KeyError):
                                pass
                        
                        if track_id not in self.data:
                            self.data[track_id] = []
                        self.data[track_id].append((frame, x, y, z))
                        
                        if frame not in self.frame_data:
                            self.frame_data[frame] = []
                        self.frame_data[frame].append((track_id, x, y, z))
                        
                        self.max_frame = max(self.max_frame, frame)
                    
                    except (ValueError, KeyError):
                        continue
                
                # Show info about which columns are available
                coord_info = []
                if has_x_mm and has_y_mm:
                    coord_info.append(f"{x_col}/{y_col}")
                if has_center_x and has_center_y:
                    coord_info.append("Center_X/Center_Y")
                
                z_info = []
                if has_z_mm:
                    z_info.append(z_col)
                if has_zprime_mm:
                    z_info.append("Zprime_mm")
                if has_zdoubleprime:
                    z_info.append("Zdoubleprime")
                if can_calc_zprime:
                    z_info.append("calculated from A/C")
                
                print(f"[INFO] Available coordinates: {', '.join(coord_info) if coord_info else 'None'}")
                print(f"[INFO] Available Z values: {', '.join(z_info) if z_info else 'None'}")
            
            # Sort data by frame for each track
            for track_id in self.data:
                self.data[track_id].sort(key=lambda p: p[0])
            
            self.track_ids = sorted(self.data.keys())
            
            # Update UI
            if hasattr(self, 'file_label'):
                self.file_label.config(text=f"Loaded: {Path(file_path).name}")
            if self.playback_controller:
                self.playback_controller.set_max_frame(self.max_frame)
                self.playback_controller.set_frame(0)
            self.current_frame = 0
            
            # Update track checkboxes
            if hasattr(self, 'update_track_checkboxes'):
                self.update_track_checkboxes()
            
            # Load and display CSV metadata JSON with specs
            if hasattr(self, 'load_csv_metadata'):
                self.load_csv_metadata(file_path)
            
            # Update info
            total_points = sum(len(self.data[tid]) for tid in self.data)
            if hasattr(self, 'info_label'):
                self.info_label.config(
                    text=f"Tracks: {len(self.track_ids)}\n"
                         f"Frames: {self.max_frame + 1}\n"
                         f"Total Points: {total_points}"
                )
            
            # Reset bounds
            self.bounds_set = False
            self.persistent_bounds = None
            self._view_set = False
            
            # Update plot
            self.update_plot()
            
            # Call custom on_load callback
            if hasattr(self, 'on_data_loaded'):
                self.on_data_loaded()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file:\n{e}")
            import traceback
            traceback.print_exc()
    
    def get_plot_title(self):
        """Override to customize title."""
        coord_type = "Smoothed" if self.use_smoothed_mm.get() == "smoothed" else "Original"
        return f'3D Pair Trajectories ({coord_type}) - Frame {self.current_frame}/{self.max_frame}'


def main():
    root = tk.Tk()
    app = Pair3DVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
