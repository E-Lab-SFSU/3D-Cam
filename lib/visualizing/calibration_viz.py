"""
Calibration 3D Visualization
-----------------------------
3D visualization for video calibration pairs.
"""

import csv
import os
import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional, Set
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import cm


class CalibrationVisualizer:
    """3D visualization for calibration pairs."""
    
    def __init__(self, root_window: tk.Tk):
        self.root = root_window
        self.viz_window: Optional[tk.Toplevel] = None
        self.viz_fig: Optional[Figure] = None
        self.viz_ax = None
        self.viz_canvas = None
        
        # Visualization data: {csv_name: {track_id: [(frame, x, y, z), ...]}}
        self.viz_data: Dict[str, Dict[int, list]] = {}
        
        # Calibration pairs: {csv_name: {track_id: set of frames}}
        self.viz_calibration_pairs: Dict[str, Dict[int, Set[int]]] = {}
        
        # Coordinate unit tracking for axis labels
        self.viz_x_unit = "mm"
        self.viz_y_unit = "mm"
        self.viz_z_unit = "mm"
    
    def open_3d_plot_window(self):
        """Open 3D visualization in a separate window."""
        if self.viz_window is not None:
            try:
                self.viz_window.lift()
                self.viz_window.focus_force()
                return
            except:
                self.viz_window = None
        
        # Create new window for 3D plot
        self.viz_window = tk.Toplevel(self.root)
        self.viz_window.title("Calibration Pairs 3D Visualization")
        self.viz_window.geometry("1000x800")
        self.viz_window.transient(self.root)
        
        # Handle window close
        self.viz_window.protocol("WM_DELETE_WINDOW", self.on_viz_window_close)
        
        # Create figure for 3D plot
        self.viz_fig = Figure(figsize=(10, 8), dpi=100)
        self.viz_ax = self.viz_fig.add_subplot(111, projection='3d')
        
        # Create canvas
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, self.viz_window)
        self.viz_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.viz_canvas, self.viz_window)
        toolbar.update()
        
        # Initial empty plot
        self.viz_ax.set_xlabel('X (mm)')
        self.viz_ax.set_ylabel('Y (mm)')
        self.viz_ax.set_zlabel('Z (mm)')
        self.viz_ax.set_title('Calibration Pairs Visualization')
        self.viz_canvas.draw()
        
        # Update visualization if data already exists
        if self.viz_data:
            self.update_3d_visualization()
    
    def on_viz_window_close(self):
        """Handle closing of 3D visualization window."""
        if self.viz_window:
            self.viz_window.destroy()
        self.viz_window = None
        # Clear references but keep data
        self.viz_fig = None
        self.viz_ax = None
        self.viz_canvas = None
    
    def update_3d_visualization(self):
        """Update 3D visualization with current filtered pairs."""
        if not hasattr(self, 'viz_ax') or self.viz_ax is None:
            return
            
        self.viz_ax.clear()
        
        if not self.viz_data:
            self.viz_ax.text(0.5, 0.5, 0.5, "Load CSVs to see calibration pairs", 
                           transform=self.viz_ax.transAxes, ha="center")
            if self.viz_canvas:
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
        if self.viz_canvas:
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
                    has_z_mm_data = bool(sample_row.get('Z_mm', '').strip())
                    has_zprime_data = bool(sample_row.get('Zprime_mm', '').strip())
                    has_zdoubleprime_data = bool(sample_row.get('Zdoubleprime', '').strip())
                    
                    if has_z_mm_data:
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
                        if has_z_mm:
                            z_str = row.get('Z_mm', '').strip()
                            if z_str:
                                z = float(z_str)
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
            print(f"[ERROR] Failed to load CSV for visualization: {e}")
    
    def set_calibration_pairs(self, csv_name: str, calibration_frames: Dict[int, Set[int]]):
        """Set calibration pairs for a CSV file."""
        self.viz_calibration_pairs[csv_name] = calibration_frames
    
    def convert_visualization_to_zmm(self, z_calibration_scale_factor: float, z_calibration_offset_mm: float, working_distance: float):
        """
        Convert all visualization data to Z_mm using calibration constants.
        
        Args:
            z_calibration_scale_factor: Z calibration scale factor
            z_calibration_offset_mm: Z calibration offset in mm
            working_distance: Working distance in mm
        """
        # Find CSV paths for all loaded visualization data
        csv_path_map = {}  # {csv_name: csv_path}
        for csv_name in self.viz_data.keys():
            # We'll need the actual CSV paths - this will be provided by the app
            pass
        
        # This method will be called by the app with the actual CSV paths
        # For now, we'll update the unit label
        self.viz_z_unit = "mm"
    
    def reload_csv(self, csv_path: str, csv_name: str):
        """Reload CSV data for visualization after Z_mm has been updated."""
        self.load_csv_for_visualization(csv_path, csv_name)

