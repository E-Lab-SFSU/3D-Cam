#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track Smoother and Cleaner
---------------------------
Tool to smooth and clean particle tracking data by removing spikes and applying smoothing filters.

Features:
  • Load CSV files from pair_detect_output
  • Interactive 3D view with before/after comparison
  • Spike detection and removal with interpolation
  • Adjustable smoothing parameters
  • Visual smoothness metrics
  • Export cleaned data
"""

import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FFMpegWriter
from typing import Dict, List, Tuple
from datetime import datetime

from lib.visualizing import Base3DVisualizer


class TrackSmoother(Base3DVisualizer):
    """Track smoother that extends Base3DVisualizer with smoothing functionality."""
    
    def __init__(self, root):
        # Initialize smoothing-specific data BEFORE calling super().__init__()
        # because update_plot() might be called during base class initialization
        self.original_data = {}  # Keep original unsmoothed data
        self.smoothed_data = {}  # Smoothed version
        self.spike_indices = {}  # {track_id: [frame_index, ...]} - frames with spikes
        
        # Smoothing parameters
        self.smoothing_window_size = 5
        self.spike_threshold = 2.0
        self.spike_velocity_threshold = 50.0
        self.enable_smoothing = True
        self.enable_spike_removal = True
        
        # Smoothness metrics
        self.smoothness_metrics = {}  # {track_id: {'original': metric, 'smoothed': metric}}
        
        # Initialize display variables BEFORE calling super().__init__()
        # because update_plot() might be called during base class initialization
        self.show_orig_var = tk.BooleanVar(value=True)
        self.show_smooth_var = tk.BooleanVar(value=True)
        self.show_spikes_var = tk.BooleanVar(value=True)
        
        # Initialize smoothing window references
        self.smoothing_window = None
        self.metrics_text = None  # Will be created in smoothing window
        
        # Initialize base class (this will call setup_ui() which may call update_plot())
        super().__init__(root, "Track Smoother and Cleaner", "1400x900")
        
        # Setup custom UI elements for smoothing (after base class setup)
        # Use after() to ensure UI is fully initialized
        self.root.after(100, self.setup_smoothing_ui)
    
    def setup_smoothing_ui(self):
        """Add button to open smoothing window in the custom section."""
        # Ensure custom_section exists (should always exist after base class init)
        if not hasattr(self, 'custom_section') or self.custom_section is None:
            print("[WARN] custom_section not available, skipping smoothing UI setup")
            return
        
        try:
            # Add a button to open the smoothing window
            smooth_button_frame = ttk.Frame(self.custom_section)
            smooth_button_frame.pack(fill="x", pady=5)
            ttk.Button(smooth_button_frame, text="Open Smoothing Controls", 
                      command=self.open_smoothing_window).pack(pady=5, fill="x")
        except Exception as e:
            print(f"[WARN] Failed to setup smoothing UI: {e}")
            import traceback
            traceback.print_exc()
    
    def open_smoothing_window(self):
        """Open a separate window for smoothing controls."""
        if self.smoothing_window is not None:
            # Window already exists, just bring it to front
            try:
                self.smoothing_window.lift()
                self.smoothing_window.focus_force()
            except:
                self.smoothing_window = None  # Window was closed
        
        if self.smoothing_window is None:
            # Create new window
            self.smoothing_window = tk.Toplevel(self.root)
            self.smoothing_window.title("Track Smoothing Controls")
            self.smoothing_window.geometry("400x750")
            self.smoothing_window.transient(self.root)
            
            # Handle window close
            self.smoothing_window.protocol("WM_DELETE_WINDOW", self.on_smoothing_window_close)
            
            # Main container
            main_frame = ttk.Frame(self.smoothing_window, padding="10")
            main_frame.pack(fill="both", expand=True)
            
            # Smoothing parameters
            smooth_frame = ttk.LabelFrame(main_frame, text="Smoothing Parameters", padding="10")
            smooth_frame.pack(fill="x", pady=5)
            
            # Enable smoothing checkbox
            self.enable_smooth_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(smooth_frame, text="Enable Smoothing", variable=self.enable_smooth_var,
                           command=self.on_smoothing_changed).pack(anchor="w")
            
            # Smoothing window
            window_frame = ttk.Frame(smooth_frame)
            window_frame.pack(fill="x", pady=5)
            ttk.Label(window_frame, text="Window:").pack(side="left")
            self.window_var = tk.IntVar(value=5)
            window_scale = ttk.Scale(window_frame, from_=1, to=21, orient="horizontal",
                                    variable=self.window_var, length=200, command=self.on_params_changed)
            window_scale.pack(side="left", fill="x", expand=True, padx=5)
            self.window_label = ttk.Label(window_frame, text="5")
            self.window_label.pack(side="left")
            window_scale.configure(command=lambda v: self.window_label.config(text=str(int(float(v)))))
            
            # Separator
            ttk.Separator(smooth_frame, orient="horizontal").pack(fill="x", pady=5)
            
            # Spike removal
            self.enable_spike_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(smooth_frame, text="Remove Spikes", variable=self.enable_spike_var,
                           command=self.on_smoothing_changed).pack(anchor="w", pady=(5, 0))
            
            # Spike threshold
            spike_thresh_frame = ttk.Frame(smooth_frame)
            spike_thresh_frame.pack(fill="x", pady=5)
            ttk.Label(spike_thresh_frame, text="Threshold:").pack(side="left")
            self.spike_thresh_var = tk.DoubleVar(value=2.0)
            spike_scale = ttk.Scale(spike_thresh_frame, from_=0.5, to=5.0, orient="horizontal",
                                   variable=self.spike_thresh_var, length=200, command=self.on_params_changed)
            spike_scale.pack(side="left", fill="x", expand=True, padx=5)
            self.spike_thresh_label = ttk.Label(spike_thresh_frame, text="2.0σ")
            self.spike_thresh_label.pack(side="left")
            spike_scale.configure(command=lambda v: self.spike_thresh_label.config(text=f"{float(v):.1f}σ"))
            
            # Velocity spike threshold
            vel_thresh_frame = ttk.Frame(smooth_frame)
            vel_thresh_frame.pack(fill="x", pady=5)
            ttk.Label(vel_thresh_frame, text="Vel. Limit:").pack(side="left")
            self.vel_thresh_var = tk.DoubleVar(value=50.0)
            vel_scale = ttk.Scale(vel_thresh_frame, from_=10.0, to=200.0, orient="horizontal",
                                 variable=self.vel_thresh_var, length=200, command=self.on_params_changed)
            vel_scale.pack(side="left", fill="x", expand=True, padx=5)
            self.vel_thresh_label = ttk.Label(vel_thresh_frame, text="50.0")
            self.vel_thresh_label.pack(side="left")
            vel_scale.configure(command=lambda v: self.vel_thresh_label.config(text=f"{float(v):.1f}"))
            
            # Apply button
            ttk.Button(smooth_frame, text="Apply Smoothing", command=self.apply_smoothing).pack(pady=10, fill="x")
            
            # Display options specific to smoothing
            display_frame = ttk.LabelFrame(main_frame, text="Smoothing Display", padding="10")
            display_frame.pack(fill="x", pady=5)
            
            # Variables are already initialized in __init__, just create the UI widgets
            ttk.Checkbutton(display_frame, text="Show Original", variable=self.show_orig_var,
                           command=self.update_plot).pack(anchor="w")
            
            ttk.Checkbutton(display_frame, text="Show Smoothed", variable=self.show_smooth_var,
                           command=self.update_plot).pack(anchor="w")
            
            ttk.Checkbutton(display_frame, text="Show Detected Spikes", variable=self.show_spikes_var,
                           command=self.update_plot).pack(anchor="w")
            
            # Smoothness metrics
            metrics_frame = ttk.LabelFrame(main_frame, text="Smoothness Metrics", padding="10")
            metrics_frame.pack(fill="both", expand=True, pady=5)
            
            metrics_text_frame = ttk.Frame(metrics_frame)
            metrics_text_frame.pack(fill="both", expand=True)
            
            metrics_scrollbar = ttk.Scrollbar(metrics_text_frame)
            metrics_scrollbar.pack(side="right", fill="y")
            
            self.metrics_text = tk.Text(metrics_text_frame, height=15, wrap="word",
                                        yscrollcommand=metrics_scrollbar.set, font=("Consolas", 9))
            self.metrics_text.pack(side="left", fill="both", expand=True)
            metrics_scrollbar.config(command=self.metrics_text.yview)
    
    def on_smoothing_window_close(self):
        """Handle closing of smoothing window."""
        self.smoothing_window.destroy()
        self.smoothing_window = None
        # Don't destroy metrics_text, just unset reference - we'll recreate if window reopens
    
    def setup_export_section(self, parent_frame):
        """Setup export section for smoothed tracks."""
        ttk.Button(parent_frame, text="Export Cleaned CSV", command=self.export_cleaned).pack(pady=5, fill="x")
        ttk.Button(parent_frame, text="Save Video", command=self.save_video).pack(pady=5, fill="x")
    
    def load_csv_file(self, file_path: str):
        """Override to store original data separately."""
        # Call parent to load data into self.data
        super().load_csv_file(file_path)
        
        # Store original data and initialize smoothed data
        self.original_data = {tid: list(points) for tid, points in self.data.items()}
        self.smoothed_data = {tid: list(points) for tid, points in self.data.items()}
        
        # Update frame_data to use smoothed data
        self.frame_data = {}
        for track_id, points in self.smoothed_data.items():
            for frame, x, y, z in points:
                if frame not in self.frame_data:
                    self.frame_data[frame] = []
                self.frame_data[frame].append((track_id, x, y, z))
        
        # Calculate initial metrics (only if metrics_text exists - window might not be open)
        if hasattr(self, 'metrics_text') and self.metrics_text is not None:
            self.calculate_smoothness_metrics()
    
    def on_params_changed(self, value=None):
        """Handle parameter changes - update labels only."""
        pass
    
    def on_smoothing_changed(self):
        """Handle smoothing enable/disable."""
        self.enable_smoothing = self.enable_smooth_var.get()
        self.enable_spike_removal = self.enable_spike_var.get()
    
    def detect_spikes(self, track_id: int, points: List[Tuple[int, float, float, float]]) -> List[int]:
        """Detect spike indices in a track based on velocity and statistical outliers."""
        if len(points) < 3:
            return []
        
        spike_indices = []
        frames = [p[0] for p in points]
        xs = np.array([p[1] for p in points])
        ys = np.array([p[2] for p in points])
        zs = np.array([p[3] for p in points])
        
        # Calculate velocities (mm per frame)
        velocities = np.zeros(len(points))
        for i in range(1, len(points)):
            dx = xs[i] - xs[i-1]
            dy = ys[i] - ys[i-1]
            dz = zs[i] - zs[i-1]
            velocities[i] = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Method 1: Velocity threshold
        vel_threshold = self.vel_thresh_var.get()
        for i in range(1, len(velocities)):
            if velocities[i] > vel_threshold:
                spike_indices.append(i)
        
        # Method 2: Statistical outlier detection (Z-score)
        if len(points) >= 5:
            dx = np.diff(xs)
            dy = np.diff(ys)
            dz = np.diff(zs)
            distances = np.sqrt(dx**2 + dy**2 + dz**2)
            
            if len(distances) > 2:
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                
                if std_dist > 0:
                    z_scores = np.abs((distances - mean_dist) / std_dist)
                    threshold = self.spike_thresh_var.get()
                    
                    for i in range(len(z_scores)):
                        if z_scores[i] > threshold and (i+1) not in spike_indices:
                            spike_indices.append(i+1)
        
        return sorted(set(spike_indices))
    
    def interpolate_point(self, points: List[Tuple[int, float, float, float]], 
                         spike_idx: int) -> Tuple[int, float, float, float]:
        """Interpolate a point at spike_idx using neighboring points."""
        # Helper function to check if point is at origin
        def is_at_origin(pt):
            return pt[1] == 0.0 and pt[2] == 0.0 and pt[3] == 0.0
        
        if len(points) == 0:
            return (0, 0.0, 0.0, 0.0)
        
        if spike_idx < 0 or spike_idx >= len(points):
            return points[min(max(0, spike_idx), len(points) - 1)]
        
        frame = points[spike_idx][0]
        original_point = points[spike_idx]
        
        # Handle edge cases - find valid neighbors (not at origin unless they should be)
        if spike_idx == 0:
            if len(points) > 1:
                # Find first valid neighbor (not at origin) - look further ahead
                for i in range(1, min(len(points), 5)):  # Check up to 4 neighbors forward
                    neighbor = points[i]
                    if not is_at_origin(neighbor):
                        # Found valid neighbor - use its position (simple extrapolation)
                        return (frame, neighbor[1], neighbor[2], neighbor[3])
                # If all neighbors are at origin, keep original point (don't change to 0,0,0)
                if not is_at_origin(original_point):
                    return original_point
                # Original is also at origin, use first non-origin neighbor or just keep it
                return points[0]
            else:
                return points[0]
        
        if spike_idx == len(points) - 1:
            if len(points) > 1:
                # Find last valid neighbor (not at origin) - look further back
                for i in range(len(points) - 2, max(-1, len(points) - 6), -1):  # Check up to 4 neighbors backward
                    neighbor = points[i]
                    if not is_at_origin(neighbor):
                        # Found valid neighbor - use its position
                        return (frame, neighbor[1], neighbor[2], neighbor[3])
                # If all neighbors are at origin, keep original point
                if not is_at_origin(original_point):
                    return original_point
                # Original is also at origin, use last non-origin neighbor or just keep it
                return points[-1]
            else:
                return points[-1]
        
        # Linear interpolation between neighbors
        prev_point = points[spike_idx - 1]
        next_point = points[spike_idx + 1]
        
        # Skip interpolation if both neighbors are at origin - keep original point instead
        prev_at_origin = is_at_origin(prev_point)
        next_at_origin = is_at_origin(next_point)
        
        if prev_at_origin and next_at_origin:
            # Both neighbors at origin - keep original point to avoid teleporting
            if not is_at_origin(original_point):
                return original_point
            # Original is also at origin, try to find other neighbors
            # Look for valid neighbors further away
            for offset in range(2, min(5, spike_idx + 1, len(points) - spike_idx)):
                if spike_idx - offset >= 0:
                    far_prev = points[spike_idx - offset]
                    if not is_at_origin(far_prev):
                        return (frame, far_prev[1], far_prev[2], far_prev[3])
                if spike_idx + offset < len(points):
                    far_next = points[spike_idx + offset]
                    if not is_at_origin(far_next):
                        return (frame, far_next[1], far_next[2], far_next[3])
            # Can't find valid neighbors, keep original
            return original_point
        
        # If one neighbor is at origin, use the other one's position
        if prev_at_origin:
            if not is_at_origin(next_point):
                return (frame, next_point[1], next_point[2], next_point[3])
            # Next is also at origin (shouldn't happen after check above, but just in case)
            if not is_at_origin(original_point):
                return original_point
        if next_at_origin:
            if not is_at_origin(prev_point):
                return (frame, prev_point[1], prev_point[2], prev_point[3])
            # Prev is also at origin
            if not is_at_origin(original_point):
                return original_point
        
        # Calculate interpolation parameter based on frame numbers
        frame_diff = next_point[0] - prev_point[0]
        if frame_diff == 0:
            # Same frame for neighbors - use average
            t = 0.5
        else:
            t = (frame - prev_point[0]) / frame_diff
            t = max(0.0, min(1.0, t))
        
        x = prev_point[1] + t * (next_point[1] - prev_point[1])
        y = prev_point[2] + t * (next_point[2] - prev_point[2])
        z = prev_point[3] + t * (next_point[3] - prev_point[3])
        
        return (frame, float(x), float(y), float(z))
    
    def smooth_track(self, points: List[Tuple[int, float, float, float]]) -> List[Tuple[int, float, float, float]]:
        """Apply smoothing to a track."""
        if len(points) < 2:
            return points
        
        frames = [p[0] for p in points]
        xs = np.array([p[1] for p in points])
        ys = np.array([p[2] for p in points])
        zs = np.array([p[3] for p in points])
        
        window = int(self.window_var.get())
        if window < 1:
            window = 1
        if window > len(points):
            window = len(points)
        
        # Helper function to check if point is at origin
        def is_at_origin(idx):
            return xs[idx] == 0.0 and ys[idx] == 0.0 and zs[idx] == 0.0
        
        # Use a proper moving average that only averages with actual valid data points
        # Exclude (0,0,0) points from the average to prevent trails from jumping to origin
        smoothed_xs = np.zeros_like(xs)
        smoothed_ys = np.zeros_like(ys)
        smoothed_zs = np.zeros_like(zs)
        
        half_window = window // 2
        
        for i in range(len(points)):
            # If current point is at origin, try to interpolate from valid neighbors
            if is_at_origin(i):
                # Try to find valid neighbors to interpolate from
                start_idx = max(0, i - half_window)
                end_idx = min(len(points), i + half_window + 1)
                
                # Find valid (non-origin) points in window
                valid_xs = [xs[j] for j in range(start_idx, end_idx) if not is_at_origin(j)]
                valid_ys = [ys[j] for j in range(start_idx, end_idx) if not is_at_origin(j)]
                valid_zs = [zs[j] for j in range(start_idx, end_idx) if not is_at_origin(j)]
                
                if valid_xs:  # If we have valid neighbors, interpolate
                    smoothed_xs[i] = np.mean(valid_xs)
                    smoothed_ys[i] = np.mean(valid_ys)
                    smoothed_zs[i] = np.mean(valid_zs)
                else:  # No valid neighbors, keep original (0,0,0)
                    smoothed_xs[i] = xs[i]
                    smoothed_ys[i] = ys[i]
                    smoothed_zs[i] = zs[i]
            else:
                # Current point is valid, calculate window boundaries
                start_idx = max(0, i - half_window)
                end_idx = min(len(points), i + half_window + 1)
                
                # Find valid (non-origin) points in window, excluding origin points
                valid_xs = [xs[j] for j in range(start_idx, end_idx) if not is_at_origin(j)]
                valid_ys = [ys[j] for j in range(start_idx, end_idx) if not is_at_origin(j)]
                valid_zs = [zs[j] for j in range(start_idx, end_idx) if not is_at_origin(j)]
                
                if valid_xs:  # Average only valid points
                    smoothed_xs[i] = np.mean(valid_xs)
                    smoothed_ys[i] = np.mean(valid_ys)
                    smoothed_zs[i] = np.mean(valid_zs)
                else:  # No valid points in window, keep original
                    smoothed_xs[i] = xs[i]
                    smoothed_ys[i] = ys[i]
                    smoothed_zs[i] = zs[i]
        
        return [(frames[i], smoothed_xs[i], smoothed_ys[i], smoothed_zs[i]) for i in range(len(points))]
    
    def apply_smoothing(self):
        """Apply smoothing and spike removal to all tracks."""
        if not self.original_data:
            messagebox.showwarning("No Data", "Please load a CSV file first.")
            return
        
        self.smoothed_data = {}
        self.spike_indices = {}
        
        self.enable_smoothing = self.enable_smooth_var.get()
        self.enable_spike_removal = self.enable_spike_var.get()
        
        for track_id in self.track_ids:
            original_points = list(self.original_data[track_id])
            points = list(original_points)  # Working copy
            
            # Step 1: Detect and remove spikes
            if self.enable_spike_removal:
                spike_idxs = self.detect_spikes(track_id, original_points)
                self.spike_indices[track_id] = spike_idxs
                
                # Process spikes in reverse order, using ORIGINAL points for interpolation
                # This prevents using already-modified points when dealing with consecutive spikes
                for idx in reversed(sorted(spike_idxs)):
                    if 0 <= idx < len(original_points):
                        # Always use original_points for interpolation, not the modified points
                        interpolated = self.interpolate_point(original_points, idx)
                        
                        # Validate interpolated point (shouldn't be NaN or at origin unless original was)
                        if not (np.isnan(interpolated[1]) or np.isnan(interpolated[2]) or np.isnan(interpolated[3])):
                            # Check if interpolation resulted in (0,0,0) when original wasn't at origin
                            orig_point = original_points[idx]
                            if interpolated[1] == 0.0 and interpolated[2] == 0.0 and interpolated[3] == 0.0:
                                if not (orig_point[1] == 0.0 and orig_point[2] == 0.0 and orig_point[3] == 0.0):
                                    # Interpolation resulted in origin unexpectedly - use average of neighbors instead
                                    if idx > 0 and idx < len(original_points) - 1:
                                        prev = original_points[idx - 1]
                                        next = original_points[idx + 1]
                                        interpolated = (
                                            interpolated[0],
                                            (prev[1] + next[1]) / 2.0,
                                            (prev[2] + next[2]) / 2.0,
                                            (prev[3] + next[3]) / 2.0
                                        )
                            
                            points[idx] = interpolated
            
            # Step 2: Apply smoothing
            if self.enable_smoothing:
                points = self.smooth_track(points)
            
            self.smoothed_data[track_id] = points
        
        # Update frame_data with smoothed values
        self.frame_data = {}
        for track_id, points in self.smoothed_data.items():
            for frame, x, y, z in points:
                if frame not in self.frame_data:
                    self.frame_data[frame] = []
                self.frame_data[frame].append((track_id, x, y, z))
        
        # Update base class data for bounds calculation and plotting
        self.data = self.smoothed_data
        
        # Reset bounds so they recalculate with smoothed data
        self.bounds_set = False
        self.persistent_bounds = None
        
        # Recalculate metrics (only if metrics_text exists)
        if hasattr(self, 'metrics_text') and self.metrics_text is not None:
            self.calculate_smoothness_metrics()
        
        # Update plot
        self.update_plot()
        
        # Show summary
        total_spikes = sum(len(spikes) for spikes in self.spike_indices.values())
        if total_spikes > 0:
            messagebox.showinfo("Smoothing Applied", 
                              f"Smoothing applied to {len(self.track_ids)} tracks.\n"
                              f"Detected and removed {total_spikes} spikes.")
        else:
            messagebox.showinfo("Smoothing Applied", 
                              f"Smoothing applied to {len(self.track_ids)} tracks.\n"
                              f"No spikes detected.")
    
    def calculate_smoothness_metrics(self):
        """Calculate smoothness metrics for original and smoothed data."""
        self.smoothness_metrics = {}
        
        for track_id in self.track_ids:
            orig_points = self.original_data.get(track_id, [])
            smooth_points = self.smoothed_data.get(track_id, orig_points)
            
            def calculate_roughness(points):
                if len(points) < 2:
                    return 0.0, 0.0, 0.0
                
                velocities = []
                for i in range(1, len(points)):
                    dx = points[i][1] - points[i-1][1]
                    dy = points[i][2] - points[i-1][2]
                    dz = points[i][3] - points[i-1][3]
                    vel = np.sqrt(dx**2 + dy**2 + dz**2)
                    velocities.append(vel)
                
                if not velocities:
                    return 0.0, 0.0, 0.0
                
                accelerations = []
                for i in range(1, len(velocities)):
                    accel = abs(velocities[i] - velocities[i-1])
                    accelerations.append(accel)
                
                vel_std = np.std(velocities) if velocities else 0.0
                accel_mean = np.mean(accelerations) if accelerations else 0.0
                accel_max = np.max(accelerations) if accelerations else 0.0
                
                return vel_std, accel_mean, accel_max
            
            orig_metrics = calculate_roughness(orig_points)
            smooth_metrics = calculate_roughness(smooth_points)
            
            self.smoothness_metrics[track_id] = {
                'original': {
                    'vel_std': orig_metrics[0],
                    'accel_mean': orig_metrics[1],
                    'accel_max': orig_metrics[2]
                },
                'smoothed': {
                    'vel_std': smooth_metrics[0],
                    'accel_mean': smooth_metrics[1],
                    'accel_max': smooth_metrics[2]
                }
            }
        
        self.update_metrics_display()
    
    def update_metrics_display(self):
        """Update the smoothness metrics text display."""
        if self.metrics_text is None:
            return  # Metrics window not open
        
        self.metrics_text.config(state="normal")
        self.metrics_text.delete(1.0, tk.END)
        
        if not self.smoothness_metrics:
            self.metrics_text.insert(tk.END, "No metrics available")
            self.metrics_text.config(state="disabled")
            return
        
        tracks_to_show = self.selected_tracks if self.selected_tracks else self.track_ids
        
        if not tracks_to_show:
            self.metrics_text.insert(tk.END, "No tracks selected")
            self.metrics_text.config(state="disabled")
            return
        
        self.metrics_text.insert(tk.END, "Track  | Vel Std | Accel Mean | Accel Max\n")
        self.metrics_text.insert(tk.END, "-" * 50 + "\n")
        
        for track_id in sorted(tracks_to_show):
            if track_id not in self.smoothness_metrics:
                continue
            
            m = self.smoothness_metrics[track_id]
            orig = m['original']
            smooth = m['smoothed']
            
            vel_improvement = ((orig['vel_std'] - smooth['vel_std']) / max(orig['vel_std'], 0.001)) * 100 if orig['vel_std'] > 0 else 0
            
            self.metrics_text.insert(tk.END, f"\nTrack {track_id}:\n")
            self.metrics_text.insert(tk.END, f"  Original:  {orig['vel_std']:.3f} | {orig['accel_mean']:.3f} | {orig['accel_max']:.3f}\n")
            self.metrics_text.insert(tk.END, f"  Smoothed:  {smooth['vel_std']:.3f} | {smooth['accel_mean']:.3f} | {smooth['accel_max']:.3f}\n")
            self.metrics_text.insert(tk.END, f"  Improvement: {vel_improvement:.1f}%\n")
        
        self.metrics_text.config(state="disabled")
    
    def on_track_selection_changed(self, event=None):
        """Override to also update metrics display."""
        super().on_track_selection_changed(event)
        self.update_metrics_display()
    
    def get_plot_title(self):
        """Override to customize title."""
        return f'Track Smoothing - Frame {self.current_frame}/{self.max_frame}'
    
    def save_video(self):
        """Override to customize output filename with -smoothed suffix."""
        if not self.data:
            messagebox.showwarning("No Data", "Please load a CSV file first.")
            return
        
        # Ask for FPS using custom dialog with Match Input button
        from lib.visualizing.base_visualizer import FPSDialog
        fps_dialog = FPSDialog(self.root, initial_fps=30, csv_path=self.csv_path)
        fps = fps_dialog.result
        
        if fps is None:
            return
        
        # Generate output filename based on CSV filename and save in same folder as CSV
        if self.csv_path:
            csv_path_obj = Path(self.csv_path)
            csv_dir = csv_path_obj.parent
            csv_name = csv_path_obj.stem  # Full filename without extension
            # Output: *csvname*-3dplot-smoothed.mp4, or *csvname*-3dplot-smoothed-N.mp4 if multiple exist
            base_output = csv_dir / f"{csv_name}-3dplot-smoothed.mp4"
            counter = 1
            output_file = base_output
            while output_file.exists():
                output_file = csv_dir / f"{csv_name}-3dplot-smoothed-{counter}.mp4"
                counter += 1
        else:
            # Fallback: use 3dvis_output folder
            output_dir = Path("3dvis_output")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"3dvis_smoothed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        output_file = str(output_file)
        
        # Disable controls during export
        for widget in self.root.winfo_children():
            for child in widget.winfo_children():
                try:
                    child.config(state='disabled')
                except:
                    pass
        
        try:
            # Setup video writer
            writer = FFMpegWriter(fps=fps, metadata=dict(title='Track Smoothing', artist='3D-Cam'))
            
            total_frames = self.max_frame + 1
            progress_marks = set(int(total_frames * q / 20) for q in range(1, 20))
            
            print(f"[INFO] Starting video export: {output_file}")
            print(f"[INFO] FPS: {fps}, Total frames: {total_frames}")
            
            with writer.saving(self.fig, output_file, dpi=100):
                # Save current frame state
                original_frame = self.current_frame
                
                # Iterate through all frames
                for frame_num in range(total_frames):
                    # Update to this frame
                    self.current_frame = frame_num
                    self.playback_controller.set_frame(frame_num)
                    
                    # Update the plot
                    self.update_plot()
                    
                    # Draw the frame (only if plot window is open)
                    if self.canvas:
                        self.canvas.draw()
                    
                    # Grab the frame
                    writer.grab_frame()
                    
                    # Progress updates
                    if frame_num in progress_marks:
                        pct = 100 * frame_num / max(1, total_frames)
                        print(f"[INFO] Export progress: {pct:.1f}% ({frame_num}/{total_frames})")
                
                # Restore original frame
                self.current_frame = original_frame
                self.playback_controller.set_frame(original_frame)
            
            print(f"[INFO] Video export complete: {output_file}")
            messagebox.showinfo("Export Complete", f"Video saved to:\n{output_file}")
            
            # Final update to restore view
            self.update_plot()
            
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Video export failed: {error_msg}")
            messagebox.showerror("Export Error", f"Failed to export video:\n{error_msg}\n\n"
                               "Make sure FFmpeg is installed and available in your PATH.")
        finally:
            # Re-enable controls
            for widget in self.root.winfo_children():
                for child in widget.winfo_children():
                    try:
                        child.config(state='normal')
                    except:
                        pass
    
    def update_plot(self):
        """Override to show original, smoothed, and spikes."""
        # Check if plot window is open
        if not self.ax or not self.plot_window:
            return
        
        self.ax.clear()
        
        # If original_data is empty but self.data exists (initial load), initialize from self.data
        if not self.original_data and self.data:
            self.original_data = {tid: list(points) for tid, points in self.data.items()}
            self.smoothed_data = {tid: list(points) for tid, points in self.data.items()}
            # Update frame_data to use smoothed data
            self.frame_data = {}
            for track_id, points in self.smoothed_data.items():
                for frame, x, y, z in points:
                    if frame not in self.frame_data:
                        self.frame_data[frame] = []
                    self.frame_data[frame].append((track_id, x, y, z))
        
        if not self.original_data:
            self.ax.text(0.5, 0.5, 0.5, "No data loaded", transform=self.ax.transAxes, ha="center")
            self.canvas.draw()
            return
        
        tracks_to_show = self.selected_tracks if self.selected_tracks else self.track_ids
        
        if not tracks_to_show:
            self.ax.text(0.5, 0.5, 0.5, "No tracks selected", transform=self.ax.transAxes, ha="center")
            self.canvas.draw()
            return
        
        # Get all unique track IDs from all data sources to build color map
        all_track_ids = set(self.track_ids)
        all_track_ids.update(self.original_data.keys())
        all_track_ids.update(self.smoothed_data.keys())
        for frame_data_list in self.frame_data.values():
            all_track_ids.update(track_id for track_id, _, _, _ in frame_data_list)
        all_track_ids = sorted(all_track_ids)
        
        colors = cm.tab20(np.linspace(0, 1, max(len(all_track_ids), 1)))
        track_color_map = {tid: colors[i % len(colors)] for i, tid in enumerate(all_track_ids)}
        
        # Plot original tracks
        if self.show_orig_var.get():
            for track_id in tracks_to_show:
                if track_id not in self.original_data:
                    continue
                
                points = self.original_data[track_id]
                frames = [p[0] for p in points]
                xs = [p[1] for p in points]
                ys = [p[2] for p in points]
                zs = [p[3] for p in points]
                
                color = track_color_map[track_id]
                
                if self.show_trails:
                    # Show trail: points up to and including current frame, respecting trail_length
                    trail_start = max(0, self.current_frame - self.trail_length)
                    trail_mask = [f >= trail_start and f <= self.current_frame for f in frames]
                    
                    if any(trail_mask):
                        trail_xs = [x for x, m in zip(xs, trail_mask) if m]
                        trail_ys = [y for y, m in zip(ys, trail_mask) if m]
                        trail_zs = [z for z, m in zip(zs, trail_mask) if m]
                        
                        self.ax.plot(trail_xs, trail_ys, trail_zs, color=color, alpha=0.3,
                                   linewidth=1.5, linestyle='--', label=f"Track {track_id} (orig)")
                
                # Show current position marker even when trails are off
                current_mask = [f == self.current_frame for f in frames]
                if any(current_mask):
                    idx = [i for i, m in enumerate(current_mask) if m][0]
                    self.ax.scatter([xs[idx]], [ys[idx]], [zs[idx]], 
                                  color=color, s=100, marker='o', edgecolors='black', linewidths=2, alpha=0.5,
                                  label=f"Track {track_id} (orig current)")
        
        # Plot smoothed tracks
        if self.show_smooth_var.get():
            for track_id in tracks_to_show:
                if track_id not in self.smoothed_data:
                    continue
                
                points = self.smoothed_data[track_id]
                frames = [p[0] for p in points]
                xs = [p[1] for p in points]
                ys = [p[2] for p in points]
                zs = [p[3] for p in points]
                
                color = track_color_map[track_id]
                
                if self.show_trails:
                    # Show trail: points up to and including current frame, respecting trail_length
                    trail_start = max(0, self.current_frame - self.trail_length)
                    trail_mask = [f >= trail_start and f <= self.current_frame for f in frames]
                    
                    if any(trail_mask):
                        trail_xs = [x for x, m in zip(xs, trail_mask) if m]
                        trail_ys = [y for y, m in zip(ys, trail_mask) if m]
                        trail_zs = [z for z, m in zip(zs, trail_mask) if m]
                        
                        self.ax.plot(trail_xs, trail_ys, trail_zs, color=color, alpha=0.8,
                                   linewidth=2, label=f"Track {track_id} (smooth)")
                
                # Show current position marker even when trails are off
                current_mask = [f == self.current_frame for f in frames]
                if any(current_mask):
                    idx = [i for i, m in enumerate(current_mask) if m][0]
                    self.ax.scatter([xs[idx]], [ys[idx]], [zs[idx]], 
                                  color=color, s=100, marker='o', edgecolors='black', linewidths=2,
                                  label=f"Track {track_id} (smooth current)")
        
        # Highlight detected spikes
        if self.show_spikes_var.get() and self.spike_indices:
            for track_id in tracks_to_show:
                if track_id not in self.spike_indices:
                    continue
                
                orig_points = self.original_data[track_id]
                spike_idxs = self.spike_indices[track_id]
                
                for idx in spike_idxs:
                    if idx < len(orig_points):
                        frame, x, y, z = orig_points[idx]
                        if frame <= self.current_frame:
                            self.ax.scatter([x], [y], [z], color='red', s=200, marker='X',
                                          edgecolors='darkred', linewidths=2, zorder=10)
        
        # Current frame markers
        if self.current_frame in self.frame_data:
            current_points = self.frame_data[self.current_frame]
            for track_id, x, y, z in current_points:
                if track_id in tracks_to_show:
                    color = track_color_map[track_id]
                    self.ax.scatter([x], [y], [z], color=color, s=150, marker='*',
                                  edgecolors='black', linewidths=2, zorder=5)
                    
                    # Add track number label if enabled
                    if self.show_labels_var.get():
                        # Calculate offset for upper-right positioning (small percentage of axis range)
                        xlim = self.ax.get_xlim()
                        ylim = self.ax.get_ylim()
                        zlim = self.ax.get_zlim()
                        x_offset = (xlim[1] - xlim[0]) * 0.03  # 3% of x range
                        y_offset = (ylim[1] - ylim[0]) * 0.03  # 3% of y range
                        z_offset = (zlim[1] - zlim[0]) * 0.03  # 3% of z range
                        self.ax.text(x + x_offset, y + y_offset, z + z_offset, 
                                   f'{track_id}', fontsize=16, color='black', weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black', linewidth=1))
        
        # Draw camera centerline if enabled
        if self.show_centerline_var.get():
            self.draw_optical_center_column(tracks_to_show)
        
        # Labels - handle empty units for Z (Zprime or Zdoubleprime)
        x_label = f'X ({self.x_unit})' if self.x_unit else 'X'
        y_label = f'Y ({self.y_unit})' if self.y_unit else 'Y'
        z_label = f'Z ({self.z_unit})' if self.z_unit else 'Z'
        self.ax.set_xlabel(x_label, fontsize=10)
        self.ax.set_ylabel(y_label, fontsize=10)
        self.ax.set_zlabel(z_label, fontsize=10)
        self.ax.set_title(self.get_plot_title())
        
        # Set bounds
        if tracks_to_show and not self.bounds_set:
            self.set_optimal_bounds(tracks_to_show)
            self.bounds_set = True
        elif self.persistent_bounds is not None:
            xlim, ylim, zlim = self.persistent_bounds
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.set_zlim(zlim)
        
        # Set optimal viewing angle (only on first load)
        if not hasattr(self, '_view_set'):
            self.set_optimal_view()
            self._view_set = True
        
        # Grid
        self.ax.grid(True, alpha=0.3)
        
        # Add legend if not too many tracks and there are labeled artists
        if len(tracks_to_show) <= 10:
            handles, labels = self.ax.get_legend_handles_labels()
            if handles:  # Only show legend if there are labeled artists
                self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        self.canvas.draw()
    
    def export_cleaned(self):
        """Export smoothed data to a new CSV file."""
        if not self.smoothed_data:
            messagebox.showwarning("No Data", "Please load and smooth data first.")
            return
        
        if self.csv_path:
            csv_path_obj = Path(self.csv_path)
            # Use the same folder as the input CSV
            csv_dir = csv_path_obj.parent
            # Extract full CSV filename without extension
            csv_name = csv_path_obj.stem
            
            # Output: *csvname*-smoothed.csv, or *csvname*-smoothed-N.csv if multiple exist
            base_output = csv_dir / f"{csv_name}-smoothed.csv"
            counter = 1
            output_file = base_output
            while output_file.exists():
                output_file = csv_dir / f"{csv_name}-smoothed-{counter}.csv"
                counter += 1
        else:
            # Fallback: use inputs_outputs/data if no CSV path
            data_dir = Path("inputs_outputs/data")
            data_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = data_dir / f"smoothed_{timestamp}.csv"
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames)
                original_rows = list(reader)
            
            # Add new smoothed columns to fieldnames if they don't exist
            smoothed_columns = []
            if 'X_mm' in fieldnames:
                smoothed_columns.append('X_mm_smoothed')
            if 'Y_mm' in fieldnames:
                smoothed_columns.append('Y_mm_smoothed')
            if 'Z_mm' in fieldnames:
                smoothed_columns.append('Z_mm_smoothed')
            
            # Add smoothed columns after the original mm columns
            new_fieldnames = fieldnames.copy()
            for col in smoothed_columns:
                if col not in new_fieldnames:
                    # Insert after the corresponding original column
                    if col == 'X_mm_smoothed' and 'X_mm' in new_fieldnames:
                        idx = new_fieldnames.index('X_mm') + 1
                        new_fieldnames.insert(idx, 'X_mm_smoothed')
                    elif col == 'Y_mm_smoothed' and 'Y_mm' in new_fieldnames:
                        idx = new_fieldnames.index('Y_mm') + 1
                        new_fieldnames.insert(idx, 'Y_mm_smoothed')
                    elif col == 'Z_mm_smoothed' and 'Z_mm' in new_fieldnames:
                        idx = new_fieldnames.index('Z_mm') + 1
                        new_fieldnames.insert(idx, 'Z_mm_smoothed')
            
            smoothed_map = {}
            for track_id, points in self.smoothed_data.items():
                for frame, x, y, z in points:
                    smoothed_map[(frame, track_id)] = (x, y, z)
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=new_fieldnames)
                writer.writeheader()
                
                for row in original_rows:
                    try:
                        frame = int(row['Frame_Number'])
                        track_id = int(row['Track_ID'])
                        
                        if (frame, track_id) in smoothed_map:
                            x, y, z = smoothed_map[(frame, track_id)]
                            # Write smoothed coordinates to new smoothed columns
                            if 'X_mm_smoothed' in new_fieldnames:
                                row['X_mm_smoothed'] = f"{x:.4f}"
                            if 'Y_mm_smoothed' in new_fieldnames:
                                row['Y_mm_smoothed'] = f"{y:.4f}"
                            if 'Z_mm_smoothed' in new_fieldnames:
                                row['Z_mm_smoothed'] = f"{z:.4f}"
                        
                        writer.writerow(row)
                    
                    except (ValueError, KeyError):
                        continue
            
            messagebox.showinfo("Export Complete", f"Smoothed data exported to:\n{output_file}")
            
            # Auto-save smoothing preset to folder with counter if multiple exist
            if self.csv_path:
                csv_dir = csv_path_obj.parent
                base_preset = csv_dir / "track_smoother_preset.json"
                counter = 1
                preset_path = base_preset
                while preset_path.exists():
                    preset_path = csv_dir / f"track_smoother_preset-{counter}.json"
                    counter += 1
                # Get the actual smoothing window size from UI if available
                window_size = self.window_var.get() if hasattr(self, 'window_var') else self.smoothing_window_size
                
                smoothing_data = {
                    "smoothing_window": int(window_size),
                    "spike_threshold": self.spike_threshold,
                    "spike_velocity_threshold": self.spike_velocity_threshold,
                    "enable_smoothing": self.enable_smoothing,
                    "enable_spike_removal": self.enable_spike_removal,
                }
                
                try:
                    with open(preset_path, 'w', encoding='utf-8') as f:
                        json.dump(smoothing_data, f, indent=2)
                    print(f"[INFO] Smoothing preset saved to folder: {preset_path}")
                except Exception as e:
                    print(f"[WARN] Failed to save smoothing preset: {e}")
        
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CSV:\n{e}")


def main():
    root = tk.Tk()
    app = TrackSmoother(root)
    root.mainloop()


if __name__ == "__main__":
    main()
