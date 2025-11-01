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

import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FFMpegWriter
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime


class Pair3DVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Pair Trajectory Visualizer")
        self.root.geometry("1200x800")
        
        # Data storage
        self.csv_path = None
        self.data = {}  # {track_id: [(frame, x, y, z), ...]}
        self.frame_data = {}  # {frame: [(track_id, x, y, z), ...]}
        self.max_frame = 0
        self.current_frame = 0
        self.track_ids = []
        
        # Visualization settings
        self.show_trails = True
        self.trail_length = 50  # frames
        self.selected_tracks = set()  # Empty = show all
        self.bounds_set = False  # Track if bounds have been set
        self.persistent_bounds = None  # Store persistent bounds: (xlim, ylim, zlim)
        
        # Setup UI
        self.setup_ui()
        
        # Load latest CSV if available
        self.auto_load_latest_csv()
    
    def setup_ui(self):
        """Create the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Left panel - controls
        left_panel = ttk.Frame(main_frame, width=250)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Load button
        ttk.Button(left_panel, text="Load CSV File", command=self.load_csv).pack(pady=5, fill="x")
        
        # File label
        self.file_label = ttk.Label(left_panel, text="No file loaded", wraplength=230)
        self.file_label.pack(pady=5)
        
        # Separator
        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", pady=10)
        
        # Time controls
        time_frame = ttk.LabelFrame(left_panel, text="Time Control", padding="10")
        time_frame.pack(fill="x", pady=5)
        
        self.frame_var = tk.IntVar(value=0)
        self.frame_label = ttk.Label(time_frame, text="Frame: 0 / 0")
        self.frame_label.pack()
        
        self.frame_slider = ttk.Scale(
            time_frame,
            from_=0,
            to=0,
            orient="horizontal",
            variable=self.frame_var,
            command=self.on_frame_changed
        )
        self.frame_slider.pack(fill="x", pady=5)
        
        # Playback controls
        playback_frame = ttk.Frame(time_frame)
        playback_frame.pack(fill="x")
        
        self.playing = False
        self.play_button = ttk.Button(playback_frame, text="▶ Play", command=self.toggle_play)
        self.play_button.pack(side="left", padx=2)
        
        ttk.Button(playback_frame, text="⏮", command=lambda: self.set_frame(0)).pack(side="left", padx=2)
        ttk.Button(playback_frame, text="⏭", command=lambda: self.set_frame(self.max_frame)).pack(side="left", padx=2)
        
        # Speed control
        speed_frame = ttk.Frame(time_frame)
        speed_frame.pack(fill="x", pady=5)
        ttk.Label(speed_frame, text="Speed:").pack(side="left")
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(speed_frame, from_=0.1, to=5.0, orient="horizontal", variable=self.speed_var, length=150)
        speed_scale.pack(side="left", fill="x", expand=True)
        self.speed_label = ttk.Label(speed_frame, text="1.0x")
        self.speed_label.pack(side="left", padx=5)
        speed_scale.configure(command=lambda v: self.speed_label.config(text=f"{float(v):.1f}x"))
        
        # Display options
        display_frame = ttk.LabelFrame(left_panel, text="Display Options", padding="10")
        display_frame.pack(fill="x", pady=5)
        
        self.trails_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show Trails", variable=self.trails_var, command=self.update_display).pack(anchor="w")
        
        trail_length_frame = ttk.Frame(display_frame)
        trail_length_frame.pack(fill="x", pady=5)
        ttk.Label(trail_length_frame, text="Trail Length:").pack(side="left")
        self.trail_length_var = tk.IntVar(value=50)
        trail_scale = ttk.Scale(trail_length_frame, from_=1, to=200, orient="horizontal", variable=self.trail_length_var, length=150)
        trail_scale.pack(side="left", fill="x", expand=True)
        trail_scale.configure(command=lambda v: self.on_trail_length_changed())
        
        self.show_all_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show All Tracks", variable=self.show_all_var, command=self.on_show_all_changed).pack(anchor="w", pady=5)
        
        # Track selection
        track_frame = ttk.LabelFrame(left_panel, text="Track Selection", padding="10")
        track_frame.pack(fill="both", expand=True, pady=5)
        
        # Scrollable listbox for tracks
        listbox_frame = ttk.Frame(track_frame)
        listbox_frame.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.track_listbox = tk.Listbox(listbox_frame, selectmode=tk.EXTENDED, yscrollcommand=scrollbar.set)
        self.track_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.track_listbox.yview)
        
        self.track_listbox.bind('<<ListboxSelect>>', self.on_track_selection_changed)
        
        ttk.Button(track_frame, text="Select All", command=self.select_all_tracks).pack(pady=5, fill="x")
        ttk.Button(track_frame, text="Clear Selection", command=self.clear_track_selection).pack(pady=5, fill="x")
        
        # Reset view button
        ttk.Button(display_frame, text="Reset View", command=self.reset_view).pack(pady=5, fill="x")
        
        # Export button
        export_frame = ttk.LabelFrame(left_panel, text="Export", padding="10")
        export_frame.pack(fill="x", pady=5)
        ttk.Button(export_frame, text="Save Video", command=self.save_video).pack(pady=5, fill="x")
        
        # Info display
        info_frame = ttk.LabelFrame(left_panel, text="Info", padding="10")
        info_frame.pack(fill="x", pady=5)
        self.info_label = ttk.Label(info_frame, text="No data loaded", wraplength=230, justify="left")
        self.info_label.pack()
        
        # Right panel - 3D plot
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        toolbar.update()
        
        # Bind mouse wheel events for zooming
        self.setup_zoom()
        
        # Initial empty plot
        self.update_plot()
    
    def setup_zoom(self):
        """Setup mouse wheel zoom functionality."""
        canvas_widget = self.canvas.get_tk_widget()
        
        def on_scroll(event):
            """Handle mouse wheel scroll for zooming."""
            # Zoom factor
            zoom_factor = 1.1 if event.delta > 0 or event.num == 4 else 0.9
            
            # Get current axis limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            zlim = self.ax.get_zlim()
            
            # Calculate centers
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            z_center = (zlim[0] + zlim[1]) / 2
            
            # Calculate ranges
            x_range = (xlim[1] - xlim[0]) / zoom_factor
            y_range = (ylim[1] - ylim[0]) / zoom_factor
            z_range = (zlim[1] - zlim[0]) / zoom_factor
            
            # Set new limits (zoom in/out around center)
            # Ensure Z doesn't go below 0 when zooming
            new_z_min = max(0, z_center - z_range/2)
            new_z_max = z_center + z_range/2
            
            new_xlim = (x_center - x_range/2, x_center + x_range/2)
            new_ylim = (y_center - y_range/2, y_center + y_range/2)
            new_zlim = (new_z_min, new_z_max)
            
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.ax.set_zlim(new_zlim)
            
            # Update persistent bounds to maintain zoom state
            self.persistent_bounds = (new_xlim, new_ylim, new_zlim)
            
            self.canvas.draw()
            return "break"
        
        # Bind mouse wheel events (Windows and Mac)
        canvas_widget.bind("<MouseWheel>", on_scroll)
        # Linux uses Button-4 and Button-5
        canvas_widget.bind("<Button-4>", on_scroll)
        canvas_widget.bind("<Button-5>", on_scroll)
        
        # Make sure the widget can receive focus for mouse wheel events
        canvas_widget.focus_set()
        
        # Also bind when mouse enters the widget
        def on_enter(event):
            canvas_widget.focus_set()
        
        canvas_widget.bind("<Enter>", on_enter)
    
    def auto_load_latest_csv(self):
        """Automatically load the latest CSV file from pair_detect_output."""
        output_dir = Path("pair_detect_output")
        if not output_dir.exists():
            return
        
        # Find all CSV files
        csv_files = list(output_dir.rglob("pairs.csv"))
        if not csv_files:
            return
        
        # Sort by modification time (newest first)
        csv_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        latest_csv = csv_files[0]
        
        # Try to load it
        try:
            self.load_csv_file(str(latest_csv))
        except Exception as e:
            print(f"Failed to auto-load {latest_csv}: {e}")
    
    def load_csv(self):
        """Open file dialog to load CSV file."""
        file_path = filedialog.askopenfilename(
            title="Load Pair CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.load_csv_file(file_path)
    
    def load_csv_file(self, file_path: str):
        """Load and parse CSV file."""
        self.csv_path = file_path
        self.data = {}
        self.frame_data = {}
        self.max_frame = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Check if X, Y, Z columns exist
                has_xyz = all(col in reader.fieldnames for col in ['X_mm', 'Y_mm', 'Z_mm'])
                
                if not has_xyz:
                    # Try to calculate from available data if calibration is present
                    messagebox.showwarning(
                        "Missing XYZ Data",
                        "CSV file doesn't contain X_mm, Y_mm, Z_mm columns.\n\n"
                        "These are only available if calibration data was loaded during export.\n"
                        "Please re-export with calibration data to visualize 3D trajectories."
                    )
                    return
                
                for row in reader:
                    try:
                        frame = int(row['Frame_Number'])
                        track_id = int(row['Track_ID'])
                        
                        # Get X, Y, Z values (may be empty strings)
                        x_str = row.get('X_mm', '').strip()
                        y_str = row.get('Y_mm', '').strip()
                        z_str = row.get('Z_mm', '').strip()
                        
                        # Skip rows with missing XYZ data
                        if not x_str or not y_str or not z_str:
                            continue
                        
                        x = float(x_str)
                        y = float(y_str)
                        z = float(z_str)
                        
                        # Store in track-based structure
                        if track_id not in self.data:
                            self.data[track_id] = []
                        self.data[track_id].append((frame, x, y, z))
                        
                        # Store in frame-based structure
                        if frame not in self.frame_data:
                            self.frame_data[frame] = []
                        self.frame_data[frame].append((track_id, x, y, z))
                        
                        self.max_frame = max(self.max_frame, frame)
                    
                    except (ValueError, KeyError) as e:
                        continue
            
            # Sort data by frame for each track
            for track_id in self.data:
                self.data[track_id].sort(key=lambda x: x[0])
            
            self.track_ids = sorted(self.data.keys())
            
            # Update UI
            self.file_label.config(text=f"Loaded: {Path(file_path).name}")
            self.frame_slider.config(to=self.max_frame)
            self.frame_var.set(0)
            self.current_frame = 0
            
            # Update track listbox
            self.track_listbox.delete(0, tk.END)
            for track_id in self.track_ids:
                point_count = len(self.data[track_id])
                self.track_listbox.insert(tk.END, f"Track {track_id} ({point_count} pts)")
            
            # Select all tracks by default
            self.selected_tracks = set(self.track_ids)
            self.show_all_var.set(True)
            
            # Update info
            total_points = sum(len(self.data[tid]) for tid in self.data)
            self.info_label.config(
                text=f"Tracks: {len(self.track_ids)}\n"
                     f"Frames: {self.max_frame + 1}\n"
                     f"Total Points: {total_points}"
            )
            
            # Reset view flags when loading new data
            self.bounds_set = False
            self.persistent_bounds = None
            self._view_set = False
            
            # Update plot
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file:\n{e}")
    
    def on_frame_changed(self, value=None):
        """Handle frame slider change."""
        frame = int(float(self.frame_var.get()))
        if frame != self.current_frame:
            self.current_frame = frame
            self.frame_label.config(text=f"Frame: {frame} / {self.max_frame}")
            self.update_plot()
    
    def set_frame(self, frame: int):
        """Set frame programmatically."""
        frame = max(0, min(frame, self.max_frame))
        self.frame_var.set(frame)
        self.on_frame_changed()
    
    def toggle_play(self):
        """Toggle playback."""
        if not self.data:
            return
        
        self.playing = not self.playing
        self.play_button.config(text="⏸ Pause" if self.playing else "▶ Play")
        
        if self.playing:
            self.play_loop()
    
    def play_loop(self):
        """Playback loop."""
        if not self.playing:
            return
        
        # Advance frame
        speed = self.speed_var.get()
        next_frame = self.current_frame + int(speed)
        
        if next_frame > self.max_frame:
            # Loop or stop
            next_frame = 0
        
        self.set_frame(next_frame)
        
        # Schedule next frame
        delay = max(1, int(1000 / (30 * speed)))  # ~30 FPS playback
        self.root.after(delay, self.play_loop)
    
    def on_trail_length_changed(self):
        """Handle trail length change."""
        self.trail_length = int(self.trail_length_var.get())
        self.update_plot()
    
    def on_show_all_changed(self):
        """Handle show all tracks checkbox."""
        if self.show_all_var.get():
            self.selected_tracks = set(self.track_ids)
            # Select all in listbox
            self.track_listbox.selection_set(0, tk.END)
        else:
            # Keep current selection
            pass
        self.update_plot()
    
    def select_all_tracks(self):
        """Select all tracks."""
        self.track_listbox.selection_set(0, tk.END)
        self.selected_tracks = set(self.track_ids)
        self.show_all_var.set(True)
        self.update_plot()
    
    def clear_track_selection(self):
        """Clear track selection."""
        self.track_listbox.selection_clear(0, tk.END)
        self.selected_tracks = set()
        self.show_all_var.set(False)
        self.update_plot()
    
    def on_track_selection_changed(self, event=None):
        """Handle track listbox selection change."""
        selected_indices = self.track_listbox.curselection()
        if selected_indices:
            self.selected_tracks = {self.track_ids[idx] for idx in selected_indices}
            self.show_all_var.set(len(self.selected_tracks) == len(self.track_ids))
        else:
            self.selected_tracks = set()
            self.show_all_var.set(False)
        self.update_plot()
    
    def reset_view(self):
        """Reset view to optimal bounds and angle."""
        self.bounds_set = False
        self.persistent_bounds = None
        self.set_optimal_view()
        self.update_plot()
    
    def update_display(self):
        """Update display options."""
        self.show_trails = self.trails_var.get()
        self.update_plot()
    
    def update_plot(self):
        """Update the 3D plot."""
        self.ax.clear()
        
        if not self.data:
            self.ax.text(0.5, 0.5, 0.5, "No data loaded", transform=self.ax.transAxes, ha="center")
            self.canvas.draw()
            return
        
        # Determine which tracks to show
        tracks_to_show = self.selected_tracks if self.selected_tracks else self.track_ids
        
        if not tracks_to_show:
            self.ax.text(0.5, 0.5, 0.5, "No tracks selected", transform=self.ax.transAxes, ha="center")
            self.canvas.draw()
            return
        
        # Color map for tracks
        colors = cm.tab20(np.linspace(0, 1, len(self.track_ids)))
        track_color_map = {tid: colors[i % len(colors)] for i, tid in enumerate(self.track_ids)}
        
        # Plot trajectories
        for track_id in tracks_to_show:
            if track_id not in self.data:
                continue
            
            points = self.data[track_id]
            if not points:
                continue
            
            # Extract coordinates
            frames = [p[0] for p in points]
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            zs = [p[3] for p in points]
            
            color = track_color_map[track_id]
            
            # Determine what to show based on current frame
            current_frame = self.current_frame
            
            if self.show_trails:
                # Show trail: points up to and including current frame
                trail_start = max(0, current_frame - self.trail_length)
                trail_mask = [f >= trail_start and f <= current_frame for f in frames]
                
                if any(trail_mask):
                    trail_frames = [f for f, m in zip(frames, trail_mask) if m]
                    trail_xs = [x for x, m in zip(xs, trail_mask) if m]
                    trail_ys = [y for y, m in zip(ys, trail_mask) if m]
                    trail_zs = [z for z, m in zip(zs, trail_mask) if m]
                    
                    # Plot trail line
                    self.ax.plot(trail_xs, trail_ys, trail_zs, 
                               color=color, alpha=0.6, linewidth=2, 
                               label=f"Track {track_id}")
                
                # Highlight current position
                current_mask = [f == current_frame for f in frames]
                if any(current_mask):
                    idx = [i for i, m in enumerate(current_mask) if m][0]
                    self.ax.scatter([xs[idx]], [ys[idx]], [zs[idx]], 
                                  color=color, s=100, marker='o', edgecolors='black', linewidths=2,
                                  label=f"Track {track_id} (current)")
            else:
                # Show only current frame position
                current_mask = [f == current_frame for f in frames]
                if any(current_mask):
                    idx = [i for i, m in enumerate(current_mask) if m][0]
                    self.ax.scatter([xs[idx]], [ys[idx]], [zs[idx]], 
                                  color=color, s=100, marker='o', edgecolors='black', linewidths=2,
                                  label=f"Track {track_id}")
        
        # Plot points at current frame
        if self.current_frame in self.frame_data:
            current_points = self.frame_data[self.current_frame]
            for track_id, x, y, z in current_points:
                if track_id in tracks_to_show:
                    color = track_color_map[track_id]
                    self.ax.scatter([x], [y], [z], color=color, s=150, marker='*', 
                                  edgecolors='black', linewidths=2, zorder=5)
        
        # Draw optical center column at (0, 0)
        self.draw_optical_center_column(tracks_to_show)
        
        # Set labels and title (dimensions in mm)
        self.ax.set_xlabel('X (mm)', fontsize=10)
        self.ax.set_ylabel('Y (mm)', fontsize=10)
        self.ax.set_zlabel('Z (mm)', fontsize=10)
        self.ax.set_title(f'3D Pair Trajectories - Frame {self.current_frame}/{self.max_frame}')
        
        # Set bounds only once initially, then keep them persistent
        if tracks_to_show and not self.bounds_set:
            self.set_optimal_bounds(tracks_to_show)
            self.bounds_set = True
        elif self.persistent_bounds is not None:
            # Restore persistent bounds
            xlim, ylim, zlim = self.persistent_bounds
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.set_zlim(zlim)
        
        # Set optimal viewing angle (only on first load)
        if not hasattr(self, '_view_set'):
            self.set_optimal_view()
            self._view_set = True
        
        # Add grid
        self.ax.grid(True, alpha=0.3)
        
        # Add legend if not too many tracks
        if len(tracks_to_show) <= 10:
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        self.canvas.draw()
    
    def set_optimal_bounds(self, tracks_to_show):
        """Set axis bounds to maximum extents from visible data, with Z starting at 0."""
        all_xs = [p[1] for tid in tracks_to_show for p in self.data.get(tid, [])]
        all_ys = [p[2] for tid in tracks_to_show for p in self.data.get(tid, [])]
        all_zs = [p[3] for tid in tracks_to_show for p in self.data.get(tid, [])]
        
        if not all_xs or not all_ys or not all_zs:
            return
        
        # Calculate min/max for each axis
        x_min, x_max = min(all_xs), max(all_xs)
        y_min, y_max = min(all_ys), max(all_ys)
        z_min, z_max = min(all_zs), max(all_zs)
        
        # Ensure Z starts at 0
        if z_min < 0:
            z_max = z_max - z_min  # Adjust max to maintain range
            z_min = 0
        else:
            z_min = 0  # Start at 0 even if minimum is positive
        
        # Set limits to maximum extents (no padding, no resizing)
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_zlim(z_min, z_max)
        
        # Store as persistent bounds
        self.persistent_bounds = ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    
    def draw_optical_center_column(self, tracks_to_show):
        """Draw a lightweight vertical line at the optical center (0, 0) position."""
        if not tracks_to_show:
            return
        
        # Get Z range from data to know how tall to make the column
        all_zs = [p[3] for tid in tracks_to_show for p in self.data.get(tid, [])]
        if not all_zs:
            return
        
        z_max = max(all_zs)
        z_min = 0  # Optical center column starts at Z=0
        
        # Draw a simple vertical line from Z=0 to Z=max at (0, 0)
        self.ax.plot([0, 0], [0, 0], [z_min, z_max], 
                   color='red', alpha=0.7, linewidth=3, linestyle='--',
                   label='Optical Center')
        
        # Add a simple marker at the base (0, 0, 0)
        self.ax.scatter([0], [0], [0], color='red', s=150, marker='o', 
                       edgecolors='darkred', linewidths=2, zorder=10)
    
    def save_video(self):
        """Export the 3D visualization as a video file."""
        if not self.data:
            messagebox.showwarning("No Data", "Please load a CSV file first.")
            return
        
        # Ask for FPS
        fps = tk.simpledialog.askinteger(
            "Video FPS",
            "Enter frames per second for the video:",
            initialvalue=30,
            minvalue=1,
            maxvalue=120
        )
        
        if fps is None:
            return
        
        # Create output directory
        output_dir = Path("3dvis_output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate output filename based on CSV filename or timestamp
        if self.csv_path:
            csv_name = Path(self.csv_path).stem
            output_file = output_dir / f"{csv_name}_3dvis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        else:
            output_file = output_dir / f"3dvis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
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
            writer = FFMpegWriter(fps=fps, metadata=dict(title='3D Pair Trajectories', artist='3D-Cam'))
            
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
                    self.frame_var.set(frame_num)
                    
                    # Update the plot
                    self.update_plot()
                    
                    # Draw the frame
                    self.canvas.draw()
                    
                    # Grab the frame
                    writer.grab_frame()
                    
                    # Progress updates
                    if frame_num in progress_marks:
                        pct = 100 * frame_num / max(1, total_frames)
                        print(f"[INFO] Export progress: {pct:.1f}% ({frame_num}/{total_frames})")
                
                # Restore original frame
                self.current_frame = original_frame
                self.frame_var.set(original_frame)
            
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
    
    def set_optimal_view(self):
        """Set optimal 3D viewing angle for better visualization."""
        # Set elevation (vertical angle) and azimuth (horizontal angle)
        # Elevation: 30 degrees (looking down slightly)
        # Azimuth: -135 degrees (diagonal view, showing X, Y, Z axes well)
        self.ax.view_init(elev=30, azim=-135)


def main():
    root = tk.Tk()
    app = Pair3DVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()

