"""
Base 3D Visualizer Widget
--------------------------
Reusable base widget for 3D trajectory visualizations with common UI elements.
"""

import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
from datetime import datetime
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FFMpegWriter
from typing import Dict, List, Tuple, Optional, Callable


class PlaybackController:
    """Internal playback control widget."""
    
    def __init__(self, parent_frame, on_frame_changed: Callable[[int], None],
                 max_frame: int = 0, initial_frame: int = 0, initial_speed: float = 1.0):
        self.on_frame_changed_callback = on_frame_changed
        self.max_frame = max_frame
        self.current_frame = initial_frame
        self.playing = False
        self.root = None
        
        widget = parent_frame
        while widget:
            if isinstance(widget, tk.Tk):
                self.root = widget
                break
            widget = widget.master
        
        if self.root is None:
            raise ValueError("Could not find root Tk window")
        
        self.control_frame = ttk.LabelFrame(parent_frame, text="Time Control", padding="10")
        self.control_frame.pack(fill="x", pady=5)
        
        self.frame_label = ttk.Label(self.control_frame, text=f"Frame: {initial_frame} / {max_frame}")
        self.frame_label.pack()
        
        self.frame_var = tk.IntVar(value=initial_frame)
        self.frame_slider = ttk.Scale(
            self.control_frame,
            from_=0,
            to=max_frame,
            orient="horizontal",
            variable=self.frame_var,
            command=self._on_slider_changed
        )
        self.frame_slider.pack(fill="x", pady=5)
        
        playback_frame = ttk.Frame(self.control_frame)
        playback_frame.pack(fill="x")
        
        self.play_button = ttk.Button(playback_frame, text="▶ Play", command=self.toggle_play)
        self.play_button.pack(side="left", padx=2)
        
        ttk.Button(playback_frame, text="⏮", command=lambda: self.set_frame(0)).pack(side="left", padx=2)
        ttk.Button(playback_frame, text="⏭", command=lambda: self.set_frame(self.max_frame)).pack(side="left", padx=2)
        
        speed_frame = ttk.Frame(self.control_frame)
        speed_frame.pack(fill="x", pady=5)
        ttk.Label(speed_frame, text="Speed:").pack(side="left")
        
        self.speed_var = tk.DoubleVar(value=initial_speed)
        speed_scale = ttk.Scale(
            speed_frame,
            from_=0.1,
            to=5.0,
            orient="horizontal",
            variable=self.speed_var,
            length=150,
            command=self._on_speed_changed
        )
        speed_scale.pack(side="left", fill="x", expand=True)
        
        self.speed_label = ttk.Label(speed_frame, text=f"{initial_speed:.1f}x")
        self.speed_label.pack(side="left", padx=5)
    
    def _on_slider_changed(self, value=None):
        frame = int(float(self.frame_var.get()))
        if frame != self.current_frame:
            self.current_frame = frame
            self.frame_label.config(text=f"Frame: {frame} / {self.max_frame}")
            self.on_frame_changed_callback(frame)
    
    def _on_speed_changed(self, value=None):
        speed = float(self.speed_var.get())
        self.speed_label.config(text=f"{speed:.1f}x")
    
    def set_frame(self, frame: int):
        frame = max(0, min(frame, self.max_frame))
        self.current_frame = frame
        self.frame_var.set(frame)
        self.frame_label.config(text=f"Frame: {frame} / {self.max_frame}")
        self.on_frame_changed_callback(frame)
    
    def toggle_play(self):
        if self.max_frame == 0:
            return
        self.playing = not self.playing
        self.play_button.config(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self.play_loop()
    
    def play_loop(self):
        if not self.playing:
            return
        speed = self.speed_var.get()
        next_frame = self.current_frame + int(speed)
        if next_frame > self.max_frame:
            next_frame = 0
        self.set_frame(next_frame)
        delay = max(1, int(1000 / (30 * speed)))
        self.root.after(delay, self.play_loop)
    
    def set_max_frame(self, max_frame: int):
        self.max_frame = max_frame
        self.frame_slider.config(to=max_frame)
        self.frame_label.config(text=f"Frame: {self.current_frame} / {self.max_frame}")
        if self.current_frame > max_frame:
            self.set_frame(max_frame)
    
    def stop(self):
        self.playing = False
        self.play_button.config(text="▶ Play")
    
    def is_playing(self) -> bool:
        return self.playing
    
    def get_current_frame(self) -> int:
        return self.current_frame
    
    def get_speed(self) -> float:
        return self.speed_var.get()


class Base3DVisualizer:
    """
    Base class for 3D trajectory visualizations with common UI elements.
    
    Provides:
    - Load CSV file functionality
    - 3D plot with matplotlib
    - Playback controls
    - Display options
    - Track selection
    - Export functionality
    - Info display
    """
    
    def __init__(self, root, title: str, geometry: str = "1400x900"):
        """
        Initialize base visualizer.
        
        Args:
            root: Tkinter root window
            title: Window title
            geometry: Window geometry string (default: "1400x900")
        """
        self.root = root
        self.root.title(title)
        self.root.geometry(geometry)
        
        # Data storage (to be populated by subclasses)
        self.csv_path = None
        self.data = {}  # {track_id: [(frame, x, y, z), ...]}
        self.frame_data = {}  # {frame: [(track_id, x, y, z), ...]}
        self.max_frame = 0
        self.current_frame = 0
        self.track_ids = []
        self.selected_tracks = set()
        
        # Visualization settings
        self.show_trails = True
        self.trail_length = 50
        self.bounds_set = False
        self.persistent_bounds = None
        self.min_track_length = 0  # Filter tracks by minimum length
        self.track_checkboxes = {}  # {track_id: (var, checkbox_widget)}
        
        # Setup UI
        self.setup_ui()
        
        # Auto-load latest CSV
        self.auto_load_latest_csv()
    
    def setup_ui(self):
        """Create the user interface with common elements."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Left panel - controls (two-column layout)
        self.left_panel = ttk.Frame(main_frame, width=600)
        self.left_panel.pack(side="left", fill="y", padx=(0, 10))
        self.left_panel.pack_propagate(False)
        
        # Create two-column layout
        left_col = ttk.Frame(self.left_panel)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        right_col = ttk.Frame(self.left_panel)
        right_col.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        # LEFT COLUMN
        # Load CSV button
        ttk.Button(left_col, text="Load CSV File", command=self.load_csv).pack(pady=5, fill="x")
        
        # File label
        self.file_label = ttk.Label(left_col, text="No file loaded", wraplength=120)
        self.file_label.pack(pady=5)
        
        # Separator
        ttk.Separator(left_col, orient="horizontal").pack(fill="x", pady=10)
        
        # Custom section (for subclasses to add their own controls)
        self.custom_section = ttk.Frame(left_col)
        self.custom_section.pack(fill="x", pady=5)
        
        # Display options
        display_frame = ttk.LabelFrame(left_col, text="Display Options", padding="10")
        display_frame.pack(fill="x", pady=5)
        
        self.trails_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show Trails", variable=self.trails_var,
                       command=self.on_display_changed).pack(anchor="w")
        
        trail_length_frame = ttk.Frame(display_frame)
        trail_length_frame.pack(fill="x", pady=5)
        ttk.Label(trail_length_frame, text="Trail Length:").pack(side="left")
        self.trail_length_var = tk.IntVar(value=50)
        trail_scale = ttk.Scale(trail_length_frame, from_=1, to=200, orient="horizontal",
                               variable=self.trail_length_var, length=100,
                               command=self.on_trail_length_changed)
        trail_scale.pack(side="left", fill="x", expand=True)
        
        self.show_all_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show All Tracks", variable=self.show_all_var,
                       command=self.on_show_all_changed).pack(anchor="w", pady=5)
        
        self.show_labels_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(display_frame, text="Show Track Labels", variable=self.show_labels_var,
                       command=self.on_display_changed).pack(anchor="w")
        
        self.show_centerline_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(display_frame, text="Show Camera Centerline", variable=self.show_centerline_var,
                       command=self.on_display_changed).pack(anchor="w")
        
        ttk.Button(display_frame, text="Reset View", command=self.reset_view).pack(pady=5, fill="x")
        
        # Export section
        export_frame = ttk.LabelFrame(left_col, text="Export", padding="10")
        export_frame.pack(fill="x", pady=5)
        self.setup_export_section(export_frame)
        
        # Info display
        info_frame = ttk.LabelFrame(left_col, text="Info", padding="10")
        info_frame.pack(fill="x", pady=5)
        self.info_label = ttk.Label(info_frame, text="No data loaded", wraplength=120, justify="left")
        self.info_label.pack()
        
        # RIGHT COLUMN
        # Track selection
        track_frame = ttk.LabelFrame(right_col, text="Track Selection", padding="10")
        track_frame.pack(fill="both", expand=True, pady=5)
        
        # Minimum track length filter
        filter_frame = ttk.Frame(track_frame)
        filter_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(filter_frame, text="Min Length:").pack(side="left")
        self.min_length_var = tk.IntVar(value=0)
        self.min_length_label = ttk.Label(filter_frame, text="0")
        self.min_length_label.pack(side="left", padx=(0, 5))
        
        def on_min_length_scale_changed(value):
            # Update label
            int_value = int(float(value))
            self.min_length_label.config(text=str(int_value))
            # Update checkboxes (live update)
            self.min_length_var.set(int_value)
            self.on_min_length_changed()
        
        min_length_scale = ttk.Scale(filter_frame, from_=0, to=100, orient="horizontal",
                                     variable=self.min_length_var, length=100,
                                     command=on_min_length_scale_changed)
        min_length_scale.pack(side="left", fill="x", expand=True, padx=5)
        
        # Track checkboxes container with scrollbar
        checkbox_container = ttk.Frame(track_frame)
        checkbox_container.pack(fill="both", expand=True, pady=5)
        
        # Canvas for scrollable checkboxes
        checkbox_canvas = tk.Canvas(checkbox_container, highlightthickness=0, height=200)
        checkbox_scrollbar = ttk.Scrollbar(checkbox_container, orient="vertical", command=checkbox_canvas.yview)
        self.checkbox_frame = ttk.Frame(checkbox_canvas)
        
        self.checkbox_frame.bind(
            "<Configure>",
            lambda e: checkbox_canvas.configure(scrollregion=checkbox_canvas.bbox("all"))
        )
        
        checkbox_canvas_window = checkbox_canvas.create_window((0, 0), window=self.checkbox_frame, anchor="nw")
        checkbox_canvas.configure(yscrollcommand=checkbox_scrollbar.set)
        
        def update_checkbox_canvas_width(event):
            canvas_width = event.width
            checkbox_canvas.itemconfig(checkbox_canvas_window, width=canvas_width)
        checkbox_canvas.bind('<Configure>', update_checkbox_canvas_width)
        
        # Bind mouse wheel to checkbox canvas
        def on_checkbox_mousewheel(event):
            checkbox_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        def on_enter_checkbox_canvas(event):
            checkbox_canvas.bind_all("<MouseWheel>", on_checkbox_mousewheel)
        
        def on_leave_checkbox_canvas(event):
            checkbox_canvas.unbind_all("<MouseWheel>")
        
        checkbox_canvas.bind("<Enter>", on_enter_checkbox_canvas)
        checkbox_canvas.bind("<Leave>", on_leave_checkbox_canvas)
        
        checkbox_canvas.pack(side="left", fill="both", expand=True)
        checkbox_scrollbar.pack(side="right", fill="y")
        
        # Control buttons
        button_frame = ttk.Frame(track_frame)
        button_frame.pack(fill="x", pady=(5, 0))
        ttk.Button(button_frame, text="Select All", command=self.select_all_tracks).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(button_frame, text="Clear Selection", command=self.clear_track_selection).pack(side="left", fill="x", expand=True, padx=2)
        
        # Time controls using PlaybackController
        self.playback_controller = PlaybackController(
            right_col,
            on_frame_changed=self.on_frame_changed,
            max_frame=0,
            initial_frame=0,
            initial_speed=1.0
        )
        
        # Right panel - 3D plot
        self.right_panel = ttk.Frame(main_frame)
        self.right_panel.pack(side="right", fill="both", expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.right_panel)
        toolbar.update()
        
        # Bind mouse wheel events for zooming
        self.setup_zoom()
        
        # Initial empty plot
        self.update_plot()
    
    def setup_export_section(self, parent_frame):
        """Setup export section for video export."""
        ttk.Button(parent_frame, text="Save Video", command=self.save_video).pack(pady=5, fill="x")
    
    def save_video(self):
        """Export the 3D visualization as a video file."""
        if not self.data:
            messagebox.showwarning("No Data", "Please load a CSV file first.")
            return
        
        # Ask for FPS
        fps = simpledialog.askinteger(
            "Video FPS",
            "Enter frames per second for the video:",
            initialvalue=30,
            minvalue=1,
            maxvalue=120
        )
        
        if fps is None:
            return
        
        # Generate output filename based on CSV filename and save in same folder as CSV
        if self.csv_path:
            csv_path_obj = Path(self.csv_path)
            csv_dir = csv_path_obj.parent
            csv_name = csv_path_obj.stem  # Full filename without extension
            # Output: *csvname*-3dplot.mp4, or *csvname*-3dplot-N.mp4 if multiple exist
            base_output = csv_dir / f"{csv_name}-3dplot.mp4"
            counter = 1
            output_file = base_output
            while output_file.exists():
                output_file = csv_dir / f"{csv_name}-3dplot-{counter}.mp4"
                counter += 1
        else:
            # Fallback: use 3dvis_output folder
            output_dir = Path("3dvis_output")
            output_dir.mkdir(exist_ok=True)
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
            writer = FFMpegWriter(fps=fps, metadata=dict(title='3D Trajectories', artist='3D-Cam'))
            
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
    
    def setup_zoom(self):
        """Setup mouse wheel zoom functionality and right-click drag panning."""
        canvas_widget = self.canvas.get_tk_widget()
        
        def on_scroll(event):
            zoom_factor = 1.1 if event.delta > 0 or event.num == 4 else 0.9
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            zlim = self.ax.get_zlim()
            
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            z_center = (zlim[0] + zlim[1]) / 2
            
            x_range = (xlim[1] - xlim[0]) / zoom_factor
            y_range = (ylim[1] - ylim[0]) / zoom_factor
            z_range = (zlim[1] - zlim[0]) / zoom_factor
            
            new_z_min = max(0, z_center - z_range/2)
            new_xlim = (x_center - x_range/2, x_center + x_range/2)
            new_ylim = (y_center - y_range/2, y_center + y_range/2)
            new_zlim = (new_z_min, z_center + z_range/2)
            
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.ax.set_zlim(new_zlim)
            self.persistent_bounds = (new_xlim, new_ylim, new_zlim)
            self.canvas.draw()
            return "break"
        
        # Right-click drag panning
        self._pan_start = None
        
        def on_right_click(event):
            """Start panning on right mouse button press."""
            self._pan_start = (event.x, event.y)
        
        def on_right_drag(event):
            """Pan the view while right mouse button is dragged."""
            if self._pan_start is None:
                return
            
            if not self.persistent_bounds:
                return  # Can't pan if bounds aren't set
            
            xlim, ylim, zlim = self.persistent_bounds
            
            # Calculate pan distance based on mouse movement
            # Convert pixel movement to data space movement
            dx_pixels = event.x - self._pan_start[0]
            dy_pixels = event.y - self._pan_start[1]
            
            # Get figure size in pixels
            fig_width, fig_height = self.fig.get_size_inches() * self.fig.dpi
            
            # Calculate pan distance as percentage of axis range
            pan_sensitivity = 0.5  # Lower value = slower panning, higher value = faster panning
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            z_range = zlim[1] - zlim[0]
            
            # Pan in data space (X increases left-to-right, Y increases bottom-to-top on screen)
            # Note: In matplotlib 3D, X and Y panning correspond to horizontal and vertical mouse movement
            dx_data = -dx_pixels / fig_width * x_range * pan_sensitivity
            dy_data = dy_pixels / fig_height * y_range * pan_sensitivity
            
            # Update bounds
            new_xlim = (xlim[0] + dx_data, xlim[1] + dx_data)
            new_ylim = (ylim[0] + dy_data, ylim[1] + dy_data)
            new_zlim = zlim  # Don't pan in Z direction
            
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.ax.set_zlim(new_zlim)
            self.persistent_bounds = (new_xlim, new_ylim, new_zlim)
            
            # Update pan start for next movement
            self._pan_start = (event.x, event.y)
            
            self.canvas.draw()
        
        def on_right_release(event):
            """Stop panning on right mouse button release."""
            self._pan_start = None
        
        canvas_widget.bind("<MouseWheel>", on_scroll)
        canvas_widget.bind("<Button-4>", on_scroll)
        canvas_widget.bind("<Button-5>", on_scroll)
        
        # Right-click panning
        canvas_widget.bind("<Button-3>", on_right_click)
        canvas_widget.bind("<B3-Motion>", on_right_drag)
        canvas_widget.bind("<ButtonRelease-3>", on_right_release)
        
        canvas_widget.focus_set()
        
        def on_enter(event):
            canvas_widget.focus_set()
        canvas_widget.bind("<Enter>", on_enter)
    
    def auto_load_latest_csv(self):
        """Automatically load the latest CSV file from inputs_outputs."""
        output_dir = Path("inputs_outputs")
        if not output_dir.exists():
            return
        
        # Find all CSV files in inputs_outputs subdirectories
        csv_files = list(output_dir.rglob("*.csv"))
        if not csv_files:
            return
        
        csv_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        latest_csv = csv_files[0]
        
        try:
            self.load_csv_file(str(latest_csv))
        except Exception as e:
            print(f"Failed to auto-load {latest_csv}: {e}")
    
    def load_csv(self):
        """Open file dialog to load CSV file."""
        file_path = filedialog.askopenfilename(
            title="Load CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="inputs_outputs"
        )
        if file_path:
            self.load_csv_file(file_path)
    
    def load_csv_file(self, file_path: str):
        """Load and parse CSV file - override in subclasses for custom loading."""
        self.csv_path = file_path
        self.data = {}
        self.frame_data = {}
        self.max_frame = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                if not all(col in reader.fieldnames for col in ['Frame_Number', 'Track_ID', 'X_mm', 'Y_mm', 'Z_mm']):
                    messagebox.showerror("Error", "CSV file must contain Frame_Number, Track_ID, X_mm, Y_mm, Z_mm columns")
                    return
                
                for row in reader:
                    try:
                        frame = int(row['Frame_Number'])
                        track_id = int(row['Track_ID'])
                        
                        x_str = row.get('X_mm', '').strip()
                        y_str = row.get('Y_mm', '').strip()
                        z_str = row.get('Z_mm', '').strip()
                        
                        if not x_str or not y_str or not z_str:
                            continue
                        
                        x = float(x_str)
                        y = float(y_str)
                        z = float(z_str)
                        
                        # Filter out (0,0,0) points which are likely invalid/missing data
                        # unless they're actually at the optical center (which is valid)
                        # We'll be more lenient - only filter if it's clearly invalid
                        # (e.g., if multiple consecutive frames are 0,0,0, or if it breaks continuity)
                        
                        if track_id not in self.data:
                            self.data[track_id] = []
                        self.data[track_id].append((frame, x, y, z))
                        
                        if frame not in self.frame_data:
                            self.frame_data[frame] = []
                        self.frame_data[frame].append((track_id, x, y, z))
                        
                        self.max_frame = max(self.max_frame, frame)
                    
                    except (ValueError, KeyError):
                        continue
            
            # Sort data by frame for each track
            for track_id in self.data:
                self.data[track_id].sort(key=lambda x: x[0])
            
            self.track_ids = sorted(self.data.keys())
            
            # Update UI
            self.file_label.config(text=f"Loaded: {Path(file_path).name}")
            self.playback_controller.set_max_frame(self.max_frame)
            self.playback_controller.set_frame(0)
            self.current_frame = 0
            
            # Update track checkboxes
            self.update_track_checkboxes()
            
            # Update info
            total_points = sum(len(self.data[tid]) for tid in self.data)
            self.info_label.config(
                text=f"Tracks: {len(self.track_ids)}\n"
                     f"Frames: {self.max_frame + 1}\n"
                     f"Total Points: {total_points}"
            )
            
            # Reset view flags
            self.bounds_set = False
            self.persistent_bounds = None
            self._view_set = False
            
            # Update plot
            self.update_plot()
            
            # Call custom on_load callback
            self.on_data_loaded()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file:\n{e}")
    
    def on_data_loaded(self):
        """Called after data is loaded - override in subclasses."""
        pass
    
    def on_frame_changed(self, frame: int):
        """Handle frame change from playback controller."""
        if frame != self.current_frame:
            self.current_frame = frame
            self.update_plot()
    
    def set_frame(self, frame: int):
        """Set frame programmatically."""
        self.playback_controller.set_frame(frame)
    
    def on_display_changed(self):
        """Handle display option changes."""
        self.show_trails = self.trails_var.get()
        self.update_plot()
    
    def on_trail_length_changed(self, value=None):
        """Handle trail length change."""
        self.trail_length = int(self.trail_length_var.get())
        self.update_plot()
    
    def on_show_all_changed(self):
        """Handle show all tracks checkbox."""
        if self.show_all_var.get():
            self.select_all_tracks()
        else:
            # Don't auto-clear, just uncheck the "show all" checkbox
            pass
    
    def select_all_tracks(self):
        """Select all available tracks."""
        available_tracks = list(self.track_checkboxes.keys())
        self.selected_tracks = set(available_tracks)
        for track_id in available_tracks:
            if track_id in self.track_checkboxes:
                self.track_checkboxes[track_id][0].set(True)
        self.show_all_var.set(True)
        self.update_plot()
    
    def clear_track_selection(self):
        """Clear track selection."""
        for var, _ in self.track_checkboxes.values():
            var.set(False)
        self.selected_tracks = set()
        self.show_all_var.set(False)
        self.update_plot()
    
    def update_track_checkboxes(self):
        """Update track checkboxes based on available tracks and filter."""
        # Clear existing checkboxes
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()
        self.track_checkboxes.clear()
        
        # Filter tracks by minimum length
        min_length = self.min_track_length
        available_tracks = [tid for tid in self.track_ids if len(self.data.get(tid, [])) >= min_length]
        
        # Create checkboxes for available tracks
        for track_id in sorted(available_tracks):
            point_count = len(self.data[track_id])
            var = tk.BooleanVar(value=track_id in self.selected_tracks)
            checkbox = ttk.Checkbutton(
                self.checkbox_frame,
                text=f"Track {track_id} ({point_count} pts)",
                variable=var,
                command=self.on_track_checkbox_changed
            )
            checkbox.pack(anchor="w", padx=5, pady=1)
            self.track_checkboxes[track_id] = (var, checkbox)
        
        # Update selected tracks to only include available tracks
        self.selected_tracks = {tid for tid in self.selected_tracks if tid in available_tracks}
        if not self.selected_tracks and available_tracks:
            # If no tracks selected but tracks are available, select all
            self.selected_tracks = set(available_tracks)
            for track_id in available_tracks:
                if track_id in self.track_checkboxes:
                    self.track_checkboxes[track_id][0].set(True)
        
        self.show_all_var.set(len(self.selected_tracks) == len(available_tracks) and len(available_tracks) > 0)
        self.update_plot()
    
    def on_track_checkbox_changed(self):
        """Handle checkbox change - update selected tracks."""
        self.selected_tracks = {tid for tid, (var, _) in self.track_checkboxes.items() if var.get()}
        available_tracks = list(self.track_checkboxes.keys())
        self.show_all_var.set(len(self.selected_tracks) == len(available_tracks) and len(available_tracks) > 0)
        self.update_plot()
    
    def on_min_length_changed(self, value=None):
        """Handle minimum track length filter change."""
        self.min_track_length = self.min_length_var.get()
        self.update_track_checkboxes()
    
    def on_track_selection_changed(self, event=None):
        """Legacy method - now handled by checkbox changes."""
        pass
    
    def reset_view(self):
        """Reset view to optimal bounds and angle."""
        self.bounds_set = False
        self.persistent_bounds = None
        self.set_optimal_view()
        self.update_plot()
    
    def update_plot(self):
        """Update the 3D plot - override in subclasses for custom plotting."""
        self.ax.clear()
        
        if not self.data:
            self.ax.text(0.5, 0.5, 0.5, "No data loaded", transform=self.ax.transAxes, ha="center")
            self.canvas.draw()
            return
        
        tracks_to_show = self.selected_tracks if self.selected_tracks else self.track_ids
        
        if not tracks_to_show:
            self.ax.text(0.5, 0.5, 0.5, "No tracks selected", transform=self.ax.transAxes, ha="center")
            self.canvas.draw()
            return
        
        # Color map
        colors = cm.tab20(np.linspace(0, 1, len(self.track_ids)))
        track_color_map = {tid: colors[i % len(colors)] for i, tid in enumerate(self.track_ids)}
        
        # Plot trajectories
        for track_id in tracks_to_show:
            if track_id not in self.data:
                continue
            
            points = self.data[track_id]
            if not points:
                continue
            
            frames = [p[0] for p in points]
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            zs = [p[3] for p in points]
            
            color = track_color_map[track_id]
            current_frame = self.current_frame
            
            if self.show_trails:
                # Show trail: points up to and including current frame
                trail_start = max(0, current_frame - self.trail_length)
                trail_mask = [f >= trail_start and f <= current_frame for f in frames]
                
                if any(trail_mask):
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
        
        # Labels
        self.ax.set_xlabel('X (mm)', fontsize=10)
        self.ax.set_ylabel('Y (mm)', fontsize=10)
        self.ax.set_zlabel('Z (mm)', fontsize=10)
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
        
        # Set optimal viewing angle
        if not hasattr(self, '_view_set'):
            self.set_optimal_view()
            self._view_set = True
        
        # Grid
        self.ax.grid(True, alpha=0.3)
        
        if len(tracks_to_show) <= 10:
            handles, labels = self.ax.get_legend_handles_labels()
            if handles:  # Only show legend if there are labeled artists
                self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        self.canvas.draw()
    
    def set_optimal_bounds(self, tracks_to_show):
        """Set axis bounds to maximum extents from visible data."""
        all_xs = [p[1] for tid in tracks_to_show for p in self.data.get(tid, [])]
        all_ys = [p[2] for tid in tracks_to_show for p in self.data.get(tid, [])]
        all_zs = [p[3] for tid in tracks_to_show for p in self.data.get(tid, [])]
        
        if not all_xs or not all_ys or not all_zs:
            return
        
        x_min, x_max = min(all_xs), max(all_xs)
        y_min, y_max = min(all_ys), max(all_ys)
        z_min, z_max = min(all_zs), max(all_zs)
        
        if z_min < 0:
            z_max = z_max - z_min
            z_min = 0
        else:
            z_min = 0
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_zlim(z_min, z_max)
        
        self.persistent_bounds = ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    
    def set_optimal_view(self):
        """Set optimal 3D viewing angle."""
        self.ax.view_init(elev=30, azim=-135)
    
    def get_plot_title(self):
        """Get plot title - override in subclasses for custom titles."""
        return f'3D Trajectories - Frame {self.current_frame}/{self.max_frame}'
    
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

