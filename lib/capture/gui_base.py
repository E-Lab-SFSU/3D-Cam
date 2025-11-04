#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Capture GUI Application
-----------------------------
Shared base class for platform-specific camera capture GUIs.
Consolidates common functionality from capture_raspi.py and capture_windows.py.
"""

import cv2
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox
from queue import Queue, Empty
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any

from lib.capture import (
    Camera,
    list_all_cameras, get_camera_info, CameraInfo,
    PreviewManager, FrameGrabber, RecordingManager,
    make_capture_output_path, make_capture_frame_path,
    set_camera_control, get_camera_control_range,  # For V4L2
)


class BaseCaptureApp(ABC):
    """
    Base class for camera capture GUI applications.
    
    Platform-specific implementations should override:
    - setup_environment() - Set OpenCV backend environment variables
    - get_camera_backend() - Return platform-specific backend constant
    - get_control_specs() - Return control ranges and defaults
    - _build_camera_controls() - Build platform-specific control UI
    - _load_camera_control_ranges() - Load control ranges from camera
    - _apply_camera_controls() - Apply control values to camera
    - _reset_control_defaults() - Get default control values
    - get_default_fps() - Get default FPS value
    - get_resolution_for_format() - Get resolution for format
    - get_window_title() - Get window title
    """
    
    def __init__(self, root: tk.Tk):
        """Initialize the capture application."""
        self.root = root
        
        # Setup platform-specific environment
        self.setup_environment()
        
        # Set window title and geometry
        self.root.title(self.get_window_title())
        width, height = self.get_window_size()
        self.root.geometry(f"{width}x{height}")
        
        # Camera state
        self.cam: Optional[Camera] = None
        self.camera_info: Optional[CameraInfo] = None
        self.available_cameras: list[CameraInfo] = []
        
        # Format and resolution
        self.format_var = tk.StringVar(value="YUYV")
        self.scale_percent = tk.DoubleVar(value=100.0)
        self.fps_var = tk.DoubleVar(value=self.get_default_fps())
        
        # Frame queue (shared by preview and recording)
        self.frame_queue = Queue(maxsize=10)
        
        # Initialize managers (order matters - preview needs recording ref)
        self.frame_grabber = FrameGrabber(self.frame_queue)
        self.recording_manager = RecordingManager(
            frame_queue=self.frame_queue,
            get_camera_fn=lambda: self.cam,
            scaled_size_fn=lambda: self.scaled_size(),
            status_callback=lambda text, color: self.status_label.config(text=text, foreground=color) if hasattr(self, 'status_label') else None,
            button_callback=lambda text: self.record_btn.config(text=text) if hasattr(self, 'record_btn') else None
        )
        self.preview_manager = PreviewManager(
            root=self.root,
            frame_queue=self.frame_queue,
            camera_var=lambda: (self.cam, self.cam.is_open() if self.cam else False),
            format_var=lambda: self.format_var.get(),
            recording_var=lambda: self.recording_manager.recording
        )
        
        # Camera controls
        self.control_vars: Dict[str, tk.DoubleVar] = {}
        self.control_ranges: Dict[str, Dict[str, Any]] = {}
        
        # Build UI
        self._build_ui()
        
        # Detect cameras on startup
        self.refresh_camera_list()
        
        # Start preview window immediately (always on)
        self.root.after(100, self.preview_manager.start)
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def setup_environment(self) -> None:
        """Setup platform-specific environment variables."""
        pass
    
    @abstractmethod
    def get_camera_backend(self) -> int:
        """Get the camera backend constant for this platform."""
        pass
    
    @abstractmethod
    def get_control_specs(self) -> Dict[str, Tuple[str, int, int, int]]:
        """
        Get control specifications: {name: (label, min, max, default)}.
        
        Returns:
            Dictionary mapping control names to (label, min, max, default) tuples.
        """
        pass
    
    @abstractmethod
    def _build_camera_controls(self) -> None:
        """Build platform-specific camera control UI."""
        pass
    
    @abstractmethod
    def _load_camera_control_ranges(self) -> None:
        """Load control ranges from camera (platform-specific)."""
        pass
    
    @abstractmethod
    def _apply_camera_controls(self) -> None:
        """Apply current control values to camera (platform-specific)."""
        pass
    
    @abstractmethod
    def _reset_control_defaults(self) -> Dict[str, int]:
        """Get default control values for reset (platform-specific)."""
        pass
    
    @abstractmethod
    def get_default_fps(self) -> float:
        """Get default FPS value for this platform."""
        pass
    
    @abstractmethod
    def get_resolution_for_format(self, format_str: str) -> Tuple[int, int]:
        """Get resolution (width, height) for a given format."""
        pass
    
    @abstractmethod
    def get_window_title(self) -> str:
        """Get window title for this platform."""
        pass
    
    def get_window_size(self) -> Tuple[int, int]:
        """Get window size (width, height). Default is 1000x700."""
        return (1000, 700)
    
    def _build_ui(self) -> None:
        """Build the Tkinter GUI (common across platforms)."""
        main = ttk.Frame(self.root, padding="5")
        main.pack(fill=tk.BOTH, expand=True)

        # Right side: Controls
        ctrl = ttk.Frame(main)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=3, pady=3)
        
        # Camera selection
        cam_frame = ttk.LabelFrame(ctrl, text="Camera", padding="3")
        cam_frame.pack(fill=tk.X, pady=(0, 4))
        
        cam_row = ttk.Frame(cam_frame)
        cam_row.pack(fill=tk.X)
        ttk.Label(cam_row, text="Cam:").pack(side=tk.LEFT, padx=(0, 2))
        self.camera_combo = ttk.Combobox(cam_row, width=18, state="readonly")
        self.camera_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)
        ttk.Button(cam_row, text="↻", width=3, command=self.refresh_camera_list).pack(side=tk.LEFT)
        
        # Camera info display (compact, smaller font)
        self.camera_info_label = ttk.Label(
            cam_frame, 
            text="No camera", 
            justify="left",
            font=("TkDefaultFont", 8),
            wraplength=200
        )
        self.camera_info_label.pack(fill=tk.X, pady=(2, 0))
        
        # Format selection
        format_frame = ttk.LabelFrame(ctrl, text="Format", padding="3")
        format_frame.pack(fill=tk.X, pady=(0, 4))
        
        # Format and FPS in one row
        fmt_row = ttk.Frame(format_frame)
        fmt_row.pack(fill=tk.X)
        ttk.Label(fmt_row, text="Format:").pack(side=tk.LEFT, padx=(0, 2))
        self.format_combo = ttk.Combobox(fmt_row, textvariable=self.format_var, width=6, state="readonly")
        self.format_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.format_combo.bind("<<ComboboxSelected>>", self.on_format_changed)
        self._add_tooltip(self.format_combo, "YUYV = raw/VGA | MJPG = compressed/HD")
        
        ttk.Label(fmt_row, text="FPS:").pack(side=tk.LEFT, padx=(0, 2))
        fps_entry = ttk.Entry(fmt_row, textvariable=self.fps_var, width=5, font=("TkDefaultFont", 9))
        fps_entry.pack(side=tk.LEFT)
        self._add_tooltip(fps_entry, "0=max speed, 1-60=specific FPS")
        
        # Scale in format frame
        scale_row = ttk.Frame(format_frame)
        scale_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(scale_row, text="Scale:").pack(side=tk.LEFT, padx=(0, 2))
        scale_entry = ttk.Entry(scale_row, textvariable=self.scale_percent, width=5, font=("TkDefaultFont", 9))
        scale_entry.pack(side=tk.LEFT, padx=(0, 2))
        self._add_tooltip(scale_entry, "Output scale %")
        self.scale_label = ttk.Label(scale_row, text="", font=("TkDefaultFont", 8))
        self.scale_label.pack(side=tk.LEFT)
        
        # Status section
        status_frame = ttk.LabelFrame(ctrl, text="Status", padding="3")
        status_frame.pack(fill=tk.X, pady=(0, 4))
        
        self.status_label = ttk.Label(status_frame, text="Ready", font=("TkDefaultFont", 9))
        self.status_label.pack(anchor="w", pady=(0, 2))
        
        self.fps_label = ttk.Label(status_frame, text="FPS: —", font=("TkDefaultFont", 8))
        self.fps_label.pack(anchor="w")
        
        # Open/Close Camera button
        self.open_camera_btn = ttk.Button(status_frame, text="Open Camera", command=self.toggle_camera)
        self.open_camera_btn.pack(fill=tk.X, pady=(4, 0))
        
        # Camera controls (stacked sliders) - platform-specific
        self.param_frame = ttk.LabelFrame(ctrl, text="Controls", padding="3")
        self.param_frame.pack(fill=tk.X, pady=(0, 4))
        self._build_camera_controls()
        
        # Actions section
        btn_frame = ttk.LabelFrame(ctrl, text="Actions", padding="3")
        btn_frame.pack(fill=tk.X, pady=(0, 4))
        
        # Record/Stop Record button
        self.record_btn = ttk.Button(btn_frame, text="Record", command=self.toggle_record)
        self.record_btn.pack(fill=tk.X, pady=(0, 2))
        
        # Capture Frame button
        ttk.Button(btn_frame, text="Capture Frame", command=self.capture_frame).pack(fill=tk.X, pady=(0, 2))
        
        # Reset Controls button
        ttk.Button(btn_frame, text="Reset Controls", command=self.reset_controls).pack(fill=tk.X, pady=(0, 2))
        
        # Exit button
        ttk.Button(btn_frame, text="Exit", command=self.on_close).pack(fill=tk.X)
    
    def _add_tooltip(self, widget: tk.Widget, text: str) -> None:
        """Add a tooltip to a widget (common implementation)."""
        from lib.gui import tooltip
        tooltip(widget, text)
    
    def refresh_camera_list(self) -> None:
        """Refresh the list of available cameras."""
        print("[INFO] Refreshing camera list...")
        try:
            self.available_cameras = list_all_cameras(max_test=10, detailed=False, test_resolutions=False)
        except Exception as e:
            print(f"[ERROR] Failed to refresh camera list: {e}")
            self.available_cameras = []
        
        # Update combo box
        values = []
        for cam_info in self.available_cameras:
            name_str = f"Camera {cam_info.index}"
            if cam_info.name:
                name_str += f" ({cam_info.name})"
            values.append(name_str)
        
        self.camera_combo['values'] = values
        if values:
            self.camera_combo.current(0)
            self.on_camera_selected()
        else:
            self.camera_info_label.config(text="No cameras found")
    
    def on_camera_selected(self, event=None) -> None:
        """Handle camera selection change."""
        selection = self.camera_combo.get()
        if not selection:
            return
        
        # Find the selected camera info
        for cam_info in self.available_cameras:
            name_str = f"Camera {cam_info.index}"
            if cam_info.name:
                name_str += f" ({cam_info.name})"
            if name_str == selection:
                self.camera_info = cam_info
                self._update_camera_info_display()
                # Update format combo based on supported formats
                if cam_info.supported_fourcc and hasattr(self, 'format_combo'):
                    self.format_combo['values'] = cam_info.supported_fourcc
                    if cam_info.supported_fourcc:
                        default_format = cam_info.default_fourcc if cam_info.default_fourcc in cam_info.supported_fourcc else cam_info.supported_fourcc[0]
                        self.format_var.set(default_format)
                        try:
                            idx = cam_info.supported_fourcc.index(default_format)
                            self.format_combo.current(idx)
                        except ValueError:
                            self.format_combo.current(0)
                break
    
    def on_format_changed(self, event=None) -> None:
        """Handle format selection change - reopen camera if already open."""
        if self.cam and self.cam.is_open():
            print("[INFO] Format changed, reopening camera with new settings...")
            self.close_camera()
            self.root.after(100, self.open_camera)
    
    def _update_camera_info_display(self) -> None:
        """Update the camera info label."""
        if not self.camera_info:
            self.camera_info_label.config(text="No camera selected")
            return
        
        info_text = f"Index: {self.camera_info.index}\n"
        if self.camera_info.name:
            info_text += f"Name: {self.camera_info.name}\n"
        info_text += f"Default: {self.camera_info.default_fourcc} "
        info_text += f"{self.camera_info.default_width}×{self.camera_info.default_height} "
        info_text += f"@ {self.camera_info.default_fps:.1f} FPS\n"
        if self.camera_info.supported_fourcc:
            info_text += f"Formats: {', '.join(self.camera_info.supported_fourcc)}"
        
        self.camera_info_label.config(text=info_text)
    
    def scaled_size(self) -> Tuple[int, int]:
        """Get scaled output size."""
        if not self.cam:
            return 640, 480
        p = max(1, min(100, float(self.scale_percent.get())))
        return int(self.cam.w * p / 100), int(self.cam.h * p / 100)
    
    def toggle_camera(self) -> None:
        """Toggle camera open/close."""
        if self.cam and self.cam.is_open():
            self.close_camera()
        else:
            self.open_camera()
    
    def open_camera(self) -> None:
        """Open the selected camera (common logic)."""
        if self.cam:
            self.cam.release()
            self.cam = None
        
        if not self.camera_info:
            messagebox.showerror("Camera Error", "Please select a camera first.")
            return
        
        format_str = self.format_var.get()
        if format_str not in self.camera_info.supported_fourcc:
            messagebox.showwarning(
                "Format Warning", 
                f"Format {format_str} may not be supported. Using default format."
            )
            format_str = self.camera_info.default_fourcc
        
        # Get resolution for format (platform-specific)
        width, height = self.get_resolution_for_format(format_str)
        
        # Get framerate from input (0 = maximum speed, >0 = specific FPS)
        try:
            fps_value = float(self.fps_var.get())
            fps_value = max(0.0, min(60.0, fps_value))
            self.fps_var.set(fps_value)
        except (ValueError, tk.TclError):
            fps_value = self.get_default_fps()
            self.fps_var.set(fps_value)
        
        # Create camera with platform-specific backend
        self.cam = Camera(
            index=self.camera_info.index,
            fps=int(fps_value) if fps_value > 0 else 0,
            fourcc=format_str,
            width=width,
            height=height,
            backend=self.get_camera_backend()
        )
        
        if not self.cam.open():
            messagebox.showerror("Camera Error", f"Failed to open camera {self.camera_info.index}")
            return
        
        # Stop frame grabber if running
        self.frame_grabber.stop()
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        # Load camera control ranges (platform-specific)
        self._load_camera_control_ranges()
        # Apply current control values (platform-specific)
        self._apply_camera_controls()
        
        # Platform-specific post-open setup
        self._post_open_camera_setup()
        
        # Get actual camera FPS
        actual_fps = self.cam.cap.get(cv2.CAP_PROP_FPS) if self.cam.cap else 0
        
        # Update FPS display
        if actual_fps == 0:
            self.fps_label.config(text="FPS: Max speed")
            print("[INFO] Camera set to capture at maximum speed (FPS not limited)")
            output_fps = 30.0
        else:
            self.fps_label.config(text=f"FPS: {actual_fps:.1f}")
            print(f"[INFO] Camera FPS: {actual_fps:.1f}")
            output_fps = actual_fps
        
        # Small delay to let camera stabilize before starting frame grabber
        # This helps prevent initial read failures on V4L2
        time.sleep(0.2)
        
        # Start frame grabber (after camera is fully initialized)
        self.frame_grabber.start(self.cam)
        
        self.update_scale_info()
        
        # Update UI
        self.open_camera_btn.config(text="Close Camera")
        self.status_label.config(text=f"Camera {self.camera_info.index} ready", foreground="green")
        if actual_fps == 0:
            self.fps_label.config(text="FPS: Max speed")
        else:
            self.fps_label.config(text=f"FPS: {actual_fps:.1f}")
        
        print("[INFO] Camera opened successfully")
        print(f"[INFO] Output video will be recorded at {output_fps:.1f} FPS")
        print(f"[INFO] Format: {format_str}, Resolution: {self.cam.w}x{self.cam.h}")
        if format_str == "MJPG":
            print("[INFO] Note: MJPG at 1920x1080 may require more processing time")
    
    def _post_open_camera_setup(self) -> None:
        """Platform-specific setup after opening camera. Override if needed."""
        pass
    
    def close_camera(self) -> None:
        """Close the camera."""
        # Stop recording first
        self.recording_manager.stop()
        
        # Stop frame grabber and wait for it to finish
        self.frame_grabber.stop()
        
        # Small delay to ensure frame grabber thread has exited
        time.sleep(0.1)
        
        # Release camera
        if self.cam:
            try:
                self.cam.release()
            except:
                pass
            self.cam = None
        
        self.open_camera_btn.config(text="Open Camera")
        self.status_label.config(text="Camera closed", foreground="black")
        print("[INFO] Camera closed (preview window remains open)")
    
    def update_scale_info(self) -> None:
        """Update scale information display."""
        if not self.cam:
            return
        
        p = max(1, min(100, float(self.scale_percent.get())))
        sw, sh = int(self.cam.w * p / 100), int(self.cam.h * p / 100)
        self.scale_label.config(text=f"{sw}×{sh}")
        print(f"[INFO] {self.format_var.get()} {self.cam.w}×{self.cam.h} → {sw}×{sh} ({p:.1f}%)")
    
    def reset_controls(self) -> None:
        """Reset camera controls to defaults."""
        if not self.cam or not self.camera_info:
            return
        
        defaults = self._reset_control_defaults()
        
        for name, default_val in defaults.items():
            if name in self.control_vars:
                if name in self.control_ranges:
                    default_val = self.control_ranges[name].get('default', default_val)
                self.control_vars[name].set(default_val)
        
        # Apply defaults to camera (platform-specific)
        self._apply_camera_controls()
        
        print("[INFO] Controls reset to defaults")
    
    def capture_frame(self) -> None:
        """Capture a single frame."""
        if not self.cam or not self.cam.is_open():
            messagebox.showinfo("Camera", "Open camera first.")
            return
        
        # Try to get frame from queue
        try:
            frame = self.frame_queue.get(timeout=1.0)
        except Empty:
            messagebox.showerror("Capture Error", "Failed to read frame from camera (timeout).")
            return
        
        if frame is None:
            messagebox.showerror("Capture Error", "Failed to read frame from camera.")
            return
        
        # Scale frame
        w, h = self.scaled_size()
        if w != self.cam.w or h != self.cam.h:
            frame_resized = cv2.resize(frame, (w, h))
        else:
            frame_resized = frame
        
        # Convert if needed
        if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
            frame_bgr = frame_resized
        else:
            frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_YUV2BGR_YUY2)
        
        # Save to capture_output directory
        output_path = make_capture_frame_path(w, h)
        cv2.imwrite(output_path, frame_bgr)
        print(f"[INFO] Saved {output_path}")
        self.status_label.config(text=f"Saved: {os.path.basename(output_path)}", foreground="blue")
    
    def toggle_record(self) -> None:
        """Toggle video recording."""
        if not self.cam or not self.cam.is_open():
            messagebox.showinfo("Camera", "Open camera first.")
            return
        
        if self.recording_manager.recording:
            self.recording_manager.stop()
        else:
            self.recording_manager.start()
    
    def on_close(self) -> None:
        """Handle application close."""
        print("[INFO] Closing application")
        self.preview_manager.stop()
        self.recording_manager.stop()
        self.frame_grabber.stop()
        
        if self.cam:
            self.cam.release()
        
        cv2.destroyAllWindows()
        self.root.destroy()
