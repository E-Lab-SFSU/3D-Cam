#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Capture App - Modular base class for platform-specific capture applications
Contains all common functionality for camera capture GUI

This base class is cross-platform and works on:
- Windows (via capture_windows.py)
- Linux / Raspberry Pi (via capture_raspi.py)

Platform-specific implementations should:
1. Set environment variables BEFORE importing this module (if needed)
2. Override _build_camera_controls() for platform-specific controls
3. Override open_camera() for platform-specific camera opening
4. Override _load_camera_control_ranges() for platform-specific control loading
5. Override _apply_camera_controls() for platform-specific control application
6. Override reset_controls() for platform-specific control reset
"""

import cv2
import os
from queue import Queue, Empty
import tkinter as tk
from tkinter import ttk, messagebox

from lib.capture import PreviewManager, FrameGrabber, RecordingManager, make_capture_frame_path


# ============ Utilities ============
def tooltip(widget, text):
    """Create a tooltip for a widget."""
    tip = tk.Toplevel(widget)
    tip.withdraw()
    tip.overrideredirect(True)
    lbl = tk.Label(tip, text=text, bg="#ffffe0", relief="solid", borderwidth=1, font=("TkDefaultFont", 9))
    lbl.pack()
    def show(_): 
        tip.geometry(f"+{widget.winfo_rootx()+30}+{widget.winfo_rooty()+10}")
        tip.deiconify()
    def hide(_): 
        tip.withdraw()
    widget.bind("<Enter>", show)
    widget.bind("<Leave>", hide)


class BaseCaptureApp:
    """
    Base class for camera capture applications - platform-specific implementations inherit from this.
    
    This class is fully cross-platform and works on Windows, Linux, and Raspberry Pi.
    All platform-specific logic (camera backends, control mechanisms) is implemented
    in subclasses via method overrides.
    
    Platform-specific methods that subclasses must override:
    - _build_camera_controls() - Build UI for camera controls
    - open_camera() - Open camera with platform-specific backend
    - _load_camera_control_ranges() - Load control ranges from camera
    - _apply_camera_controls() - Apply control values to camera
    - reset_controls() - Reset controls to defaults
    """
    
    def __init__(self, root, title="Camera Capture", default_format="YUYV"):
        self.root = root
        self.root.title(title)
        
        # Camera state (to be set by platform-specific implementation)
        self.cam = None
        self.camera_info = None  # Current CameraInfo object
        self.available_cameras = []  # List of CameraInfo objects
        
        # Format and resolution
        self.format_var = tk.StringVar(value=default_format)
        self.scale_percent = tk.DoubleVar(value=100.0)
        # Default FPS - platform-specific implementations can override
        # Windows default is 0 (max speed) set in capture_windows.py __init__
        self.fps_var = tk.DoubleVar(value=30.0)
        
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
        self.control_vars = {}
        self.control_ranges = {}  # Store ranges for each control
        
        # Build UI
        self._build_ui()
        
        # Detect cameras on startup
        self.refresh_camera_list()
        
        # Start preview window immediately (always on)
        self.root.after(100, self.preview_manager.start)
    
    def _build_ui(self):
        """Build the common Tkinter GUI elements."""
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
        tooltip(self.format_combo, "YUYV = raw/VGA | MJPG = compressed/HD")
        
        ttk.Label(fmt_row, text="FPS:").pack(side=tk.LEFT, padx=(0, 2))
        fps_entry = ttk.Entry(fmt_row, textvariable=self.fps_var, width=5, font=("TkDefaultFont", 9))
        fps_entry.pack(side=tk.LEFT)
        tooltip(fps_entry, "0=max speed, 1-60=specific FPS")
        
        # Scale in format frame
        scale_row = ttk.Frame(format_frame)
        scale_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(scale_row, text="Scale:").pack(side=tk.LEFT, padx=(0, 2))
        scale_entry = ttk.Entry(scale_row, textvariable=self.scale_percent, width=5, font=("TkDefaultFont", 9))
        scale_entry.pack(side=tk.LEFT, padx=(0, 2))
        tooltip(scale_entry, "Output scale %")
        self.scale_label = ttk.Label(scale_row, text="—", font=("TkDefaultFont", 8))
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
        
        # Platform-specific controls (override in subclass)
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
    
    def _build_camera_controls(self):
        """Build camera control sliders - override in platform-specific implementation."""
        pass
    
    def refresh_camera_list(self):
        """Refresh the list of available cameras - uses platform-specific list_all_cameras."""
        from lib.camera_info import list_all_cameras
        
        print("[INFO] Refreshing camera list...")
        try:
            self.available_cameras = list_all_cameras(max_test=10, detailed=False, test_resolutions=False)
        except Exception as e:
            print(f"[ERROR] Failed to refresh camera list: {e}")
            import traceback
            traceback.print_exc()
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
    
    def on_camera_selected(self, event=None):
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
                        # Set to default format or first available
                        default_format = cam_info.default_fourcc if cam_info.default_fourcc in cam_info.supported_fourcc else cam_info.supported_fourcc[0]
                        self.format_var.set(default_format)
                        try:
                            idx = cam_info.supported_fourcc.index(default_format)
                            self.format_combo.current(idx)
                        except ValueError:
                            self.format_combo.current(0)
                break
    
    def on_format_changed(self, event=None):
        """Handle format selection change - reopen camera if already open."""
        if self.cam and self.cam.is_open():
            # Camera is open, close and reopen with new format
            print("[INFO] Format changed, reopening camera with new settings...")
            self.close_camera()
            # Small delay then reopen
            self.root.after(100, self.open_camera)
    
    def _update_camera_info_display(self):
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
    
    def scaled_size(self):
        """Get scaled output size."""
        if not self.cam:
            return 640, 480
        p = max(1, min(100, float(self.scale_percent.get())))
        return int(self.cam.w * p / 100), int(self.cam.h * p / 100)
    
    def toggle_camera(self):
        """Toggle camera open/close."""
        if self.cam and self.cam.is_open():
            self.close_camera()
        else:
            self.open_camera()
    
    def open_camera(self):
        """Platform-specific camera opening - must be implemented by subclass.
        Subclass should call _finalize_camera_open() after creating and opening the camera."""
        raise NotImplementedError("Subclass must implement open_camera()")
    
    def _finalize_camera_open(self, format_str):
        """Helper method to finalize camera opening - common logic after camera is created."""
        if not self.cam or not self.cam.is_open():
            return False
        
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
        
        # Get actual camera FPS (may be 0 if not set, meaning max speed)
        actual_fps = self.cam.cap.get(cv2.CAP_PROP_FPS) if self.cam.cap else 0
        
        # Update FPS display
        if actual_fps == 0:
            self.fps_label.config(text="FPS: Max speed")
            print("[INFO] Camera set to capture at maximum speed (FPS not limited)")
            # Use 30 FPS for output if capturing at max speed
            output_fps = 30.0
        else:
            self.fps_label.config(text=f"FPS: {actual_fps:.1f}")
            print(f"[INFO] Camera FPS: {actual_fps:.1f}")
            output_fps = actual_fps
        
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
        
        return True
    
    def close_camera(self):
        """Close the camera - common implementation."""
        # Stop recording
        self.recording_manager.stop()
        
        # Stop frame grabber
        self.frame_grabber.stop()
        
        if self.cam:
            self.cam.release()
            self.cam = None
        
        self.open_camera_btn.config(text="Open Camera")
        self.status_label.config(text="Camera closed", foreground="black")
        print("[INFO] Camera closed (preview window remains open)")
    
    def _load_camera_control_ranges(self):
        """Load control ranges from camera - platform-specific implementation."""
        pass
    
    def _apply_camera_controls(self):
        """Apply current control values to camera - platform-specific implementation."""
        pass
    
    def update_scale_info(self):
        """Update scale information display."""
        if not self.cam:
            return
        
        p = max(1, min(100, float(self.scale_percent.get())))
        sw, sh = int(self.cam.w * p / 100), int(self.cam.h * p / 100)
        self.scale_label.config(text=f"{sw}×{sh}")
        print(f"[INFO] {self.format_var.get()} {self.cam.w}×{self.cam.h} → {sw}×{sh} ({p:.1f}%)")
    
    def reset_controls(self):
        """Reset camera controls to defaults - platform-specific implementation."""
        pass
    
    def _convert_frame_to_bgr(self, frame):
        """
        Convert frame to BGR format if needed.
        OpenCV VideoCapture typically auto-converts YUYV to BGR, but this handles edge cases.
        
        Args:
            frame: Input frame (may be BGR, YUYV, or other format)
            
        Returns:
            BGR frame ready for display/recording
        """
        if frame is None:
            return None
        
        try:
            # If already BGR (3 channels, 3 bytes per pixel), use as-is
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Verify it's actually BGR by checking value ranges
                # BGR should have values 0-255
                if frame.dtype == 'uint8' and frame.max() <= 255 and frame.min() >= 0:
                    return frame
                # If weird values, might be wrong format - try conversion
                return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
            
            # If 2D (grayscale or packed YUYV), convert
            elif len(frame.shape) == 2:
                # Try YUYV conversion (most common for cameras)
                try:
                    return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
                except:
                    # If that fails, convert grayscale to BGR
                    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Other formats - try YUYV conversion as fallback
            else:
                try:
                    return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
                except:
                    # Last resort - return as-is if conversion fails
                    print(f"[WARN] Unknown frame format, shape: {frame.shape}, dtype: {frame.dtype}")
                    return frame
                    
        except Exception as e:
            print(f"[WARN] Frame conversion error: {e}, shape: {frame.shape if frame is not None else 'None'}")
            # Return original frame if conversion fails
            return frame
    
    def capture_frame(self):
        """Capture a single frame - common implementation."""
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
        
        # Convert if needed - use helper to safely handle YUYV/BGR detection
        frame_bgr = self._convert_frame_to_bgr(frame_resized)
        
        # Save to capture_output directory
        output_path = make_capture_frame_path(w, h)
        cv2.imwrite(output_path, frame_bgr)
        print(f"[INFO] Saved {output_path}")
        self.status_label.config(text=f"Saved: {os.path.basename(output_path)}", foreground="blue")
    
    def toggle_record(self):
        """Toggle video recording - common implementation."""
        if not self.cam or not self.cam.is_open():
            messagebox.showinfo("Camera", "Open camera first.")
            return
        
        if self.recording_manager.recording:
            self.recording_manager.stop()
        else:
            self.recording_manager.start()
    
    def on_close(self):
        """Handle application close - common implementation."""
        print("[INFO] Closing application")
        self.preview_manager.stop()
        self.recording_manager.stop()
        self.frame_grabber.stop()
        
        if self.cam:
            self.cam.release()
        
        cv2.destroyAllWindows()
        self.root.destroy()
