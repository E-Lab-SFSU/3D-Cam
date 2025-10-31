#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi UVC Camera GUI with Camera Info Integration
---------------------------------------------------------
✓ Uses camera_info.py for camera detection and information
✓ Select capture format (YUYV or MJPG) based on camera capabilities
✓ Live adjustable brightness, contrast, saturation, gain
✓ Power line frequency dropdown
✓ MP4 recording
✓ Live preview with debug info
✓ Frame capture
"""

import cv2
import time
import threading
import os
from queue import Queue, Empty
import subprocess
import traceback
import signal

import tkinter as tk
from tkinter import ttk, messagebox

from lib.camera import Camera, get_camera_backend, is_linux, is_raspberry_pi
from lib.camera_info import (
    list_all_cameras, get_camera_info, CameraInfo,
    set_camera_control, get_camera_control, get_camera_control_range
)
from lib.util_paths import make_capture_output_path


# ============ Debug Configuration ============
DEBUG_PREVIEW = True  # Set to False to disable detailed preview debug messages

# Force OpenCV to prefer GTK backend over Qt (more reliable on Raspberry Pi)
# Note: cv2 is already imported above, so this may not take effect,
# but we also set it in the thread as a backup
# Try to prefer GTK, but don't fail if it's not available
# (os is already imported at the top of the file)
os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_GTK', '1')
os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_MSMF', '0')


def debug_print(message, force=False):
    """Print debug message if DEBUG_PREVIEW is enabled."""
    if DEBUG_PREVIEW or force:
        print(f"[DEBUG] {message}")


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


# ============ GUI Application ============
class CaptureApp:
    """Raspberry Pi UVC Camera GUI with full camera control."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("UVC Camera Capture — Raspberry Pi")
        
        # Camera state
        self.cam = None
        self.camera_info = None  # Current CameraInfo object
        self.available_cameras = []  # List of CameraInfo objects
        
        # Format and resolution
        self.format_var = tk.StringVar(value="YUYV")
        self.scale_percent = tk.DoubleVar(value=100.0)
        self.fps_var = tk.DoubleVar(value=30.0)  # Framerate
        
        # Preview state (use simple flag, avoid locks to prevent deadlocks)
        self.preview_on = False
        self.preview_thread = None
        self.last_time = time.time()
        self.fps_est = 0.0
        self.preview_window_name = "Preview"  # Base name, we'll make it unique
        self.preview_window_counter = 0  # Counter to ensure unique window names
        self.preview_window_name_current = None  # Current active window name
        self.opencv_event_processing_active = False  # Flag for OpenCV event processing
        
        # Recording state
        self.recording = False
        self.stop_flag = False
        self.video_writer = None
        self.record_thread = None
        
        # Frame grabbing (single thread feeds both preview and recording)
        self.frame_queue = Queue(maxsize=10)
        self.frame_grabber_thread = None
        self.frame_grabber_running = False
        
        # Camera controls
        self.control_vars = {}
        self.control_ranges = {}  # Store ranges for each control
        
        # Debug state tracking
        self.preview_start_count = 0
        self.preview_stop_count = 0
        
        # Build UI
        self._build_ui()
        
        # Detect cameras on startup
        self.refresh_camera_list()

    def _build_ui(self):
        """Build the Tkinter GUI."""
        # No canvas - preview will be in OpenCV window
        main = ttk.Frame(self.root, padding="5")
        main.pack(fill=tk.BOTH, expand=True)

        # Right side: Controls
        ctrl = ttk.Frame(main)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=3, pady=3)
        
        # Camera selection (same)
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
        
        # Format selection (same)
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
        
        # Status (new section)
        status_frame = ttk.LabelFrame(ctrl, text="Status", padding="3")
        status_frame.pack(fill=tk.X, pady=(0, 4))
        
        self.status_label = ttk.Label(status_frame, text="Ready", font=("TkDefaultFont", 9))
        self.status_label.pack(anchor="w", pady=(0, 2))
        
        self.fps_label = ttk.Label(status_frame, text="FPS: —", font=("TkDefaultFont", 8))
        self.fps_label.pack(anchor="w")
        
        # Open/Close Camera button
        self.open_camera_btn = ttk.Button(status_frame, text="Open Camera", command=self.toggle_camera)
        self.open_camera_btn.pack(fill=tk.X, pady=(4, 0))
        
        # Camera controls (stacked sliders for maximum horizontal adjustment)
        self.param_frame = ttk.LabelFrame(ctrl, text="Controls", padding="3")
        self.param_frame.pack(fill=tk.X, pady=(0, 4))
        
        self._build_camera_controls()
        
        # Actions section
        btn_frame = ttk.LabelFrame(ctrl, text="Actions", padding="3")
        btn_frame.pack(fill=tk.X, pady=(0, 4))
        
        # Open/Close Preview button
        self.preview_btn = ttk.Button(btn_frame, text="Open Preview", command=self.toggle_preview)
        self.preview_btn.pack(fill=tk.X, pady=(0, 2))
        
        # Record/Stop Record button
        self.record_btn = ttk.Button(btn_frame, text="Record", command=self.toggle_record)
        self.record_btn.pack(fill=tk.X, pady=(0, 2))
        
        # Capture Frame button
        ttk.Button(btn_frame, text="Capture Frame", command=self.capture_frame).pack(fill=tk.X, pady=(0, 2))
        
        # Reset Controls button
        ttk.Button(btn_frame, text="Reset Controls", command=self.reset_controls).pack(fill=tk.X)
    
    def _build_camera_controls(self):
        """Build camera control sliders stacked vertically for maximum horizontal space."""
        # Define controls with defaults (removed power line frequency - always 60)
        controls = {
            "brightness": ("Brightness", -64, 64, 0),
            "contrast": ("Contrast", 0, 64, 32),
            "saturation": ("Saturation", 0, 128, 60),
            "gain": ("Gain", 0, 100, 32),
        }
        
        self.control_vars = {}
        
        for name, (label, default_min, default_max, default_val) in controls.items():
            # Create frame for this control (stacked vertically)
            frame = ttk.Frame(self.param_frame)
            frame.pack(fill=tk.X, pady=1)
            
            ttk.Label(frame, text=label, width=10, anchor="w", font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(0, 4))
            
            var = tk.DoubleVar(value=default_val)
            slider = tk.Scale(
                frame, 
                from_=default_min, 
                to=default_max, 
                orient="horizontal", 
                variable=var,
                resolution=1,
                length=150
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
            
            entry = ttk.Entry(frame, textvariable=var, width=6, font=("TkDefaultFont", 9))
            entry.pack(side=tk.LEFT)
            
            # Update callback
            def update_control(ctrl_name=name, ctrl_var=var):
                if self.cam and self.camera_info and self.camera_info.device_path:
                    val = int(ctrl_var.get())
                    if set_camera_control(self.camera_info.device_path, ctrl_name, val):
                        print(f"[DEBUG] {ctrl_name} = {val}")
                    else:
                        print(f"[WARN] Failed to set {ctrl_name}")
            
            var.trace_add("write", lambda *a, update=update_control: update())
            entry.bind("<Return>", lambda e, update=update_control: update())
            
            self.control_vars[name] = var
        
        # Set power line frequency to 60 Hz automatically when camera opens
        # (no UI control, always 60 Hz)
    
    def refresh_camera_list(self):
        """Refresh the list of available cameras."""
        print("[INFO] Refreshing camera list...")
        self.available_cameras = list_all_cameras(max_test=10, detailed=False, test_resolutions=False)
        
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
    
    def _update_format_desc(self):
        """Update format description."""
        fmt = self.format_var.get()
        if fmt == "MJPG":
            txt = ("MJPG: JPEG-compressed frames.\n"
                   "• Pros: supports 720p/1080p, lower USB load.\n"
                   "• Cons: slight latency, compression artifacts.")
        else:
            txt = ("YUYV: uncompressed 4:2:2 stream.\n"
                   "• Pros: fast preview, low latency.\n"
                   "• Cons: limited to VGA on USB 2.0.")
        self.format_desc.config(text=txt)
    
    def toggle_camera(self):
        """Toggle camera open/close."""
        if self.cam and self.cam.is_open():
            self.close_camera()
        else:
            self.open_camera()
    
    def close_camera(self):
        """Close the camera."""
        # Stop preview before closing camera
        if self.preview_on:
            self.stop_preview()
        
        if self.recording:
            self.stop_record()
        
        self._stop_frame_grabber()
        
        if self.cam:
            self.cam.release()
            self.cam = None
        
        self.open_camera_btn.config(text="Open Camera")
        self.status_label.config(text="Camera closed", foreground="black")
        print("[INFO] Camera closed")
    
    def open_camera(self):
        """Open the selected camera."""
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
        
        # Determine resolution based on format
        if format_str == "MJPG":
            width, height = 1920, 1080
        else:
            width, height = 640, 480
        
        # Get framerate from input (0 = maximum speed, >0 = specific FPS)
        try:
            fps_value = float(self.fps_var.get())
            # Clamp between 0 (max speed) and 60
            fps_value = max(0.0, min(60.0, fps_value))
            self.fps_var.set(fps_value)
        except (ValueError, tk.TclError):
            fps_value = 30.0  # Default to 30 FPS if invalid
            self.fps_var.set(fps_value)
        
        self.cam = Camera(
            index=self.camera_info.index,
            fps=int(fps_value) if fps_value > 0 else 0,  # 0 = max speed, >0 = specific FPS
            fourcc=format_str,
            width=width,
            height=height,
            backend=cv2.CAP_V4L2
        )
        
        if not self.cam.open():
            messagebox.showerror("Camera Error", f"Failed to open camera {self.camera_info.index}")
            return
        
        # Stop frame grabber if running
        self._stop_frame_grabber()
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        # Load camera control ranges
        if self.camera_info.device_path:
            self._load_camera_control_ranges()
            # Apply current control values
            self._apply_camera_controls()
            # Set power line frequency to 60 Hz automatically
            set_camera_control(self.camera_info.device_path, "power_line_frequency", 2)
        
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
        
        # No delay - start immediately
        
        # Start frame grabber (after camera is fully initialized)
        self._start_frame_grabber()
        
        self.update_scale_info()
        
        # Update UI
        self.open_camera_btn.config(text="Close Camera")
        self.status_label.config(text=f"Camera {self.camera_info.index} ready", foreground="green")
        if actual_fps == 0:
            self.fps_label.config(text="FPS: Max speed")
        else:
            self.fps_label.config(text=f"FPS: {actual_fps:.1f}")
        
        print("[INFO] Camera opened successfully")
        print(f"[INFO] Using automatic exposure (default)")
        print(f"[INFO] Power line frequency set to 60 Hz")
        print(f"[INFO] Output video will be recorded at {output_fps:.1f} FPS")
        print(f"[INFO] Format: {format_str}, Resolution: {self.cam.w}x{self.cam.h}")
        if format_str == "MJPG":
            print("[INFO] Note: MJPG at 1920x1080 may require more processing time")
        
        # Automatically start preview when camera opens
        if not self.preview_on:
            debug_print("Auto-starting preview after camera opened...")
            self.root.after(100, self.start_preview)  # Small delay to ensure camera is ready
    
    def _load_camera_control_ranges(self):
        """Load control ranges from camera."""
        if not self.camera_info or not self.camera_info.device_path:
            return
        
        controls_to_check = ["brightness", "contrast", "saturation", "gain"]
        for ctrl_name in controls_to_check:
            range_info = get_camera_control_range(self.camera_info.device_path, ctrl_name)
            if range_info:
                self.control_ranges[ctrl_name] = range_info
                # Update slider range if found
                if ctrl_name in self.control_vars:
                    # Find the slider widget
                    for widget in self.param_frame.winfo_children():
                        if isinstance(widget, ttk.Frame):
                            for child in widget.winfo_children():
                                if isinstance(child, tk.Scale):
                                    # Check if this is the right control by checking label
                                    prev_widget = None
                                    for w in widget.winfo_children():
                                        if isinstance(w, ttk.Label):
                                            if ctrl_name.lower() in w.cget("text").lower():
                                                child.config(from_=range_info['min'], to=range_info['max'])
                                                self.control_vars[ctrl_name].set(range_info.get('default', 0))
                                                break
    
    def _apply_camera_controls(self):
        """Apply current control values to camera."""
        if not self.camera_info or not self.camera_info.device_path:
            return
        
        for ctrl_name, var in self.control_vars.items():
            val = int(var.get())
            set_camera_control(self.camera_info.device_path, ctrl_name, val)
    
    def update_scale_info(self):
        """Update scale information display."""
        if not self.cam:
            return
        
        p = max(1, min(100, float(self.scale_percent.get())))
        sw, sh = int(self.cam.w * p / 100), int(self.cam.h * p / 100)
        self.scale_label.config(text=f"{sw}×{sh}")
        print(f"[INFO] {self.format_var.get()} {self.cam.w}×{self.cam.h} → {sw}×{sh} ({p:.1f}%)")
    
    def scaled_size(self):
        """Get scaled output size."""
        if not self.cam:
            return 640, 480
        p = max(1, min(100, float(self.scale_percent.get())))
        return int(self.cam.w * p / 100), int(self.cam.h * p / 100)
    
    def _start_frame_grabber(self):
        """Start the frame grabbing thread."""
        if self.frame_grabber_running:
            return
        self.frame_grabber_running = True
        self.frame_grabber_thread = threading.Thread(target=self._frame_grabber_loop, daemon=True)
        self.frame_grabber_thread.start()
        print("[INFO] Frame grabber started")
    
    def _stop_frame_grabber(self):
        """Stop the frame grabbing thread."""
        self.frame_grabber_running = False
        if self.frame_grabber_thread and self.frame_grabber_thread.is_alive():
            self.frame_grabber_thread.join(timeout=2.0)
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        print("[INFO] Frame grabber stopped")
    
    def _frame_grabber_loop(self):
        """Single thread that reads frames from camera and feeds the queue."""
        consecutive_errors = 0
        max_errors = 100  # More lenient for slow cameras
        frames_read = 0
        last_status_time = time.time()
        
        # No delay - start immediately
        
        print("[INFO] Frame grabber: Starting frame capture...")
        
        while self.frame_grabber_running and self.cam and self.cam.is_open():
            frame = self.cam.read(max_retries=1)  # Use fewer retries to avoid blocking
            if frame is not None:
                consecutive_errors = 0
                frames_read += 1
                
                # Put frame in queue (drop old frame if queue is full)
                try:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    
                    self.frame_queue.put_nowait(frame)
                    
                    # Print status every 30 frames
                    if frames_read % 30 == 0:
                        elapsed = time.time() - last_status_time
                        actual_fps = 30.0 / elapsed if elapsed > 0 else 0
                        print(f"[INFO] Frame grabber: {frames_read} frames read ({actual_fps:.1f} FPS avg)")
                        last_status_time = time.time()
                except Exception as e:
                    print(f"[WARN] Frame queue error: {e}")
            else:
                consecutive_errors += 1
                
                # Print warning every 10 consecutive errors
                if consecutive_errors % 10 == 0 and consecutive_errors < max_errors:
                    print(f"[WARN] Frame grabber: {consecutive_errors} consecutive read failures (timeout expected for V4L2)")
                
                if consecutive_errors >= max_errors:
                    print(f"[ERROR] Too many consecutive frame read errors ({consecutive_errors}), stopping frame grabber")
                    print(f"[INFO] Frame grabber: Total frames read: {frames_read}")
                    break
                
                # Shorter sleep on error - don't wait too long between attempts
                time.sleep(0.05)
    
    def toggle_preview(self):
        """Toggle preview on/off."""
        debug_print(f"=== toggle_preview() called, current state: preview_on={self.preview_on} ===")
        if self.preview_on:
            debug_print("Preview is on, calling stop_preview()...")
            self.stop_preview()
        else:
            debug_print("Preview is off, calling start_preview()...")
            self.start_preview()
    
    def start_preview(self):
        """Start live preview in OpenCV window."""
        self.preview_start_count += 1
        debug_print(f"=== start_preview() called (count: {self.preview_start_count}) ===")
        debug_print(f"Current state: preview_on={self.preview_on}, thread={self.preview_thread}, thread_alive={self.preview_thread.is_alive() if self.preview_thread else None}")
        
        # Check if preview is already running with a valid window
        if self.preview_on and self.preview_window_name_current:
            # Check if window still exists
            try:
                window_visible = cv2.getWindowProperty(self.preview_window_name_current, cv2.WND_PROP_VISIBLE)
                if window_visible >= 0:
                    debug_print("Preview already running with valid window, skipping start")
                    return
            except:
                # Window doesn't exist, continue to recreate
                debug_print("Preview flag set but window doesn't exist, recreating...")
                self.preview_on = False
                self.preview_window_name_current = None
        
        if not self.cam or not self.cam.is_open():
            debug_print("Camera not open, cannot start preview")
            return
        
        # Ensure previous thread is fully stopped
        if self.preview_thread is not None:
            debug_print(f"Previous thread exists: alive={self.preview_thread.is_alive()}")
            if self.preview_thread.is_alive():
                debug_print("Previous preview thread still alive, stopping it first...")
                print("[INFO] Previous preview thread still alive, stopping it first...")
                self.preview_on = False  # Signal stop
                debug_print("Waiting for thread to join (timeout=0.5)...")
                self.preview_thread.join(timeout=0.5)  # Short timeout
                if self.preview_thread.is_alive():
                    debug_print("WARN: Previous preview thread did not stop in time")
                    print("[WARN] Previous preview thread did not stop in time, continuing anyway")
                else:
                    debug_print("Previous thread joined successfully")
            # Reset thread reference regardless
            self.preview_thread = None
            debug_print("Thread reference reset to None")
        
        # Ensure flag is clear
        self.preview_on = False
        debug_print("Flag set to False before cleanup")
        
        # Clean up any existing window and reset OpenCV state
        debug_print("Starting window cleanup in start_preview() (main thread)...")
        # Use destroyAllWindows to fully reset OpenCV's window state
        debug_print("Calling cv2.destroyAllWindows() to reset OpenCV state...")
        try:
            cv2.destroyAllWindows()
            # Process events multiple times to ensure cleanup
            for _ in range(5):
                cv2.waitKey(10)
            debug_print("destroyAllWindows() succeeded, processed events")
        except Exception as e:
            debug_print(f"Exception in destroyAllWindows: {type(e).__name__}: {e}")
        
        # Small delay to let OpenCV fully reset
        debug_print("Waiting 0.15s for OpenCV to fully reset...")
        time.sleep(0.15)
        debug_print("Cleanup complete")
        
        # Create window in main thread (required for Qt backend)
        self.preview_window_counter += 1
        window_name = f"{self.preview_window_name}_{self.preview_window_counter}"
        debug_print(f"Creating window in main thread: {window_name}")
        
        try:
            # Determine window flags
            try:
                window_flags = cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED
            except AttributeError:
                window_flags = cv2.WINDOW_NORMAL
            
            debug_print(f"Calling cv2.namedWindow('{window_name}', ...) in main thread...")
            cv2.namedWindow(window_name, window_flags)
            cv2.resizeWindow(window_name, 640, 480)
            # Process events to ensure window is created
            for _ in range(3):
                cv2.waitKey(50)
            
            # Verify window was created
            verify_prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            if verify_prop < 0:
                raise cv2.error("Window verification failed")
            
            self.preview_window_name_current = window_name
            debug_print(f"Window '{window_name}' created successfully in main thread")
            print(f"[INFO] Preview window created: {window_name}")
        except Exception as e:
            debug_print(f"ERROR: Failed to create window in main thread: {type(e).__name__}: {e}")
            traceback.print_exc()
            self.preview_on = False
            messagebox.showerror("Preview Error", f"Failed to create preview window: {e}")
            return
        
        # Now set flag to True and start thread
        self.preview_on = True
        debug_print("Flag set to True, starting preview thread...")
        
        print("[INFO] Starting preview")
        self.last_time = time.time()
        self.fps_est = 0.0
        
        # Update button
        self.preview_btn.config(text="Close Preview")
        debug_print("Button text updated to 'Close Preview'")
        
        # Start periodic OpenCV event processing in main thread
        self._start_opencv_event_processing()
        
        # Start preview thread
        try:
            debug_print("Creating preview thread...")
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            debug_print("Starting preview thread...")
            self.preview_thread.start()
            debug_print(f"Preview thread started: thread_id={self.preview_thread.ident}, alive={self.preview_thread.is_alive()}")
        except Exception as e:
            debug_print(f"ERROR: Failed to start preview thread: {type(e).__name__}: {e}")
            traceback.print_exc()
            self.preview_on = False
            self.preview_thread = None
            raise
    
    def stop_preview(self):
        """Stop live preview."""
        self.preview_stop_count += 1
        debug_print(f"=== stop_preview() called (count: {self.preview_stop_count}) ===")
        debug_print(f"Current state: preview_on={self.preview_on}, thread={self.preview_thread}, thread_alive={self.preview_thread.is_alive() if self.preview_thread else None}")
        
        if not self.preview_on:
            debug_print("Preview not on, returning early")
            return
        
        print("[INFO] Stopping preview")
        debug_print("Setting preview_on flag to False...")
        
        # Set flag to stop loop (thread will clean up)
        self.preview_on = False
        
        # Update button immediately
        self.preview_btn.config(text="Open Preview")
        debug_print("Button text updated to 'Open Preview'")
        
        # Wait for thread to exit with longer timeout
        if self.preview_thread and self.preview_thread.is_alive():
            debug_print(f"Waiting for preview thread to exit (timeout=2.0)... thread_id={self.preview_thread.ident}")
            self.preview_thread.join(timeout=2.0)  # Longer timeout to allow cleanup
            if self.preview_thread.is_alive():
                debug_print("WARN: Preview thread did not exit in time, but continuing cleanup")
            else:
                debug_print("Preview thread exited successfully")
        else:
            debug_print("No thread or thread not alive, skipping join")
        
        # Stop OpenCV event processing
        self.opencv_event_processing_active = False
        
        # Force cleanup of window in main thread (where it was created)
        debug_print("Starting window cleanup in stop_preview() (main thread)...")
        if self.preview_window_name_current:
            window_name = self.preview_window_name_current
            debug_print(f"Destroying window '{window_name}' in main thread...")
            try:
                cv2.destroyWindow(window_name)
                for _ in range(5):
                    cv2.waitKey(10)
                debug_print(f"Window '{window_name}' destroyed successfully")
            except Exception as e:
                debug_print(f"Exception destroying window: {type(e).__name__}: {e}")
            self.preview_window_name_current = None
        
        # Also call destroyAllWindows as backup
        debug_print("Calling cv2.destroyAllWindows() in stop_preview()...")
        try:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(10)
            debug_print("destroyAllWindows() succeeded in stop_preview()")
        except Exception as e:
            debug_print(f"Exception in destroyAllWindows (stop_preview): {type(e).__name__}: {e}")
        
        # Small delay to ensure cleanup completes
        debug_print("Waiting 0.15s for cleanup to complete...")
        time.sleep(0.15)
        
        # Reset thread reference
        self.preview_thread = None
        debug_print("Thread reference reset to None")
        debug_print("stop_preview() complete")
        
        print("[INFO] Preview stopped")

    def _start_opencv_event_processing(self):
        """Start periodic OpenCV event processing in main thread."""
        if self.opencv_event_processing_active:
            return
        
        self.opencv_event_processing_active = True
        debug_print("Starting OpenCV event processing in main thread")
        self._process_opencv_events()
    
    def _process_opencv_events(self):
        """Process OpenCV window events in main thread and auto-reopen if closed."""
        if not self.preview_on:
            self.opencv_event_processing_active = False
            return
        
        # If camera is closed, stop preview
        if not self.cam or not self.cam.is_open():
            self.opencv_event_processing_active = False
            return
        
        try:
            # Check if window exists and is visible
            window_name = self.preview_window_name_current
            if not window_name:
                # No window name, try to reopen
                debug_print("No window name, reopening preview...")
                self.root.after(100, self.start_preview)
                self.opencv_event_processing_active = False
                return
            
            try:
                window_visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                if window_visible < 1:
                    # Window was closed, automatically reopen it
                    debug_print("Window closed by user, auto-reopening preview...")
                    self.preview_window_name_current = None  # Reset window name
                    self.root.after(100, self.start_preview)  # Reopen after short delay
                    self.opencv_event_processing_active = False
                    return
                # Process events and check for ESC key (but don't close, just ignore ESC)
                cv2.waitKey(1)
            except cv2.error:
                # Window doesn't exist, try to reopen
                debug_print("Window no longer exists, auto-reopening preview...")
                self.preview_window_name_current = None  # Reset window name
                self.root.after(100, self.start_preview)  # Reopen after short delay
                self.opencv_event_processing_active = False
                return
        except Exception as e:
            debug_print(f"Exception in OpenCV event processing: {type(e).__name__}: {e}")
        
        # Schedule next event processing
        if self.opencv_event_processing_active:
            self.root.after(50, self._process_opencv_events)  # Every 50ms
    
    def _preview_loop(self):
        """Preview thread loop - only displays frames, window created in main thread."""
        thread_id = threading.current_thread().ident
        debug_print(f"=== _preview_loop() started (thread_id={thread_id}) ===")
        
        # Get window name from main thread
        window_name = self.preview_window_name_current
        if not window_name:
            debug_print("ERROR: No window name available")
            self.preview_on = False
            return
        
        debug_print(f"Preview loop using window: {window_name}")
        
        # Initialize loop variables early so they're always available in finally block
        last_time = time.time()
        frame_count = 0
        fps = 0.0
        no_frame_count = 0
        first_frame = True
        
        try:
            # Window already created in main thread, just start the display loop
            print("[INFO] Preview: Waiting for frames...")
            debug_print("Entering main preview loop...")
            
            while self.preview_on:
                # Check if camera is still valid
                if not self.cam or not self.cam.is_open():
                    debug_print("Camera not valid or not open, exiting loop")
                    print("[INFO] Preview loop exiting (camera closed)")
                    break
                
                # Window visibility is checked in main thread via _process_opencv_events()
                # Just continue with frame display
                
                # Check flag again before blocking
                if not self.preview_on:
                    debug_print("preview_on flag is False, exiting loop")
                    break
                
                try:
                    # Get frame from queue with timeout
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # Check flag again after potentially blocking
                    if not self.preview_on:
                        break
                    
                    if frame is not None:
                        no_frame_count = 0
                        
                        # Validate frame
                        if len(frame.shape) >= 2 and frame.shape[0] > 0 and frame.shape[1] > 0:
                            try:
                                # Determine format based on shape and channels
                                if len(frame.shape) == 3 and frame.shape[2] == 3:
                                    # Already BGR from OpenCV (MJPG or already decoded)
                                    display_frame = frame.copy()
                                else:
                                    # Handle YUYV format (2D array, needs conversion)
                                    display_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
                                
                                if first_frame:
                                    print(f"[INFO] Preview: First frame received! Shape: {display_frame.shape}, Format: {self.format_var.get()}")
                                    first_frame = False
                                
                                # Calculate FPS
                                frame_count += 1
                                now = time.time()
                                if frame_count % 10 == 0:
                                    elapsed = now - last_time
                                    fps = 10.0 / elapsed if elapsed > 0 else 0
                                    last_time = now
                                
                                # Draw overlay
                                cv2.putText(
                                    display_frame,
                                    f"{self.format_var.get()} {self.cam.w}x{self.cam.h}  FPS:{fps:.1f}",
                                    (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2
                                )
                                
                                # Check recording state
                                if self.recording:
                                    cv2.putText(
                                        display_frame,
                                        "REC",
                                        (10, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 0, 255),
                                        2
                                    )
                                
                                # Show frame (window exists, created in main thread)
                                try:
                                    cv2.imshow(window_name, display_frame)
                                    # Note: waitKey() is handled in main thread via _process_opencv_events()
                                except cv2.error as e:
                                    debug_print(f"cv2.error in imshow: {e}, window may be closed")
                                    # Window may have been closed - main thread will detect this and reopen
                                    # Don't break, just continue - the main thread will handle reopening
                                    pass
                            except Exception as e:
                                debug_print(f"Frame processing error: {type(e).__name__}: {e}")
                                print(f"[WARN] Preview: Frame processing error: {e}")
                                traceback.print_exc()
                                continue
                except Empty:
                    # No frame available
                    no_frame_count += 1
                    if no_frame_count == 1:
                        debug_print("No frame available, waiting for frames from grabber...")
                        print("[INFO] Preview: Waiting for frames from grabber...")
                    
                    if not self.frame_grabber_running:
                        debug_print("Frame grabber stopped, exiting preview")
                        print("[INFO] Preview: Frame grabber stopped, exiting preview")
                        break
                    
                    # Window events are processed in main thread via _process_opencv_events()
                    continue
                except Exception as e:
                    debug_print(f"Unexpected error in preview loop: {type(e).__name__}: {e}")
                    print(f"[WARN] Preview: Unexpected error: {e}")
                    traceback.print_exc()
                    continue
        except Exception as e:
            debug_print(f"CRITICAL: Exception in preview loop try block: {type(e).__name__}: {e}")
            print(f"[ERROR] Preview loop error: {e}")
            traceback.print_exc()
        finally:
            debug_print("=== Entering finally block for preview cleanup ===")
            # Note: Window destruction is handled in stop_preview() in main thread
            # Just reset the flag here (thread cleanup only)
            debug_print("Setting preview_on flag to False in finally block")
            self.preview_on = False
            
            # Update button in main thread
            try:
                debug_print("Updating button text to 'Open Preview' in finally block")
                self.root.after(0, lambda: self.preview_btn.config(text="Open Preview"))
            except Exception as e:
                debug_print(f"Failed to update button: {type(e).__name__}: {e}")
            
            debug_print(f"_preview_loop() exiting. Total frames displayed: {frame_count}, thread_id={thread_id}")
            print(f"[INFO] Preview: Exited. Total frames displayed: {frame_count}")
    
    def reset_controls(self):
        """Reset camera controls to defaults."""
        if not self.cam or not self.camera_info:
            return
        
        defaults = {
            "brightness": 0,
            "contrast": 32,
            "saturation": 60,
            "gain": 32
        }
        
        for name, default_val in defaults.items():
            if name in self.control_vars:
                # Use range default if available
                if name in self.control_ranges:
                    default_val = self.control_ranges[name].get('default', default_val)
                self.control_vars[name].set(default_val)
                if self.camera_info.device_path:
                    set_camera_control(self.camera_info.device_path, name, default_val)
        
        print("[INFO] Controls reset to defaults")
    
    def capture_frame(self):
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
        
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = f"frame_{w}x{h}_{ts}.png"
        cv2.imwrite(name, frame_bgr)
        print(f"[INFO] Saved {name}")
        self.status_label.config(text=f"Saved: {name}", foreground="blue")
    
    def toggle_record(self):
        """Toggle video recording."""
        if not self.cam or not self.cam.is_open():
            messagebox.showinfo("Camera", "Open camera first.")
            return
        
        if self.recording:
            self.stop_record()
        else:
            # Start recording
            w, h = self.scaled_size()
            
            # Get output FPS (use actual camera FPS, or 30 if max speed)
            actual_fps = self.cam.cap.get(cv2.CAP_PROP_FPS) if self.cam.cap else 0
            output_fps = actual_fps if actual_fps > 0 else 30.0
            
            output_path = make_capture_output_path(w, h, int(output_fps))
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # Use consistent output FPS for smooth playback
            self.video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))
            
            if not self.video_writer.isOpened():
                messagebox.showerror("Recording Error", f"Failed to create video file: {output_path}")
                return
            
            self.stop_flag = False
            self.recording = True
            self.status_label.config(text=f"Recording: {os.path.basename(output_path)}", foreground="red")
            
            # Start recording thread
            self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
            self.record_thread.start()
            
            # Update button
            self.record_btn.config(text="Stop Record")
            
            print(f"[INFO] Recording started: {output_path}")
    
    def stop_record(self):
        """Stop video recording."""
        if not self.recording:
            return
        
        # Stop recording
        self.stop_flag = True
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=2.0)
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.recording = False
        self.record_btn.config(text="Record")
        self.status_label.config(text="Recording stopped", foreground="black")
        print("[INFO] Recording stopped")
    
    def _record_loop(self):
        """Recording thread loop with consistent output FPS."""
        frame_count = 0
        dropped_frames = 0
        skipped_frames = 0
        t0 = time.time()
        
        # Get output FPS (use actual camera FPS, or 30 if max speed)
        actual_fps = self.cam.cap.get(cv2.CAP_PROP_FPS) if self.cam.cap else 0
        output_fps = actual_fps if actual_fps > 0 else 30.0
        
        # Frame timing for consistent output FPS
        frame_interval = 1.0 / output_fps  # Time between frames at output FPS
        next_frame_time = t0
        last_frame = None
        
        while not self.stop_flag and self.cam and self.cam.is_open():
            try:
                # Get frame (non-blocking with timeout)
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except Empty:
                    # No frame available, check if we should write last frame (duplicate)
                    # to maintain consistent output FPS
                    current_time = time.time()
                    if current_time >= next_frame_time and last_frame is not None:
                        # Time to write next frame, use last frame to maintain FPS
                        w, h = self.scaled_size()
                        if w != self.cam.w or h != self.cam.h:
                            frame_resized = cv2.resize(last_frame, (w, h))
                        else:
                            frame_resized = last_frame.copy()
                        
                        # Convert if needed (ensure BGR)
                        if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
                            frame_bgr = frame_resized
                        else:
                            frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_YUV2BGR_YUY2)
                        
                        self.video_writer.write(frame_bgr)
                        frame_count += 1
                        next_frame_time += frame_interval
                    continue
                
                if frame is not None:
                    last_frame = frame  # Store for potential duplication
                    
                    # Scale frame
                    w, h = self.scaled_size()
                    if w != self.cam.w or h != self.cam.h:
                        frame_resized = cv2.resize(frame, (w, h))
                    else:
                        frame_resized = frame
                    
                    # Convert if needed (ensure BGR)
                    if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
                        frame_bgr = frame_resized
                    else:
                        frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_YUV2BGR_YUY2)
                    
                    # Write frames at consistent intervals for smooth playback
                    current_time = time.time()
                    
                    # If it's time to write, write immediately
                    if current_time >= next_frame_time:
                        self.video_writer.write(frame_bgr)
                        frame_count += 1
                        next_frame_time += frame_interval
                        
                        # If we're significantly behind, skip ahead to current time
                        # (don't accumulate too much lag)
                        if current_time > next_frame_time + frame_interval * 2:
                            next_frame_time = current_time + frame_interval
                    else:
                        # Frame came too early - skip it to maintain timing
                        # We'll use the next frame when it's time
                        skipped_frames += 1
                        
            except Exception as e:
                print(f"[WARN] Frame write error: {e}")
                dropped_frames += 1
        
        duration = time.time() - t0
        fps_capture_avg = frame_count / duration if duration > 0 else 0
        
        print(f"[INFO] Recorded {frame_count} frames in {duration:.1f}s")
        print(f"[INFO] Average capture rate: {fps_capture_avg:.1f} FPS")
        print(f"[INFO] Output video FPS: {output_fps:.1f} FPS (consistent)")
        if skipped_frames > 0:
            print(f"[INFO] Skipped {skipped_frames} early frames to maintain consistent FPS")
        if dropped_frames > 0:
            print(f"[INFO] Dropped {dropped_frames} frames due to errors")
    

    def on_close(self):
        """Handle application close."""
        print("[INFO] Closing application")
        self.stop_preview()
        self.stop_flag = True
        
        # Wait for recording thread
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=2.0)
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self._stop_frame_grabber()
        
        if self.cam:
            self.cam.release()
        
        cv2.destroyAllWindows()
        self.root.destroy()


# ============ Main ============
def main():
    """Main entry point."""
    print(f"[INFO] UVC Camera Capture GUI starting...")
    print(f"[INFO] Preview debug mode: {'ENABLED' if DEBUG_PREVIEW else 'DISABLED'}")
    if DEBUG_PREVIEW:
        print("[INFO] Detailed debug messages will be printed for preview operations")
    
    root = tk.Tk()
    app = CaptureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.bind("<Escape>", lambda e: app.on_close())
    
    print("[INFO] UVC Camera Capture GUI ready")
    root.mainloop()


if __name__ == "__main__":
    main()
