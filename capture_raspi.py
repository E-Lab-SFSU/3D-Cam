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

import tkinter as tk
from tkinter import ttk, messagebox

from lib.camera import Camera, get_camera_backend, is_linux, is_raspberry_pi
from lib.camera_info import (
    list_all_cameras, get_camera_info, CameraInfo,
    set_camera_control, get_camera_control, get_camera_control_range
)
from lib.util_paths import make_capture_output_path


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
        
        # Preview state
        self.preview_on = False
        self.preview_thread = None
        self.last_time = time.time()
        self.fps_est = 0.0
        
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
        
        # Build UI
        self._build_ui()
        
        # Detect cameras on startup
        self.refresh_camera_list()
    
    def _build_ui(self):
        """Build the Tkinter GUI."""
        # No canvas - preview will be in OpenCV window
        main = ttk.Frame(self.root, padding="5")
        main.pack(fill=tk.BOTH, expand=True)
        
        # Right side: Controls in a scrollable frame if needed
        ctrl = ttk.Frame(main)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=3, pady=3)
        
        # Camera selection (compact)
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
        
        # Format selection (compact)
        format_frame = ttk.LabelFrame(ctrl, text="Format", padding="3")
        format_frame.pack(fill=tk.X, pady=(0, 4))
        
        # Format and FPS in one row
        fmt_row = ttk.Frame(format_frame)
        fmt_row.pack(fill=tk.X)
        ttk.Label(fmt_row, text="Format:").pack(side=tk.LEFT, padx=(0, 2))
        self.format_combo = ttk.Combobox(fmt_row, textvariable=self.format_var, width=6, state="readonly")
        self.format_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.format_combo.bind("<<ComboboxSelected>>", lambda e: None)  # No need to update desc
        tooltip(self.format_combo, "YUYV = raw/VGA | MJPG = compressed/HD")
        
        ttk.Label(fmt_row, text="FPS:").pack(side=tk.LEFT, padx=(0, 2))
        fps_entry = ttk.Entry(fmt_row, textvariable=self.fps_var, width=5)
        fps_entry.pack(side=tk.LEFT)
        tooltip(fps_entry, "0=max speed, 1-60=specific FPS")
        
        # FPS and Scale status in one compact row
        status_row = ttk.Frame(format_frame)
        status_row.pack(fill=tk.X, pady=(2, 0))
        self.fps_label = ttk.Label(status_row, text="FPS: —", font=("TkDefaultFont", 8))
        self.fps_label.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(status_row, text="Scale:").pack(side=tk.LEFT, padx=(0, 2))
        scale_entry = ttk.Entry(status_row, textvariable=self.scale_percent, width=4)
        scale_entry.pack(side=tk.LEFT, padx=(0, 2))
        tooltip(scale_entry, "Output scale %")
        self.scale_label = ttk.Label(status_row, text="—", font=("TkDefaultFont", 8))
        self.scale_label.pack(side=tk.LEFT)
        
        # Camera controls (compact, 2 columns)
        self.param_frame = ttk.LabelFrame(ctrl, text="Controls", padding="3")
        self.param_frame.pack(fill=tk.X, pady=(0, 4))
        
        self._build_camera_controls()
        
        # Action buttons (compact grid)
        btn_frame = ttk.LabelFrame(ctrl, text="Actions", padding="3")
        btn_frame.pack(fill=tk.X, pady=(0, 4))
        
        btn_grid = ttk.Frame(btn_frame)
        btn_grid.pack()
        
        ttk.Button(btn_grid, text="Open", command=self.open_camera).grid(row=0, column=0, padx=2, pady=1, sticky="ew")
        ttk.Button(btn_grid, text="Preview", command=self.start_preview).grid(row=0, column=1, padx=2, pady=1, sticky="ew")
        ttk.Button(btn_grid, text="Stop", command=self.stop_preview).grid(row=0, column=2, padx=2, pady=1, sticky="ew")
        ttk.Button(btn_grid, text="Capture", command=self.capture_frame).grid(row=1, column=0, padx=2, pady=1, sticky="ew")
        ttk.Button(btn_grid, text="Record", command=self.toggle_record).grid(row=1, column=1, padx=2, pady=1, sticky="ew")
        ttk.Button(btn_grid, text="Reset", command=self.reset_controls).grid(row=1, column=2, padx=2, pady=1, sticky="ew")
        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)
        btn_grid.columnconfigure(2, weight=1)
        
        # Status
        self.status_label = ttk.Label(ctrl, text="Ready", font=("TkDefaultFont", 9, "bold"))
        self.status_label.pack(pady=(4, 0))
    
    def _build_camera_controls(self):
        """Build camera control sliders in compact 2-column layout."""
        # Define controls with defaults
        controls = {
            "brightness": ("Bright", -64, 64, 0),
            "contrast": ("Contrast", 0, 64, 32),
            "saturation": ("Sat", 0, 128, 60),
            "gain": ("Gain", 0, 100, 32),
        }
        
        self.control_vars = {}
        row = 0
        col = 0
        
        for name, (label, default_min, default_max, default_val) in controls.items():
            # Create frame for this control (2 columns)
            frame = ttk.Frame(self.param_frame)
            frame.grid(row=row, column=col, sticky="ew", padx=2, pady=1)
            
            ttk.Label(frame, text=label, width=6, anchor="w", font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=(0, 2))
            
            var = tk.DoubleVar(value=default_val)
            slider = tk.Scale(
                frame, 
                from_=default_min, 
                to=default_max, 
                orient="horizontal", 
                variable=var,
                resolution=1,
                length=70
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            entry = ttk.Entry(frame, textvariable=var, width=4, font=("TkDefaultFont", 8))
            entry.pack(side=tk.LEFT, padx=(2, 0))
            
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
            
            # Move to next row after 2 columns
            col += 1
            if col >= 2:
                col = 0
                row += 1
        
        # Power line frequency dropdown (full width)
        freq_frame = ttk.Frame(self.param_frame)
        freq_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=2, pady=1)
        
        ttk.Label(freq_frame, text="Power:", width=6, anchor="w", font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=(0, 2))
        
        freq_var = tk.IntVar(value=2)
        freq_combo = ttk.Combobox(
            freq_frame, 
            values=["0: Disabled", "1: 50 Hz", "2: 60 Hz"],
            width=12,
            state="readonly"
        )
        freq_combo.set("2: 60 Hz")
        freq_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        def update_freq():
            if self.cam and self.camera_info and self.camera_info.device_path:
                try:
                    sel = int(freq_combo.get().split(":")[0])
                    freq_var.set(sel)
                    if set_camera_control(self.camera_info.device_path, "power_line_frequency", sel):
                        print(f"[DEBUG] power_line_frequency = {sel}")
                except Exception as e:
                    print(f"[WARN] Power line frequency error: {e}")
        
        freq_combo.bind("<<ComboboxSelected>>", lambda e: update_freq())
        self.control_vars["power_line_frequency"] = freq_var
        
        # Configure grid columns
        self.param_frame.columnconfigure(0, weight=1)
        self.param_frame.columnconfigure(1, weight=1)
    
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
        
        # Small delay to ensure camera is fully ready
        time.sleep(0.2)
        
        # Start frame grabber (after camera is fully initialized)
        self._start_frame_grabber()
        
        self.update_scale_info()
        self.status_label.config(text=f"Camera {self.camera_info.index} ready", foreground="green")
        print("[INFO] Camera opened successfully")
        print(f"[INFO] Using automatic exposure (default)")
        print(f"[INFO] Output video will be recorded at {output_fps:.1f} FPS")
        print(f"[INFO] Format: {format_str}, Resolution: {self.cam.w}x{self.cam.h}")
        if format_str == "MJPG":
            print("[INFO] Note: MJPG at 1920x1080 may require more processing time")
    
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
            if ctrl_name == "power_line_frequency":
                continue  # Handled separately
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
        
        # Wait a bit for camera to stabilize (MJPG might need more time)
        time.sleep(1.0)
        
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
    
    def start_preview(self):
        """Start live preview in OpenCV window."""
        if not self.cam or not self.cam.is_open():
            self.open_camera()
            if not self.cam or not self.cam.is_open():
                return
        
        if not self.preview_on:
            print("[INFO] Starting preview")
            self.preview_on = True
            self.last_time = time.time()
            self.fps_est = 0.0
            
            # Start preview thread
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()
    
    def stop_preview(self):
        """Stop live preview."""
        if self.preview_on:
            print("[INFO] Stopping preview")
        self.preview_on = False
        
        # Wait for preview thread to finish
        if self.preview_thread and self.preview_thread.is_alive():
            self.preview_thread.join(timeout=1.0)
        
        # Close OpenCV window
        try:
            cv2.destroyWindow("Preview")
        except:
            pass
    
    def _preview_loop(self):
        """Preview thread loop with OpenCV window."""
        # Set OpenCV to use X11 backend on Raspberry Pi
        os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':0')
        
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preview", 640, 480)
        
        last_time = time.time()
        frame_count = 0
        fps = 0.0
        no_frame_count = 0
        first_frame = True
        
        print("[INFO] Preview: Waiting for frames...")
        
        try:
            while self.preview_on and self.cam and self.cam.is_open():
                try:
                    # Get frame from queue with longer timeout
                    frame = self.frame_queue.get(timeout=1.0)
                    
                    if frame is not None:
                        no_frame_count = 0
                        
                        # Validate frame
                        if len(frame.shape) >= 2 and frame.shape[0] > 0 and frame.shape[1] > 0:
                            # Determine format based on shape and channels
                            if len(frame.shape) == 3 and frame.shape[2] == 3:
                                # Already BGR from OpenCV (MJPG or already decoded)
                                display_frame = frame.copy()
                            else:
                                # Handle YUYV format (2D array, needs conversion)
                                try:
                                    display_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
                                except Exception as e:
                                    print(f"[WARN] Preview: Frame conversion error: {e}, frame shape: {frame.shape}")
                                    continue
                            
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
                            
                            cv2.imshow("Preview", display_frame)
                            # CRITICAL: waitKey is needed for OpenCV to process window events
                            key = cv2.waitKey(1) & 0xFF
                            if key == 27:  # ESC key
                                self.preview_on = False
                                break
                        else:
                            print(f"[WARN] Preview: Invalid frame shape: {frame.shape if frame is not None else 'None'}")
                except Empty:
                    no_frame_count += 1
                    # No frame available
                    if no_frame_count == 1:
                        print("[INFO] Preview: Waiting for frames from grabber...")
                    elif no_frame_count % 20 == 0:
                        print(f"[WARN] Preview: No frames received for {no_frame_count} seconds")
                    
                    if not self.frame_grabber_running:
                        print("[INFO] Preview: Frame grabber stopped, exiting preview")
                        break
                    # Still process window events to keep window responsive
                    cv2.waitKey(1)
                    continue
        except Exception as e:
            print(f"[ERROR] Preview loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                cv2.destroyWindow("Preview")
            except:
                pass
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
            # Stop recording
            self.stop_flag = True
            if self.record_thread and self.record_thread.is_alive():
                self.record_thread.join(timeout=2.0)
            
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            self.recording = False
            self.status_label.config(text="Recording stopped", foreground="black")
            print("[INFO] Recording stopped")
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
            
            print(f"[INFO] Recording started: {output_path}")
    
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
    root = tk.Tk()
    app = CaptureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.bind("<Escape>", lambda e: app.on_close())
    
    print("[INFO] UVC Camera Capture GUI ready")
    root.mainloop()


if __name__ == "__main__":
    main()
