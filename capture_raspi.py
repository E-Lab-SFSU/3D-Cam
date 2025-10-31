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
        self.output_fps = 30.0  # Consistent output FPS for video file
        
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
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill=tk.BOTH, expand=True)
        
        # Right side: Controls
        ctrl = ttk.Frame(main)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)
        
        # Camera selection
        cam_frame = ttk.LabelFrame(ctrl, text="Camera Selection", padding="5")
        cam_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(cam_frame, text="Camera:").pack(anchor="w")
        self.camera_combo = ttk.Combobox(cam_frame, width=25, state="readonly")
        self.camera_combo.pack(pady=2, fill=tk.X)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)
        
        ttk.Button(cam_frame, text="Refresh", command=self.refresh_camera_list).pack(pady=2)
        
        # Camera info display
        self.camera_info_label = ttk.Label(
            cam_frame, 
            text="No camera selected", 
            justify="left",
            wraplength=200
        )
        self.camera_info_label.pack(pady=2, fill=tk.X)
        
        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=6)
        
        # Format selection
        format_frame = ttk.LabelFrame(ctrl, text="Capture Format", padding="5")
        format_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(format_frame, text="Format:").pack(anchor="w")
        self.format_combo = ttk.Combobox(format_frame, textvariable=self.format_var, width=8, state="readonly")
        self.format_combo.pack(pady=2)
        self.format_combo.bind("<<ComboboxSelected>>", lambda e: self._update_format_desc())
        
        tooltip(self.format_combo,
                "YUYV = raw, fastest, limited to ~640×480.\n"
                "MJPG = compressed, allows 720p/1080p but higher CPU decode.")
        
        self.format_desc = tk.Label(
            format_frame, 
            justify="left", 
            bg="#eef", 
            relief="groove",
            wraplength=200
        )
        self.format_desc.pack(fill=tk.X, pady=2)
        self._update_format_desc()
        
        # Scale
        ttk.Label(format_frame, text="Output Scale (%):").pack(anchor="w", pady=(5, 0))
        scale_entry = ttk.Entry(format_frame, textvariable=self.scale_percent, width=8)
        scale_entry.pack(pady=2)
        tooltip(scale_entry, "Resize output relative to native resolution (1–100 %)")
        ttk.Button(format_frame, text="Apply Scale", command=self.update_scale_info).pack(pady=3)
        self.scale_label = ttk.Label(format_frame, text="Scaled: —")
        self.scale_label.pack()
        
        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=6)
        
        # Camera controls
        self.param_frame = ttk.LabelFrame(ctrl, text="Camera Controls", padding="5")
        self.param_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._build_camera_controls()
        
        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=6)
        
        # Action buttons
        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Open Camera", command=self.open_camera).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Start Preview", command=self.start_preview).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Stop Preview", command=self.stop_preview).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Capture Frame", command=self.capture_frame).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Toggle Record", command=self.toggle_record).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Reset Controls", command=self.reset_controls).pack(fill=tk.X, pady=2)
        
        # Status
        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=6)
        self.status_label = ttk.Label(ctrl, text="Ready", font=("TkDefaultFont", 10, "bold"))
        self.status_label.pack(pady=5)
    
    def _build_camera_controls(self):
        """Build camera control sliders."""
        # Define controls with defaults
        controls = {
            "brightness": ("Brightness", -64, 64, 0),
            "contrast": ("Contrast", 0, 64, 32),
            "saturation": ("Saturation", 0, 128, 60),
            "gain": ("Gain", 0, 100, 32),
        }
        
        self.control_vars = {}
        
        for name, (label, default_min, default_max, default_val) in controls.items():
            # Create frame for this control
            frame = ttk.Frame(self.param_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=label, width=12, anchor="w").pack(side=tk.LEFT, padx=(0, 5))
            
            var = tk.DoubleVar(value=default_val)
            slider = tk.Scale(
                frame, 
                from_=default_min, 
                to=default_max, 
                orient="horizontal", 
                variable=var,
                resolution=1,
                length=120
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            
            entry = ttk.Entry(frame, textvariable=var, width=6)
            entry.pack(side=tk.LEFT, padx=2)
            
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
        
        # Power line frequency dropdown
        freq_frame = ttk.Frame(self.param_frame)
        freq_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(freq_frame, text="Power Line:", width=12, anchor="w").pack(side=tk.LEFT, padx=(0, 5))
        
        freq_var = tk.IntVar(value=2)
        freq_combo = ttk.Combobox(
            freq_frame, 
            values=["0: Disabled", "1: 50 Hz", "2: 60 Hz"],
            width=15,
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
        
        # Don't limit FPS - let camera run at maximum speed
        # Use 0 to auto-detect or let camera choose its maximum
        max_fps = 0  # 0 means don't set FPS, capture at max speed
        
        self.cam = Camera(
            index=self.camera_info.index,
            fps=max_fps,  # Let camera run at maximum speed
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
        
        # Start frame grabber
        self._start_frame_grabber()
        
        # Get actual camera FPS (may be 0 if not set, meaning max speed)
        actual_fps = self.cam.cap.get(cv2.CAP_PROP_FPS) if self.cam.cap else 0
        if actual_fps == 0:
            print("[INFO] Camera set to capture at maximum speed (FPS not limited)")
        else:
            print(f"[INFO] Camera FPS: {actual_fps:.1f}")
        
        self.update_scale_info()
        self.status_label.config(text=f"Camera {self.camera_info.index} ready", foreground="green")
        print("[INFO] Camera opened successfully")
        print(f"[INFO] Using automatic exposure (default)")
        print(f"[INFO] Output video will be recorded at {self.output_fps} FPS (consistent)")
    
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
        self.scale_label.config(text=f"Scaled: {sw}×{sh}")
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
        max_errors = 50  # More lenient for slow cameras
        
        # Wait a bit for camera to stabilize
        time.sleep(0.5)
        
        while self.frame_grabber_running and self.cam and self.cam.is_open():
            frame = self.cam.read()
            if frame is not None:
                consecutive_errors = 0
                
                # Put frame in queue (drop old frame if queue is full)
                try:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    
                    self.frame_queue.put_nowait(frame)
                except Exception as e:
                    print(f"[WARN] Frame queue error: {e}")
            else:
                consecutive_errors += 1
                if consecutive_errors >= max_errors:
                    print(f"[WARN] Too many consecutive frame read errors ({consecutive_errors}), stopping frame grabber")
                    break
                # Longer sleep on error to avoid hammering the camera
                time.sleep(0.1)
    
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
        
        try:
            while self.preview_on and self.cam and self.cam.is_open():
                try:
                    # Get frame from queue
                    frame = self.frame_queue.get(timeout=0.5)
                    
                    if frame is not None:
                        # Validate frame
                        if len(frame.shape) >= 2 and frame.shape[0] > 0 and frame.shape[1] > 0:
                            # Convert to RGB if needed
                            if len(frame.shape) == 3 and frame.shape[2] == 3:
                                # Already BGR from OpenCV
                                display_frame = frame.copy()
                            else:
                                # Handle YUYV format
                                display_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
                            
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
                except Empty:
                    # No frame available, check if we should continue
                    if not self.frame_grabber_running:
                        print("[INFO] Preview: Frame grabber stopped, exiting preview")
                        break
                    # Still process window events
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
            # Use consistent output FPS (not camera FPS which may vary)
            output_path = make_capture_output_path(w, h, int(self.output_fps))
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # Use consistent output FPS for smooth playback
            self.video_writer = cv2.VideoWriter(output_path, fourcc, self.output_fps, (w, h))
            
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
        
        # Frame timing for consistent output FPS
        frame_interval = 1.0 / self.output_fps  # Time between frames at output FPS
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
        print(f"[INFO] Output video FPS: {self.output_fps:.1f} FPS (consistent)")
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
