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
from lib.capture import PreviewManager, FrameGrabber, RecordingManager, make_capture_output_path


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
        self.preview_manager.stop()
        
        # Stop recording
        self.recording_manager.stop()
        
        # Stop frame grabber
        self.frame_grabber.stop()
        
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
        self.frame_grabber.stop()
        
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
        print(f"[INFO] Using automatic exposure (default)")
        print(f"[INFO] Power line frequency set to 60 Hz")
        print(f"[INFO] Output video will be recorded at {output_fps:.1f} FPS")
        print(f"[INFO] Format: {format_str}, Resolution: {self.cam.w}x{self.cam.h}")
        if format_str == "MJPG":
            print("[INFO] Note: MJPG at 1920x1080 may require more processing time")
        
        # Automatically start preview when camera opens
        debug_print("Auto-starting preview after camera opened...")
        self.root.after(100, self.preview_manager.start)  # Small delay to ensure camera is ready
    
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
        
        if self.recording_manager.recording:
            self.recording_manager.stop()
        else:
            self.recording_manager.start()
    

    def on_close(self):
        """Handle application close."""
        print("[INFO] Closing application")
        self.preview_manager.stop()
        self.recording_manager.stop()
        self.frame_grabber.stop()
        
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
