#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows Camera Capture GUI
---------------------------
Lightweight platform-specific implementation for Windows
Heavy lifting done in base_capture_app.py
"""

import cv2
import os
import tkinter as tk
from tkinter import ttk, messagebox

from lib.camera import Camera
from lib.capture.base_capture_app import BaseCaptureApp


# ============ Debug Configuration ============
DEBUG_PREVIEW = True  # Set to False to disable detailed preview debug messages

# Force OpenCV to prefer MSMF backend on Windows (more reliable)
os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_MSMF', '1')
os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_DSHOW', '1')


# OpenCV property mappings for camera controls on Windows
OCV_PROP_BRIGHTNESS = cv2.CAP_PROP_BRIGHTNESS
OCV_PROP_CONTRAST = cv2.CAP_PROP_CONTRAST
OCV_PROP_SATURATION = cv2.CAP_PROP_SATURATION
OCV_PROP_GAIN = cv2.CAP_PROP_GAIN

# Control name to OpenCV property mapping
CONTROL_TO_PROP = {
    "brightness": OCV_PROP_BRIGHTNESS,
    "contrast": OCV_PROP_CONTRAST,
    "saturation": OCV_PROP_SATURATION,
    "gain": OCV_PROP_GAIN,
}


class CaptureApp(BaseCaptureApp):
    """Windows UVC Camera GUI - platform-specific implementation."""
    
    def __init__(self, root):
        # Default to YUYV for fastest performance on Windows (no decompression needed)
        super().__init__(root, title="UVC Camera Capture — Windows", default_format="YUYV")
        # Override FPS default to 0 (max speed) for Windows
        self.fps_var.set(0.0)
    
    def _build_camera_controls(self):
        """Build camera control sliders using OpenCV properties."""
        # Define controls with defaults for Windows (OpenCV properties)
        # Note: OpenCV property ranges vary by camera, so we use typical ranges
        controls = {
            "brightness": ("Brightness", 0, 255, 128),
            "contrast": ("Contrast", 0, 255, 128),
            "saturation": ("Saturation", 0, 255, 128),
            "gain": ("Gain", 0, 255, 64),
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
            
            # Update callback using OpenCV properties
            def update_control(ctrl_name=name, ctrl_var=var):
                if self.cam and self.cam.cap and self.cam.is_open():
                    val = float(ctrl_var.get())
                    prop = CONTROL_TO_PROP.get(ctrl_name)
                    if prop is not None:
                        try:
                            self.cam.cap.set(prop, val)
                            print(f"[DEBUG] {ctrl_name} = {val}")
                        except Exception as e:
                            print(f"[WARN] Failed to set {ctrl_name}: {e}")
            
            var.trace_add("write", lambda *a, update=update_control: update())
            entry.bind("<Return>", lambda e, update=update_control: update())
            
            self.control_vars[name] = var
    
    def open_camera(self):
        """Open the selected camera using MSMF backend."""
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
        
        # Determine resolution based on format - optimized for Windows speed
        # YUYV: Use 1280x720 for good quality and high FPS (no decompression overhead)
        # MJPEG: Use 1920x1080 if needed, but will be slower
        if format_str == "MJPG":
            width, height = 1920, 1080
        elif format_str == "YUYV":
            width, height = 1280, 720  # Good balance of quality and speed for YUYV
        else:
            width, height = 640, 480
        
        # Get framerate from input (0 = maximum speed, >0 = specific FPS)
        # For MJPG on Windows, explicitly setting 30 FPS often works better than 0
        try:
            fps_value = float(self.fps_var.get())
            # Clamp between 0 (max speed) and 60
            fps_value = max(0.0, min(60.0, fps_value))
            # For MJPG format, if user set 0, use 30 instead (better for USB cameras)
            if fps_value == 0 and format_str == "MJPG":
                fps_value = 30.0
            self.fps_var.set(fps_value)
        except (ValueError, tk.TclError):
            fps_value = 30.0  # Default to 30 FPS (better than 0 for MJPG)
            self.fps_var.set(fps_value)
        
        # Create camera - try DSHOW first (often faster than MSMF), fallback to MSMF
        self.cam = Camera(
            index=self.camera_info.index,
            fps=int(fps_value) if fps_value > 0 else 0,  # 0 = max speed, >0 = specific FPS
            fourcc=format_str,
            width=width,
            height=height,
            backend=cv2.CAP_DSHOW  # DSHOW often faster than MSMF for MJPEG on Windows
        )
        
        if not self.cam.open():
            messagebox.showerror("Camera Error", f"Failed to open camera {self.camera_info.index}")
            return
        
        # Finalize camera opening (common logic)
        self._finalize_camera_open(format_str)
    
    def _load_camera_control_ranges(self):
        """Load control ranges from camera using OpenCV properties."""
        if not self.cam or not self.cam.cap or not self.cam.is_open():
            return
        
        controls_to_check = ["brightness", "contrast", "saturation", "gain"]
        for ctrl_name in controls_to_check:
            prop = CONTROL_TO_PROP.get(ctrl_name)
            if prop is None:
                continue
            
            try:
                # Try to get current value (some cameras may not support getting)
                current = self.cam.cap.get(prop)
                # If we get a valid value, assume the property is supported
                if current is not None and current >= 0:
                    if ctrl_name in self.control_vars:
                        var = self.control_vars[ctrl_name]
                        # Set current value from camera
                        var.set(current)
            except Exception as e:
                print(f"[WARN] Could not get {ctrl_name} from camera: {e}")
    
    def _apply_camera_controls(self):
        """Apply current control values to camera using OpenCV properties."""
        if not self.cam or not self.cam.cap or not self.cam.is_open():
            return
        
        for ctrl_name, var in self.control_vars.items():
            val = float(var.get())
            prop = CONTROL_TO_PROP.get(ctrl_name)
            if prop is not None:
                try:
                    self.cam.cap.set(prop, val)
                except Exception as e:
                    print(f"[WARN] Failed to set {ctrl_name}: {e}")
    
    def reset_controls(self):
        """Reset camera controls to defaults (Windows)."""
        if not self.cam or not self.camera_info:
            return
        
        defaults = {
            "brightness": 128,
            "contrast": 128,
            "saturation": 128,
            "gain": 64
        }
        
        for name, default_val in defaults.items():
            if name in self.control_vars:
                self.control_vars[name].set(default_val)
                prop = CONTROL_TO_PROP.get(name)
                if prop is not None and self.cam.cap:
                    try:
                        self.cam.cap.set(prop, default_val)
                    except Exception as e:
                        print(f"[WARN] Failed to reset {name}: {e}")
        
        print("[INFO] Controls reset to defaults")


# ============ Main ============
def main():
    """Main entry point."""
    print(f"[INFO] UVC Camera Capture GUI starting (Windows)...")
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
