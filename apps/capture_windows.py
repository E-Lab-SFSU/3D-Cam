#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows Camera Capture GUI
---------------------------
Platform-specific implementation using BaseCaptureApp.
"""

import cv2
import os
import tkinter as tk
from typing import Tuple, Dict

from lib.capture.gui_base import BaseCaptureApp
from lib.gui import get_standard_size, format_window_title


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
    """Windows UVC Camera GUI."""
    
    def setup_environment(self) -> None:
        """Setup platform-specific environment variables."""
        # Already set at module level
        pass
    
    def get_camera_backend(self) -> int:
        """Get the camera backend constant for this platform."""
        return cv2.CAP_DSHOW
    
    def get_control_specs(self) -> Dict[str, Tuple[str, int, int, int]]:
        """Get control specifications: {name: (label, min, max, default)}."""
        return {
            "brightness": ("Brightness", 0, 255, 128),
            "contrast": ("Contrast", 0, 255, 128),
            "saturation": ("Saturation", 0, 255, 128),
            "gain": ("Gain", 0, 255, 64),
        }
    
    def _build_camera_controls(self) -> None:
        """Build camera control sliders using OpenCV properties."""
        from tkinter import ttk
        
        controls = self.get_control_specs()
        
        for name, (label, default_min, default_max, default_val) in controls.items():
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
    
    def _load_camera_control_ranges(self) -> None:
        """Load control ranges from camera using OpenCV properties."""
        if not self.cam or not self.cam.cap or not self.cam.is_open():
            return
        
        controls_to_check = ["brightness", "contrast", "saturation", "gain"]
        for ctrl_name in controls_to_check:
            prop = CONTROL_TO_PROP.get(ctrl_name)
            if prop is None:
                continue
            
            try:
                current = self.cam.cap.get(prop)
                if current is not None and current >= 0:
                    if ctrl_name in self.control_vars:
                        var = self.control_vars[ctrl_name]
                        var.set(current)
            except Exception as e:
                print(f"[WARN] Could not get {ctrl_name} from camera: {e}")
    
    def _apply_camera_controls(self) -> None:
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
    
    def _reset_control_defaults(self) -> Dict[str, int]:
        """Get default control values for reset."""
        return {
            "brightness": 128,
            "contrast": 128,
            "saturation": 128,
            "gain": 64
        }
    
    def get_default_fps(self) -> float:
        """Get default FPS value for this platform."""
        return 0.0  # Default to max speed on Windows
    
    def get_resolution_for_format(self, format_str: str) -> Tuple[int, int]:
        """Get resolution (width, height) for a given format."""
        if format_str == "MJPG":
            return 1920, 1080
        elif format_str == "YUYV":
            return 1280, 720
        else:
            return 640, 480
    
    def get_window_title(self) -> str:
        """Get window title for this platform."""
        return format_window_title("UVC Camera Capture", platform="Windows")
    
    def get_window_size(self) -> Tuple[int, int]:
        """Get window size (width, height)."""
        return get_standard_size("capture")
    
    def _post_open_camera_setup(self) -> None:
        """Platform-specific setup after opening camera."""
        # For Windows, ensure MJPG gets a reasonable FPS if set to 0
        format_str = self.format_var.get()
        if format_str == "MJPG":
            try:
                fps_value = float(self.fps_var.get())
                if fps_value == 0:
                    self.fps_var.set(30.0)
            except (ValueError, tk.TclError):
                pass


# ============ Main ============
def main():
    """Main entry point."""
    print(f"[INFO] UVC Camera Capture GUI starting (Windows)...")
    
    root = tk.Tk()
    app = CaptureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.bind("<Escape>", lambda e: app.on_close())
    
    print("[INFO] UVC Camera Capture GUI ready")
    root.mainloop()


if __name__ == "__main__":
    main()
