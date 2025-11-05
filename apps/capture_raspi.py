#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi Camera Capture GUI
--------------------------------
Platform-specific implementation using BaseCaptureApp.
"""

import cv2
import os
import tkinter as tk
from typing import Tuple, Dict

from lib.capture.gui_base import BaseCaptureApp
from lib.capture import set_camera_control, get_camera_control_range
from lib.gui import get_standard_size, format_window_title


# Force OpenCV to prefer GTK backend over Qt (more reliable on Raspberry Pi)
os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_GTK', '1')
os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_MSMF', '0')


class CaptureApp(BaseCaptureApp):
    """Raspberry Pi UVC Camera GUI with full camera control."""
    
    def setup_environment(self) -> None:
        """Setup platform-specific environment variables."""
        # Already set at module level
        pass
    
    def get_camera_backend(self) -> int:
        """Get the camera backend constant for this platform."""
        return cv2.CAP_V4L2
    
    def get_control_specs(self) -> Dict[str, Tuple[str, int, int, int]]:
        """Get control specifications: {name: (label, min, max, default)}."""
        return {
            "brightness": ("Brightness", -64, 64, 0),
            "contrast": ("Contrast", 0, 64, 32),
            "saturation": ("Saturation", 0, 128, 60),
            "gain": ("Gain", 0, 100, 32),
        }
    
    def _build_camera_controls(self) -> None:
        """Build camera control sliders stacked vertically."""
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
            
            # Update callback using V4L2
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
    
    def _load_camera_control_ranges(self) -> None:
        """Load control ranges from camera using V4L2."""
        if not self.camera_info or not self.camera_info.device_path:
            return
        
        controls_to_check = ["brightness", "contrast", "saturation", "gain"]
        for ctrl_name in controls_to_check:
            range_info = get_camera_control_range(self.camera_info.device_path, ctrl_name)
            if range_info:
                self.control_ranges[ctrl_name] = range_info
                if ctrl_name in self.control_vars:
                    # Update slider range and value
                    for widget in self.param_frame.winfo_children():
                        if isinstance(widget, tk.Frame):
                            for child in widget.winfo_children():
                                if isinstance(child, tk.Scale):
                                    for w in widget.winfo_children():
                                        if isinstance(w, tk.ttk.Label):
                                            if ctrl_name.lower() in w.cget("text").lower():
                                                child.config(from_=range_info['min'], to=range_info['max'])
                                                self.control_vars[ctrl_name].set(range_info.get('default', 0))
                                                break
    
    def _apply_camera_controls(self) -> None:
        """Apply current control values to camera using V4L2."""
        if not self.camera_info or not self.camera_info.device_path:
            return
        
        for ctrl_name, var in self.control_vars.items():
            val = int(var.get())
            set_camera_control(self.camera_info.device_path, ctrl_name, val)
    
    def _reset_control_defaults(self) -> Dict[str, int]:
        """Get default control values for reset."""
        defaults = {
            "brightness": 0,
            "contrast": 32,
            "saturation": 60,
            "gain": 32
        }
        
        # Update with actual camera defaults if available
        for name in defaults.keys():
            if name in self.control_ranges:
                defaults[name] = self.control_ranges[name].get('default', defaults[name])
        
        return defaults
    
    def get_default_fps(self) -> float:
        """Get default FPS value for this platform."""
        return 30.0
    
    def get_resolution_for_format(self, format_str: str) -> Tuple[int, int]:
        """Get resolution (width, height) for a given format."""
        if format_str == "MJPG":
            return 1920, 1080
        else:
            return 640, 480
    
    def get_window_title(self) -> str:
        """Get window title for this platform."""
        return format_window_title("UVC Camera Capture", platform="Raspberry Pi")
    
    def get_window_size(self) -> Tuple[int, int]:
        """Get window size (width, height)."""
        return get_standard_size("capture")
    
    def _post_open_camera_setup(self) -> None:
        """Platform-specific setup after opening camera."""
        # Set power line frequency to 60 Hz automatically
        if self.camera_info and self.camera_info.device_path:
            set_camera_control(self.camera_info.device_path, "power_line_frequency", 2)
            print("[INFO] Power line frequency set to 60 Hz")


# ============ Main ============
def main():
    """Main entry point."""
    print(f"[INFO] UVC Camera Capture GUI starting (Raspberry Pi)...")
    
    root = tk.Tk()
    app = CaptureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.bind("<Escape>", lambda e: app.on_close())
    
    print("[INFO] UVC Camera Capture GUI ready")
    root.mainloop()


if __name__ == "__main__":
    main()
