#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi Camera Capture GUI
--------------------------------
Lightweight platform-specific implementation for Raspberry Pi
Heavy lifting done in base_capture_app.py
"""

import os

# IMPORTANT: Set environment variables BEFORE importing cv2 or any module that imports cv2
# Force OpenCV to prefer GTK backend over Qt (more reliable on Raspberry Pi)
os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_GTK', '1')
os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_MSMF', '0')

import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from queue import Empty

from lib.camera import Camera
from lib.camera_info import set_camera_control, get_camera_control_range
from lib.capture.base_capture_app import BaseCaptureApp


# ============ Debug Configuration ============
DEBUG_PREVIEW = True  # Set to False to disable detailed preview debug messages


class CaptureApp(BaseCaptureApp):
    """Raspberry Pi UVC Camera GUI - platform-specific implementation."""
    
    def __init__(self, root):
        super().__init__(root, title="UVC Camera Capture — Raspberry Pi", default_format="YUYV")
    
    def _build_camera_controls(self):
        """Build camera control sliders using V4L2 controls."""
        # Define controls with defaults for Raspberry Pi (V4L2)
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
    
    def open_camera(self):
        """Open the selected camera using V4L2 backend."""
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
        
        # Create camera with V4L2 backend
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
        print("[INFO] Using automatic exposure (default)")
        print("[INFO] Power line frequency set to 60 Hz")
        print(f"[INFO] Output video will be recorded at {output_fps:.1f} FPS")
        print(f"[INFO] Format: {format_str}, Resolution: {self.cam.w}x{self.cam.h}")
        if format_str == "MJPG":
            print("[INFO] Note: MJPG at 1920x1080 may require more processing time")
    
    def _load_camera_control_ranges(self):
        """Load control ranges from camera using V4L2."""
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
                                    for w in widget.winfo_children():
                                        if isinstance(w, ttk.Label):
                                            if ctrl_name.lower() in w.cget("text").lower():
                                                child.config(from_=range_info['min'], to=range_info['max'])
                                                self.control_vars[ctrl_name].set(range_info.get('default', 0))
                                                break
    
    def _apply_camera_controls(self):
        """Apply current control values to camera using V4L2."""
        if not self.camera_info or not self.camera_info.device_path:
            return
        
        for ctrl_name, var in self.control_vars.items():
            val = int(var.get())
            set_camera_control(self.camera_info.device_path, ctrl_name, val)
    
    def reset_controls(self):
        """Reset camera controls to defaults (Raspberry Pi)."""
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


# ============ Main ============
def main():
    """Main entry point."""
    print(f"[INFO] UVC Camera Capture GUI starting (Raspberry Pi)...")
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
