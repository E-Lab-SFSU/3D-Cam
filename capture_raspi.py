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
from PIL import Image, ImageTk

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
        self.last_time = time.time()
        self.fps_est = 0.0
        
        # Recording state
        self.recording = False
        self.video_writer = None
        
        # Camera controls
        self.control_vars = {}
        self.control_ranges = {}  # Store ranges for each control
        
        # Build UI
        self._build_ui()
        
        # Detect cameras on startup
        self.refresh_camera_list()
    
    def _build_ui(self):
        """Build the Tkinter GUI."""
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Left side: Preview canvas
        self.canvas = tk.Canvas(main, bg="black")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
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
        
        self.cam = Camera(
            index=self.camera_info.index,
            fps=int(self.camera_info.default_fps),
            fourcc=format_str,
            width=width,
            height=height,
            backend=cv2.CAP_V4L2
        )
        
        if not self.cam.open():
            messagebox.showerror("Camera Error", f"Failed to open camera {self.camera_info.index}")
            return
        
        # Load camera control ranges
        if self.camera_info.device_path:
            self._load_camera_control_ranges()
            # Apply current control values
            self._apply_camera_controls()
        
        self.update_scale_info()
        self.status_label.config(text=f"Camera {self.camera_info.index} ready", foreground="green")
        print("[INFO] Camera opened successfully")
        print(f"[INFO] Using automatic exposure (default)")
    
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
    
    def start_preview(self):
        """Start live preview."""
        if not self.cam:
            self.open_camera()
            if not self.cam:
                return
        
        if not self.preview_on:
            print("[INFO] Starting preview")
            self.preview_on = True
            self.last_time = time.time()
            self.fps_est = 0.0
            self._update_frame()
    
    def stop_preview(self):
        """Stop live preview."""
        if self.preview_on:
            print("[INFO] Stopping preview")
        self.preview_on = False
    
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
        if not self.cam:
            messagebox.showinfo("Camera", "Open camera first.")
            return
        
        frame = self.cam.read()
        if frame is None:
            messagebox.showerror("Capture Error", "Failed to read frame from camera.")
            return
        
        w, h = self.scaled_size()
        frame_resized = cv2.resize(frame, (w, h))
        
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = f"frame_{w}x{h}_{ts}.png"
        cv2.imwrite(name, frame_resized)
        print(f"[INFO] Saved {name}")
        self.status_label.config(text=f"Saved: {name}", foreground="blue")
    
    def toggle_record(self):
        """Toggle video recording."""
        if not self.cam:
            messagebox.showinfo("Camera", "Open camera first.")
            return
        
        if self.video_writer:
            # Stop recording
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            self.status_label.config(text="Recording stopped", foreground="black")
            print("[INFO] Recording stopped")
        else:
            # Start recording
            ts = time.strftime("%Y%m%d_%H%M%S")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w, h = self.scaled_size()
            name = f"video_{w}x{h}_{ts}.mp4"
            
            self.video_writer = cv2.VideoWriter(name, fourcc, self.cam.fps, (w, h))
            if not self.video_writer.isOpened():
                messagebox.showerror("Recording Error", f"Failed to create video file: {name}")
                return
            
            self.recording = True
            self.status_label.config(text=f"Recording: {name}", foreground="red")
            print(f"[INFO] Recording started: {name}")
    
    def _update_frame(self):
        """Update preview frame."""
        if not self.preview_on or not self.cam:
            return
        
        frame = self.cam.read()
        if frame is not None:
            # Convert to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Handle YUYV format
                rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_YUY2)
            
            # Scale
            w, h = self.scaled_size()
            rgb_resized = cv2.resize(rgb, (w, h))
            
            # FPS estimation
            now = time.time()
            dt = now - self.last_time
            self.last_time = now
            if dt > 0:
                self.fps_est = 0.9 * self.fps_est + 0.1 * (1.0 / dt)
            
            # Add overlay text
            rgb_preview = rgb_resized.copy()
            cv2.putText(
                rgb_preview,
                f"{self.format_var.get()} {self.cam.w}x{self.cam.h}  FPS:{self.fps_est:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
            if self.recording:
                cv2.putText(
                    rgb_preview,
                    "REC",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
            
            # Update canvas
            im = Image.fromarray(rgb_preview)
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            if cw > 1 and ch > 1:
                scale = min(cw / w, ch / h)
                im = im.resize((int(w * scale), int(h * scale)), Image.NEAREST)
                imgtk = ImageTk.PhotoImage(image=im)
                self.canvas.delete("all")
                self.canvas.create_image(cw // 2, ch // 2, image=imgtk)
                self.canvas.image = imgtk
            
            # Save frame for recording
            if self.video_writer:
                self.video_writer.write(cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR))
        
        # Schedule next update
        fps = self.cam.fps if self.cam.fps else 30
        delay = int(1000 / fps)
        self.root.after(delay, self._update_frame)
    
    def on_close(self):
        """Handle application close."""
        print("[INFO] Closing application")
        self.stop_preview()
        if self.video_writer:
            self.video_writer.release()
        if self.cam:
            self.cam.release()
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
