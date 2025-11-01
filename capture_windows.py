#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows Camera Capture GUI
---------------------------
Standalone implementation for Windows
"""

import cv2
import os
import tkinter as tk
from tkinter import ttk, messagebox
from queue import Queue, Empty

from lib.camera import Camera
from lib.camera_info import list_all_cameras
from lib.capture import PreviewManager, FrameGrabber, RecordingManager, make_capture_output_path, make_capture_frame_path


# ============ Debug Configuration ============
DEBUG_PREVIEW = True

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


class CaptureApp:
    """Windows UVC Camera GUI."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("UVC Camera Capture — Windows")
        
        # Camera state
        self.cam = None
        self.camera_info = None
        self.available_cameras = []
        
        # Format and resolution
        self.format_var = tk.StringVar(value="YUYV")
        self.scale_percent = tk.DoubleVar(value=100.0)
        self.fps_var = tk.DoubleVar(value=0.0)  # Default to max speed on Windows
        
        # Frame queue
        self.frame_queue = Queue(maxsize=10)
        
        # Initialize managers
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
        self.control_ranges = {}
        
        # Build UI
        self._build_ui()
        
        # Detect cameras on startup
        self.refresh_camera_list()
        
        # Start preview window immediately
        self.root.after(100, self.preview_manager.start)
    
    def _build_ui(self):
        """Build the Tkinter GUI."""
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
        
        self.open_camera_btn = ttk.Button(status_frame, text="Open Camera", command=self.toggle_camera)
        self.open_camera_btn.pack(fill=tk.X, pady=(4, 0))
        
        # Camera controls
        self.param_frame = ttk.LabelFrame(ctrl, text="Controls", padding="3")
        self.param_frame.pack(fill=tk.X, pady=(0, 4))
        self._build_camera_controls()
        
        # Actions section
        btn_frame = ttk.LabelFrame(ctrl, text="Actions", padding="3")
        btn_frame.pack(fill=tk.X, pady=(0, 4))
        
        self.record_btn = ttk.Button(btn_frame, text="Record", command=self.toggle_record)
        self.record_btn.pack(fill=tk.X, pady=(0, 2))
        
        ttk.Button(btn_frame, text="Capture Frame", command=self.capture_frame).pack(fill=tk.X, pady=(0, 2))
        ttk.Button(btn_frame, text="Reset Controls", command=self.reset_controls).pack(fill=tk.X, pady=(0, 2))
        ttk.Button(btn_frame, text="Exit", command=self.on_close).pack(fill=tk.X)
    
    def _build_camera_controls(self):
        """Build camera control sliders using OpenCV properties."""
        controls = {
            "brightness": ("Brightness", 0, 255, 128),
            "contrast": ("Contrast", 0, 255, 128),
            "saturation": ("Saturation", 0, 255, 128),
            "gain": ("Gain", 0, 255, 64),
        }
        
        self.control_vars = {}
        
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
    
    def refresh_camera_list(self):
        """Refresh the list of available cameras."""
        print("[INFO] Refreshing camera list...")
        try:
            self.available_cameras = list_all_cameras(max_test=10, detailed=False, test_resolutions=False)
        except Exception as e:
            print(f"[ERROR] Failed to refresh camera list: {e}")
            self.available_cameras = []
        
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
        
        for cam_info in self.available_cameras:
            name_str = f"Camera {cam_info.index}"
            if cam_info.name:
                name_str += f" ({cam_info.name})"
            if name_str == selection:
                self.camera_info = cam_info
                self._update_camera_info_display()
                if cam_info.supported_fourcc and hasattr(self, 'format_combo'):
                    self.format_combo['values'] = cam_info.supported_fourcc
                    if cam_info.supported_fourcc:
                        default_format = cam_info.default_fourcc if cam_info.default_fourcc in cam_info.supported_fourcc else cam_info.supported_fourcc[0]
                        self.format_var.set(default_format)
                        try:
                            idx = cam_info.supported_fourcc.index(default_format)
                            self.format_combo.current(idx)
                        except ValueError:
                            self.format_combo.current(0)
                break
    
    def on_format_changed(self, event=None):
        """Handle format selection change."""
        if self.cam and self.cam.is_open():
            print("[INFO] Format changed, reopening camera...")
            self.close_camera()
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
        """Open the selected camera using DSHOW backend."""
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
        
        # Determine resolution
        if format_str == "MJPG":
            width, height = 1920, 1080
        elif format_str == "YUYV":
            width, height = 1280, 720
        else:
            width, height = 640, 480
        
        # Get framerate
        try:
            fps_value = float(self.fps_var.get())
            fps_value = max(0.0, min(60.0, fps_value))
            if fps_value == 0 and format_str == "MJPG":
                fps_value = 30.0
            self.fps_var.set(fps_value)
        except (ValueError, tk.TclError):
            fps_value = 30.0
            self.fps_var.set(fps_value)
        
        # Create camera
        self.cam = Camera(
            index=self.camera_info.index,
            fps=int(fps_value) if fps_value > 0 else 0,
            fourcc=format_str,
            width=width,
            height=height,
            backend=cv2.CAP_DSHOW
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
        self._load_camera_control_ranges()
        # Apply current control values
        self._apply_camera_controls()
        
        # Get actual camera FPS
        actual_fps = self.cam.cap.get(cv2.CAP_PROP_FPS) if self.cam.cap else 0
        
        # Update FPS display
        if actual_fps == 0:
            self.fps_label.config(text="FPS: Max speed")
            print("[INFO] Camera set to capture at maximum speed (FPS not limited)")
            output_fps = 30.0
        else:
            self.fps_label.config(text=f"FPS: {actual_fps:.1f}")
            print(f"[INFO] Camera FPS: {actual_fps:.1f}")
            output_fps = actual_fps
        
        # Start frame grabber
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
    
    def close_camera(self):
        """Close the camera."""
        self.recording_manager.stop()
        self.frame_grabber.stop()
        
        if self.cam:
            self.cam.release()
            self.cam = None
        
        self.open_camera_btn.config(text="Open Camera")
        self.status_label.config(text="Camera closed", foreground="black")
        print("[INFO] Camera closed (preview window remains open)")
    
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
                current = self.cam.cap.get(prop)
                if current is not None and current >= 0:
                    if ctrl_name in self.control_vars:
                        var = self.control_vars[ctrl_name]
                        var.set(current)
            except Exception as e:
                print(f"[WARN] Could not get {ctrl_name} from camera: {e}")
    
    def _apply_camera_controls(self):
        """Apply current control values to camera."""
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
    
    def update_scale_info(self):
        """Update scale information display."""
        if not self.cam:
            return
        
        p = max(1, min(100, float(self.scale_percent.get())))
        sw, sh = int(self.cam.w * p / 100), int(self.cam.h * p / 100)
        self.scale_label.config(text=f"{sw}×{sh}")
        print(f"[INFO] {self.format_var.get()} {self.cam.w}×{self.cam.h} → {sw}×{sh} ({p:.1f}%)")
    
    def reset_controls(self):
        """Reset camera controls to defaults."""
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
    
    def capture_frame(self):
        """Capture a single frame."""
        if not self.cam or not self.cam.is_open():
            messagebox.showinfo("Camera", "Open camera first.")
            return
        
        try:
            frame = self.frame_queue.get(timeout=1.0)
        except Empty:
            messagebox.showerror("Capture Error", "Failed to read frame from camera (timeout).")
            return
        
        if frame is None:
            messagebox.showerror("Capture Error", "Failed to read frame from camera.")
            return
        
        w, h = self.scaled_size()
        if w != self.cam.w or h != self.cam.h:
            frame_resized = cv2.resize(frame, (w, h))
        else:
            frame_resized = frame
        
        if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
            frame_bgr = frame_resized
        else:
            frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_YUV2BGR_YUY2)
        
        output_path = make_capture_frame_path(w, h)
        cv2.imwrite(output_path, frame_bgr)
        print(f"[INFO] Saved {output_path}")
        self.status_label.config(text=f"Saved: {os.path.basename(output_path)}", foreground="blue")
    
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
