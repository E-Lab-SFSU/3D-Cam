#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Capture App - Modular base class for platform-specific capture applications
"""

import cv2
import time
from queue import Queue, Empty
import tkinter as tk
from tkinter import ttk, messagebox

from lib.capture import PreviewManager, FrameGrabber, RecordingManager, make_capture_output_path, make_capture_frame_path


class BaseCaptureApp:
    """Base class for camera capture applications - platform-specific implementations inherit from this."""
    
    def __init__(self, root, title="Camera Capture"):
        self.root = root
        self.root.title(title)
        
        # Camera state (to be set by platform-specific implementation)
        self.cam = None
        
        # Format and resolution
        self.format_var = tk.StringVar(value="YUYV")
        self.scale_percent = tk.DoubleVar(value=100.0)
        self.fps_var = tk.DoubleVar(value=30.0)
        
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
        
        # Build UI
        self._build_ui()
        
        # Start preview window immediately (always on)
        self.root.after(100, self.preview_manager.start)
    
    def _build_ui(self):
        """Build the common Tkinter GUI elements."""
        main = ttk.Frame(self.root, padding="5")
        main.pack(fill=tk.BOTH, expand=True)

        # Right side: Controls
        ctrl = ttk.Frame(main)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=3, pady=3)
        
        # Platform-specific UI elements (override in subclass)
        self._build_platform_ui(ctrl)
        
        # Status section (common)
        status_frame = ttk.LabelFrame(ctrl, text="Status", padding="3")
        status_frame.pack(fill=tk.X, pady=(0, 4))
        
        self.status_label = ttk.Label(status_frame, text="Ready", font=("TkDefaultFont", 9))
        self.status_label.pack(anchor="w", pady=(0, 2))
        
        self.fps_label = ttk.Label(status_frame, text="FPS: —", font=("TkDefaultFont", 8))
        self.fps_label.pack(anchor="w")
        
        # Open/Close Camera button (common)
        self.open_camera_btn = ttk.Button(status_frame, text="Open Camera", command=self.toggle_camera)
        self.open_camera_btn.pack(fill=tk.X, pady=(4, 0))
        
        # Platform-specific controls (override in subclass)
        self._build_platform_controls(ctrl)
        
        # Actions section (common)
        btn_frame = ttk.LabelFrame(ctrl, text="Actions", padding="3")
        btn_frame.pack(fill=tk.X, pady=(0, 4))
        
        # Record/Stop Record button
        self.record_btn = ttk.Button(btn_frame, text="Record", command=self.toggle_record)
        self.record_btn.pack(fill=tk.X, pady=(0, 2))
        
        # Capture Frame button
        ttk.Button(btn_frame, text="Capture Frame", command=self.capture_frame).pack(fill=tk.X, pady=(0, 2))
    
    def _build_platform_ui(self, parent):
        """Override in platform-specific implementation to add camera selection, format, etc."""
        pass
    
    def _build_platform_controls(self, parent):
        """Override in platform-specific implementation to add camera controls."""
        pass
    
    def scaled_size(self):
        """Get scaled output size."""
        if not self.cam:
            return 640, 480
        p = max(1, min(100, float(self.scale_percent.get())))
        return int(self.cam.w * p / 100), int(self.cam.h * p / 100)
    
    def open_camera(self):
        """Platform-specific camera opening - must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement open_camera()")
    
    def close_camera(self):
        """Close the camera - common implementation."""
        # Stop recording
        self.recording_manager.stop()
        
        # Stop frame grabber
        self.frame_grabber.stop()
        
        if self.cam:
            self.cam.release()
            self.cam = None
        
        self.open_camera_btn.config(text="Open Camera")
        self.status_label.config(text="Camera closed", foreground="black")
        print("[INFO] Camera closed (preview window remains open)")
    
    def toggle_camera(self):
        """Toggle camera open/close."""
        if self.cam and self.cam.is_open():
            self.close_camera()
        else:
            self.open_camera()
    
    def capture_frame(self):
        """Capture a single frame - common implementation."""
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
        
        # Save to capture_output directory
        import os
        output_path = make_capture_frame_path(w, h)
        cv2.imwrite(output_path, frame_bgr)
        print(f"[INFO] Saved {output_path}")
        self.status_label.config(text=f"Saved: {os.path.basename(output_path)}", foreground="blue")
    
    def toggle_record(self):
        """Toggle video recording - common implementation."""
        if not self.cam or not self.cam.is_open():
            messagebox.showinfo("Camera", "Open camera first.")
            return
        
        if self.recording_manager.recording:
            self.recording_manager.stop()
        else:
            self.recording_manager.start()
    
    def on_close(self):
        """Handle application close - common implementation."""
        print("[INFO] Closing application")
        self.preview_manager.stop()
        self.recording_manager.stop()
        self.frame_grabber.stop()
        
        if self.cam:
            self.cam.release()
        
        cv2.destroyAllWindows()
        self.root.destroy()

