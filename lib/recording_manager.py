#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recording Manager - Handles video recording functionality
"""

import cv2
import time
import threading
import os
from queue import Empty

from lib.util_paths import make_capture_output_path


class RecordingManager:
    """Manages video recording."""
    
    def __init__(self, frame_queue, get_camera_fn, scaled_size_fn, status_callback, button_callback):
        """
        Initialize recording manager.
        
        Args:
            frame_queue: Queue to get frames from
            get_camera_fn: Callable to get camera: () -> Camera or None
            scaled_size_fn: Callable to get scaled size: () -> (width, height)
            status_callback: Callable to update status: (text, color) -> None
            button_callback: Callable to update button: (text) -> None
        """
        self.frame_queue = frame_queue
        self.get_camera = get_camera_fn
        self.scaled_size = scaled_size_fn
        self.update_status = status_callback
        self.update_button = button_callback
        
        self.recording = False
        self.stop_flag = False
        self.video_writer = None
        self.record_thread = None
    
    def start(self):
        """Start video recording."""
        camera = self.get_camera()
        if not camera or not camera.is_open():
            from tkinter import messagebox
            messagebox.showinfo("Camera", "Open camera first.")
            return
        
        if self.recording:
            return
        
        w, h = self.scaled_size()
        
        actual_fps = camera.cap.get(cv2.CAP_PROP_FPS) if camera.cap else 0
        output_fps = actual_fps if actual_fps > 0 else 30.0
        
        output_path = make_capture_output_path(w, h, int(output_fps))
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))
        
        if not self.video_writer.isOpened():
            from tkinter import messagebox
            messagebox.showerror("Recording Error", f"Failed to create video file: {output_path}")
            return
        
        self.stop_flag = False
        self.recording = True
        self.update_status(f"Recording: {os.path.basename(output_path)}", "red")
        
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()
        
        self.update_button("Stop Record")
        
        print(f"[INFO] Recording started: {output_path}")
    
    def stop(self):
        """Stop video recording."""
        if not self.recording:
            return
        
        self.stop_flag = True
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=2.0)
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.recording = False
        self.update_button("Record")
        self.update_status("Recording stopped", "black")
        print("[INFO] Recording stopped")
    
    def _record_loop(self):
        """Recording thread loop with consistent output FPS."""
        camera = self.get_camera()
        if not camera:
            return
        
        frame_count = 0
        dropped_frames = 0
        skipped_frames = 0
        t0 = time.time()
        
        actual_fps = camera.cap.get(cv2.CAP_PROP_FPS) if camera.cap else 0
        output_fps = actual_fps if actual_fps > 0 else 30.0
        
        frame_interval = 1.0 / output_fps
        next_frame_time = t0
        last_frame = None
        
        while not self.stop_flag and camera and camera.is_open():
            try:
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except Empty:
                    current_time = time.time()
                    if current_time >= next_frame_time and last_frame is not None:
                        w, h = self.scaled_size()
                        if w != camera.w or h != camera.h:
                            frame_resized = cv2.resize(last_frame, (w, h))
                        else:
                            frame_resized = last_frame.copy()
                        
                        if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
                            frame_bgr = frame_resized
                        else:
                            frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_YUV2BGR_YUY2)
                        
                        self.video_writer.write(frame_bgr)
                        frame_count += 1
                        next_frame_time += frame_interval
                    continue
                
                if frame is not None:
                    last_frame = frame
                    
                    w, h = self.scaled_size()
                    if w != camera.w or h != camera.h:
                        frame_resized = cv2.resize(frame, (w, h))
                    else:
                        frame_resized = frame
                    
                    if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
                        frame_bgr = frame_resized
                    else:
                        frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_YUV2BGR_YUY2)
                    
                    current_time = time.time()
                    
                    if current_time >= next_frame_time:
                        self.video_writer.write(frame_bgr)
                        frame_count += 1
                        next_frame_time += frame_interval
                        
                        if current_time > next_frame_time + frame_interval * 2:
                            next_frame_time = current_time + frame_interval
                    else:
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

