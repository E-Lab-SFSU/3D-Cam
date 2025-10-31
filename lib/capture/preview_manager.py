#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preview Manager - Handles OpenCV preview window for camera capture
"""

import cv2
import time
import threading
import os
from queue import Queue, Empty
import traceback
import numpy as np


# Debug configuration
DEBUG_PREVIEW = True


def debug_print(message, force=False):
    """Print debug message if DEBUG_PREVIEW is enabled."""
    if DEBUG_PREVIEW or force:
        print(f"[DEBUG] {message}")


class PreviewManager:
    """Manages preview window lifecycle and frame display."""
    
    def __init__(self, root, frame_queue, camera_var, format_var, recording_var):
        """
        Initialize preview manager.
        
        Args:
            root: Tkinter root window
            frame_queue: Queue to get frames from
            camera_var: Callable to check if camera is open: () -> (cam, is_open)
            format_var: Callable to get current format: () -> str
            recording_var: Callable to check if recording: () -> bool
        """
        self.root = root
        self.frame_queue = frame_queue
        self.get_camera = camera_var
        self.get_format = format_var
        self.is_recording = recording_var
        
        # Preview state
        self.preview_on = False
        self.preview_thread = None
    
    def start(self):
        """Start preview window."""
        cam, is_open = self.get_camera()
        if not cam or not is_open:
            print("[WARN] Camera not open, cannot start preview")
            return
        
        # Always stop previous preview if running (ensures clean restart)
        if self.preview_on or (self.preview_thread and self.preview_thread.is_alive()):
            self.stop()
            time.sleep(0.2)  # Give time for cleanup
        
        # Destroy any existing window to ensure clean start
        try:
            cv2.destroyWindow("Preview")
            cv2.waitKey(1)
        except:
            pass
        
        self.preview_on = True
        print("[INFO] Starting preview")
        
        # Start preview thread (creates window in thread like capture.py)
        try:
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()
        except Exception as e:
            print(f"[ERROR] Failed to start preview thread: {e}")
            traceback.print_exc()
            self.preview_on = False
            self.preview_thread = None
    
    def stop(self):
        """Stop preview window."""
        if not self.preview_on and (not self.preview_thread or not self.preview_thread.is_alive()):
            return
        
        print("[INFO] Stopping preview")
        self.preview_on = False
        
        # Wait for thread to exit
        if self.preview_thread and self.preview_thread.is_alive():
            self.preview_thread.join(timeout=2.0)
        
        self.preview_thread = None
        
        # Ensure window is destroyed
        try:
            cv2.destroyWindow("Preview")
            cv2.waitKey(1)  # Process window events
        except:
            pass
    
    def _preview_loop(self):
        """Preview thread loop - simple like capture.py."""
        # Create window in preview thread (simpler approach)
        window_name = "Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        last_time = time.time()
        frame_count = 0
        fps = 0.0
        
        print("[INFO] Preview window created")
        
        try:
            while self.preview_on:
                cam, is_open = self.get_camera()
                if not cam or not is_open:
                    print("[INFO] Preview: Camera closed")
                    break
                
                try:
                    # Get frame from queue (with timeout)
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    if frame is not None and len(frame.shape) >= 2:
                        # Calculate FPS
                        frame_count += 1
                        now = time.time()
                        if frame_count % 10 == 0:
                            elapsed = now - last_time
                            fps = 10.0 / elapsed if elapsed > 0 else 0
                            last_time = now
                        
                        # Draw FPS overlay (copy frame first to avoid modifying original)
                        if frame_count == 1:
                            print(f"[INFO] Preview: First frame! Shape: {frame.shape}")
                        
                        display_frame = frame.copy()
                        format_str = self.get_format()
                        cam, _ = self.get_camera()
                        if cam:
                            cv2.putText(display_frame, f"{format_str} {cam.w}x{cam.h}  FPS:{fps:.1f}",
                                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        if self.is_recording():
                            cv2.putText(display_frame, "REC", (10, 55),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        cv2.imshow(window_name, display_frame)
                    
                except Empty:
                    # No frame available, continue
                    pass
                
                # Process window events and check for ESC key
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    self.preview_on = False
                    break
                    
        except Exception as e:
            print(f"[ERROR] Preview loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.preview_on = False
            try:
                cv2.destroyWindow(window_name)
            except:
                pass
            print(f"[INFO] Preview stopped. Total frames: {frame_count}")

