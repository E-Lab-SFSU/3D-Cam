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
        self.preview_window_name = "Preview"
        self.preview_window_counter = 0
        self.preview_window_name_current = None
        self.opencv_event_processing_active = False
        
        # Debug state tracking
        self.preview_start_count = 0
        self.preview_stop_count = 0
    
    def start(self):
        """Start preview window."""
        self.preview_start_count += 1
        debug_print(f"=== start_preview() called (count: {self.preview_start_count}) ===")
        
        # Check if preview is already running with a valid window
        if self.preview_on and self.preview_window_name_current:
            try:
                window_visible = cv2.getWindowProperty(self.preview_window_name_current, cv2.WND_PROP_VISIBLE)
                if window_visible >= 0:
                    debug_print("Preview already running with valid window, skipping start")
                    return
            except:
                debug_print("Preview flag set but window doesn't exist, recreating...")
                self.preview_on = False
                self.preview_window_name_current = None
        
        cam, is_open = self.get_camera()
        if not cam or not is_open:
            debug_print("Camera not open, cannot start preview")
            return
        
        # Ensure previous thread is fully stopped
        if self.preview_thread is not None and self.preview_thread.is_alive():
            debug_print("Previous preview thread still alive, stopping it first...")
            self.preview_on = False
            self.preview_thread.join(timeout=0.5)
            self.preview_thread = None
        
        self.preview_on = False
        debug_print("Flag set to False before cleanup")
        
        # Clean up any existing window and reset OpenCV state
        debug_print("Starting window cleanup in start_preview() (main thread)...")
        try:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(10)
            debug_print("destroyAllWindows() succeeded, processed events")
        except Exception as e:
            debug_print(f"Exception in destroyAllWindows: {type(e).__name__}: {e}")
        
        time.sleep(0.15)
        debug_print("Cleanup complete")
        
        # Create window in main thread (required for Qt backend)
        self.preview_window_counter += 1
        window_name = f"{self.preview_window_name}_{self.preview_window_counter}"
        print(f"[INFO] Creating preview window: {window_name}")
        
        try:
            window_flags = cv2.WINDOW_NORMAL
            try:
                window_flags |= cv2.WINDOW_GUI_EXPANDED
            except AttributeError:
                pass
            
            cv2.namedWindow(window_name, window_flags)
            cv2.resizeWindow(window_name, 640, 480)
            cv2.moveWindow(window_name, 100, 100)
            
            # Create a black frame to show initially so window appears
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black_frame, "Waiting for camera...", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(window_name, black_frame)
            cv2.waitKey(1)
            
            self.preview_window_name_current = window_name
            print(f"[INFO] Preview window created: {window_name}")
        except Exception as e:
            print(f"[ERROR] Failed to create window: {e}")
            traceback.print_exc()
            self.preview_on = False
            from tkinter import messagebox
            messagebox.showerror("Preview Error", f"Failed to create preview window: {e}")
            return
        
        self.preview_on = True
        print("[INFO] Starting preview thread")
        
        # Start periodic OpenCV event processing in main thread (needed for some backends)
        self._start_opencv_event_processing()
        
        # Start preview thread
        try:
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()
            print(f"[INFO] Preview thread started")
        except Exception as e:
            print(f"[ERROR] Failed to start preview thread: {e}")
            traceback.print_exc()
            self.preview_on = False
            self.preview_thread = None
            raise
    
    def stop(self):
        """Stop preview window."""
        self.preview_stop_count += 1
        debug_print(f"=== stop_preview() called (count: {self.preview_stop_count}) ===")
        
        if not self.preview_on:
            debug_print("Preview not on, returning early")
            return
        
        print("[INFO] Stopping preview")
        debug_print("Setting preview_on flag to False...")
        
        self.preview_on = False
        debug_print("Preview stopped")
        
        # Wait for thread to exit
        if self.preview_thread and self.preview_thread.is_alive():
            debug_print(f"Waiting for preview thread to exit (timeout=2.0)... thread_id={self.preview_thread.ident}")
            self.preview_thread.join(timeout=2.0)
            if self.preview_thread.is_alive():
                debug_print("WARN: Preview thread did not exit in time, but continuing cleanup")
            else:
                debug_print("Preview thread exited successfully")
        
        # Stop OpenCV event processing
        self.opencv_event_processing_active = False
        
        # Force cleanup of window in main thread (where it was created)
        debug_print("Starting window cleanup in stop_preview() (main thread)...")
        if self.preview_window_name_current:
            window_name = self.preview_window_name_current
            debug_print(f"Destroying window '{window_name}' in main thread...")
            try:
                cv2.destroyWindow(window_name)
                for _ in range(5):
                    cv2.waitKey(10)
                debug_print(f"Window '{window_name}' destroyed successfully")
            except Exception as e:
                debug_print(f"Exception destroying window: {type(e).__name__}: {e}")
            self.preview_window_name_current = None
        
        # Also call destroyAllWindows as backup
        try:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(10)
        except Exception as e:
            debug_print(f"Exception in destroyAllWindows: {type(e).__name__}: {e}")
        
        time.sleep(0.15)
        self.preview_thread = None
        debug_print("stop_preview() complete")
        print("[INFO] Preview stopped")
    
    def _start_opencv_event_processing(self):
        """Start periodic OpenCV event processing in main thread."""
        if self.opencv_event_processing_active:
            return
        
        self.opencv_event_processing_active = True
        debug_print("Starting OpenCV event processing in main thread")
        self._process_opencv_events()
    
    def _process_opencv_events(self):
        """Process OpenCV window events in main thread and auto-reopen if closed."""
        if not self.preview_on:
            self.opencv_event_processing_active = False
            return
        
        cam, is_open = self.get_camera()
        if not cam or not is_open:
            self.opencv_event_processing_active = False
            return
        
        try:
            window_name = self.preview_window_name_current
            if not window_name:
                debug_print("No window name, reopening preview...")
                self.root.after(100, self.start)
                self.opencv_event_processing_active = False
                return
            
            try:
                window_visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                if window_visible < 1:
                    debug_print("Window closed by user, auto-reopening preview...")
                    self.preview_window_name_current = None
                    self.root.after(100, self.start)
                    self.opencv_event_processing_active = False
                    return
                cv2.waitKey(1)
            except cv2.error:
                debug_print("Window no longer exists, auto-reopening preview...")
                self.preview_window_name_current = None
                self.root.after(100, self.start)
                self.opencv_event_processing_active = False
                return
        except Exception as e:
            debug_print(f"Exception in OpenCV event processing: {type(e).__name__}: {e}")
        
        if self.opencv_event_processing_active:
            self.root.after(50, self._process_opencv_events)
    
    def _preview_loop(self):
        """Preview thread loop - barebones, just show frames."""
        window_name = self.preview_window_name_current
        if not window_name:
            print("[ERROR] No window name available")
            self.preview_on = False
            return
        
        print(f"[INFO] Preview loop started with window: {window_name}")
        frame_count = 0
        first_frame = True
        no_frame_warnings = 0
        
        while self.preview_on:
            cam, is_open = self.get_camera()
            if not cam or not is_open:
                print("[INFO] Preview: Camera closed, exiting")
                break
            
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                if frame is None:
                    no_frame_warnings += 1
                    if no_frame_warnings == 10:
                        print(f"[WARN] Preview: Got {no_frame_warnings} None frames from queue")
                    continue
                
                no_frame_warnings = 0
                
                # Validate frame
                if len(frame.shape) < 2:
                    print(f"[WARN] Preview: Invalid frame shape: {frame.shape}")
                    continue
                    
                if frame.shape[0] <= 0 or frame.shape[1] <= 0:
                    print(f"[WARN] Preview: Invalid frame dimensions: {frame.shape}")
                    continue
                
                # OpenCV's read() typically returns BGR frames, but handle other formats
                try:
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        # Already BGR (most common case)
                        display_frame = frame
                    elif len(frame.shape) == 2:
                        # Grayscale - convert to BGR
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif len(frame.shape) == 3:
                        # Other 3-channel format (unlikely) - try YUYV conversion
                        # This handles raw YUYV if OpenCV didn't convert it
                        try:
                            display_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
                        except:
                            # If conversion fails, use frame as-is
                            display_frame = frame
                    else:
                        # Unknown format - try to use as-is
                        display_frame = frame
                except Exception as e:
                    print(f"[ERROR] Preview: Frame processing failed: {e}, frame shape: {frame.shape}")
                    continue
                
                if first_frame:
                    print(f"[INFO] Preview: First frame! Original shape: {frame.shape}, Display shape: {display_frame.shape}")
                    first_frame = False
                
                # Show frame
                try:
                    cv2.imshow(window_name, display_frame)
                    cv2.waitKey(1)
                except cv2.error as e:
                    print(f"[ERROR] Preview: imshow failed: {e}")
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"[INFO] Preview: Displayed {frame_count} frames")
                    
            except Empty:
                # No frame available, wait
                if frame_count == 0 and first_frame:
                    # Only warn once if we're waiting for first frame
                    pass
            except Exception as e:
                print(f"[ERROR] Preview error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        self.preview_on = False
        print(f"[INFO] Preview loop exited. Total frames: {frame_count}")

