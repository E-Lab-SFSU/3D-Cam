#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame Grabber - Reads frames from camera and feeds a queue
"""

import time
import threading
from queue import Queue, Empty


class FrameGrabber:
    """Manages frame grabbing from camera to queue."""
    
    def __init__(self, frame_queue):
        """
        Initialize frame grabber.
        
        Args:
            frame_queue: Queue to put frames into
        """
        self.frame_queue = frame_queue
        self.running = False
        self.thread = None
        self.camera = None
    
    def start(self, camera):
        """Start frame grabbing thread."""
        if self.running:
            return
        
        self.camera = camera
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("[INFO] Frame grabber started")
    
    def stop(self):
        """Stop frame grabbing thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        print("[INFO] Frame grabber stopped")
    
    def _loop(self):
        """Frame grabbing loop."""
        import threading
        
        consecutive_errors = 0
        max_errors = 100
        frames_read = 0
        last_status_time = time.time()
        
        print("[INFO] Frame grabber: Starting frame capture...")
        
        while self.running:
            # Check if camera is still valid and open
            if not self.camera:
                break
            try:
                if not self.camera.is_open():
                    break
            except:
                # Camera object might be invalid
                break
            
            # Read frame
            try:
                frame = self.camera.read(max_retries=1)
            except Exception as e:
                print(f"[WARN] Frame read exception: {e}")
                frame = None
                consecutive_errors += 1
                if consecutive_errors >= max_errors:
                    print(f"[ERROR] Too many errors, stopping frame grabber")
                    break
                time.sleep(0.1)
                continue
            
            if frame is not None:
                consecutive_errors = 0
                frames_read += 1
                
                try:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    
                    self.frame_queue.put_nowait(frame)
                    
                    if frames_read % 30 == 0:
                        elapsed = time.time() - last_status_time
                        actual_fps = 30.0 / elapsed if elapsed > 0 else 0
                        print(f"[INFO] Frame grabber: {frames_read} frames read ({actual_fps:.1f} FPS avg)")
                        last_status_time = time.time()
                except Exception as e:
                    print(f"[WARN] Frame queue error: {e}")
            else:
                consecutive_errors += 1
                
                if consecutive_errors % 10 == 0 and consecutive_errors < max_errors:
                    print(f"[WARN] Frame grabber: {consecutive_errors} consecutive read failures")
                
                if consecutive_errors >= max_errors:
                    print(f"[ERROR] Too many consecutive frame read errors ({consecutive_errors}), stopping frame grabber")
                    print(f"[INFO] Frame grabber: Total frames read: {frames_read}")
                    break
                
                time.sleep(0.05)

