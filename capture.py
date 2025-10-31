#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Platform Camera Recorder with External OpenCV Preview
------------------------------------------------------------
✓ Supports UVC (Linux), DirectShow (Windows), and other backends
✓ Auto-detects available cameras
✓ Recording thread writes frames directly
✓ Optional live preview in separate OpenCV window
✓ Tkinter GUI with camera selection dropdown

Refined to match pair_detect.py structure:
- Uses lib/ modules for camera and utilities
- Uses util_paths for output directory management
- Follows same code organization patterns
"""

import cv2
import time
import threading
import os
from queue import Queue, Empty

import tkinter as tk
from tkinter import ttk, messagebox

from lib.camera import Camera, detect_cameras
from lib.capture.capture_utils import safe_run
from lib.capture.util_paths import make_capture_output_path


# ============ Default settings ============
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FPS = 30
DEFAULT_FOURCC = "MJPG"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


# ============ GUI Application ============
class CaptureApp:
    """Main application class for camera recording with GUI."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Recorder — Cross-Platform")
        
        # Camera state
        self.cam = None
        self.camera_index = DEFAULT_CAMERA_INDEX
        self.fps = DEFAULT_FPS
        self.fourcc = DEFAULT_FOURCC
        self.available_cameras = []
        
        # Recording state
        self.recording = False
        self.stop_flag = False
        self.video_writer = None
        self.thread = None
        
        # Preview state
        self.preview_on = False
        self.preview_thread = None
        
        # Frame grabbing (single thread feeds both preview and recording)
        self.frame_queue = Queue(maxsize=10)  # Buffer to allow both preview and recording
        self.frame_grabber_thread = None
        self.frame_grabber_running = False
        
        # Build UI first (needed for camera_combo widget)
        self._build_ui()
        
        # Detect cameras on startup (after UI is built)
        self.refresh_camera_list()

    def _build_ui(self):
        """Build the Tkinter GUI."""
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill=tk.BOTH, expand=True)

        # Camera selection frame
        cam_frame = ttk.LabelFrame(main, text="Camera Selection", padding="5")
        cam_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(cam_frame, text="Camera:").grid(row=0, column=0, padx=(0, 5), sticky="w")
        
        self.camera_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(
            cam_frame, 
            textvariable=self.camera_var,
            width=30,
            state="readonly"
        )
        self.camera_combo.grid(row=0, column=1, padx=5, sticky="ew")
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)
        
        ttk.Button(cam_frame, text="Refresh", command=self.refresh_camera_list).grid(row=0, column=2, padx=5)
        
        cam_frame.columnconfigure(1, weight=1)
        
        self._update_camera_combo()

        # Status label
        self.status_label = ttk.Label(
            main, 
            text="Idle", 
            font=("TkDefaultFont", 14, "bold")
        )
        self.status_label.pack(pady=20)

        # Button frame
        btns = ttk.Frame(main)
        btns.pack(pady=10)
        
        ttk.Button(btns, text="Open Camera", command=self.open_camera).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="Start Recording", command=self.start_record).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text="Stop Recording", command=self.stop_record).grid(row=0, column=2, padx=4)
        ttk.Button(btns, text="Preview On/Off", command=self.toggle_preview).grid(row=0, column=3, padx=4)
        ttk.Button(btns, text="Exit", command=self.on_close).grid(row=0, column=4, padx=4)

        # Info label
        info_text = f"{DEFAULT_FOURCC} {DEFAULT_WIDTH}×{DEFAULT_HEIGHT} @ {DEFAULT_FPS} FPS"
        ttk.Label(main, text=info_text).pack(pady=10)
    
    def refresh_camera_list(self):
        """Refresh list of available cameras."""
        print("[INFO] Refreshing camera list...")
        self.available_cameras = detect_cameras(max_test=10)
        if not self.available_cameras:
            print("[WARN] No cameras detected")
            self.available_cameras = [0]  # Default to index 0
        self._update_camera_combo()
    
    def _update_camera_combo(self):
        """Update camera dropdown with available cameras."""
        if not hasattr(self, 'camera_combo'):
            return  # UI not built yet
        values = [f"Camera {idx}" for idx in self.available_cameras]
        self.camera_combo['values'] = values
        if self.available_cameras:
            current_idx = min(self.camera_index, len(self.available_cameras) - 1)
            if current_idx < len(values):
                self.camera_combo.current(current_idx)
                self.camera_var.set(values[current_idx])
    
    def on_camera_selected(self, event=None):
        """Handle camera selection change."""
        selection = self.camera_var.get()
        if selection:
            try:
                idx_str = selection.replace("Camera ", "")
                self.camera_index = int(idx_str)
                print(f"[INFO] Selected camera index: {self.camera_index}")
            except:
                pass

    # ------------------------------------------------------------------
    def open_camera(self):
        """Open camera with current settings."""
        # Stop frame grabber if running
        self._stop_frame_grabber()
        
        if self.cam:
            self.cam.release()
            self.cam = None
        
        # Get selected camera index
        try:
            selection = self.camera_var.get()
            if selection:
                idx_str = selection.replace("Camera ", "")
                self.camera_index = int(idx_str)
        except:
            pass
        
        self.cam = Camera(
            index=self.camera_index,
            fps=self.fps,
            fourcc=self.fourcc,
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT
        )
        
        if not self.cam.open():
            error_msg = f"Failed to open camera {self.camera_index}\n\n"
            if self.available_cameras:
                error_msg += f"Available cameras: {', '.join(map(str, self.available_cameras))}\n"
            error_msg += "Try clicking 'Refresh' to detect cameras again."
            messagebox.showerror("Camera Error", error_msg)
            return
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        # Start frame grabber thread
        self._start_frame_grabber()
        
        print("[INFO] Camera ready")
        self.status_label.config(text=f"Camera {self.camera_index} ready", foreground="green")
    
    def _start_frame_grabber(self):
        """Start the frame grabbing thread."""
        if self.frame_grabber_running:
            return
        self.frame_grabber_running = True
        self.frame_grabber_thread = threading.Thread(target=self._frame_grabber_loop, daemon=True)
        self.frame_grabber_thread.start()
    
    def _stop_frame_grabber(self):
        """Stop the frame grabbing thread."""
        self.frame_grabber_running = False
        if self.frame_grabber_thread and self.frame_grabber_thread.is_alive():
            self.frame_grabber_thread.join(timeout=2.0)
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
    
    def _frame_grabber_loop(self):
        """Single thread that reads frames from camera and feeds the queue."""
        consecutive_errors = 0
        max_errors = 30  # Allow more errors before giving up (timeouts are expected)
        
        while self.frame_grabber_running and self.cam and self.cam.is_open():
            frame = self.cam.read()
            if frame is not None:
                consecutive_errors = 0
                # Put frame in queue (drop old frame if queue is full to avoid lag)
                try:
                    # If both preview and recording are active, we need to feed both
                    # Put original frame, and if queue has room, put a clone too
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()  # Remove oldest frame
                        except Empty:
                            pass
                    
                    # Put the frame
                    self.frame_queue.put_nowait(frame)
                    
                    # If both preview and recording are active, put a clone so both can consume
                    # This ensures neither thread starves
                    if self.preview_on and self.recording and not self.frame_queue.full():
                        try:
                            frame_clone = frame.copy()
                            self.frame_queue.put_nowait(frame_clone)
                        except:
                            pass  # If clone fails, skip it
                            
                except Exception as e:
                    # Queue error, skip this frame
                    pass
            else:
                consecutive_errors += 1
                if consecutive_errors >= max_errors:
                    print("[WARN] Too many consecutive frame read errors, stopping frame grabber")
                    break
                time.sleep(0.01)  # Small delay to prevent busy waiting

    # ------------------------------------------------------------------
    def start_record(self):
        """Start recording video to file."""
        if not self.cam or not self.cam.is_open():
            self.open_camera()
            if not self.cam or not self.cam.is_open():
                return
        
        if self.recording:
            print("[WARN] Already recording")
            return

        # Generate output path using util_paths
        output_path = make_capture_output_path(self.cam.w, self.cam.h, self.cam.fps)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            output_path, 
            fourcc, 
            self.cam.fps, 
            (self.cam.w, self.cam.h)
        )
        
        if not self.video_writer.isOpened():
            messagebox.showerror("Recording", f"Failed to create video file: {output_path}")
            return
        
        print(f"[INFO] Recording started: {output_path}")

        self.stop_flag = False
        self.recording = True
        self.status_label.config(text=f"Recording: {os.path.basename(output_path)}", foreground="red")
        
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------
    def _record_loop(self):
        """Recording thread loop."""
        frame_count = 0
        dropped_frames = 0
        t0 = time.time()
        
        while not self.stop_flag and self.cam and self.cam.is_open():
            try:
                # Get frame from queue (with timeout to avoid blocking forever)
                frame = self.frame_queue.get(timeout=0.1)
                if frame is not None:
                    try:
                        # Clone frame before writing (in case preview also has reference)
                        frame_to_write = frame.copy() if self.preview_on else frame
                        self.video_writer.write(frame_to_write)
                        frame_count += 1
                    except Exception as e:
                        print(f"[WARN] Frame write error: {e}")
                        dropped_frames += 1
            except Empty:
                # No frame available, continue waiting
                if not self.frame_grabber_running:
                    # Frame grabber stopped, exit recording
                    break
                continue
            except Exception as e:
                dropped_frames += 1
        
        duration = time.time() - t0
        fps_measured = frame_count / duration if duration > 0 else 0
        
        print(f"[INFO] Recorded {frame_count} frames in {duration:.1f}s ({fps_measured:.1f} FPS)")
        if dropped_frames > 0:
            print(f"[INFO] Dropped {dropped_frames} invalid frames during recording")

    # ------------------------------------------------------------------
    def stop_record(self):
        """Stop recording and close video file."""
        if not self.recording:
            return
        
        self.stop_flag = True
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.recording = False
        self.status_label.config(text="Idle", foreground="black")
        print("[INFO] Recording stopped")

    # ------------------------------------------------------------------
    def toggle_preview(self):
        """Toggle live preview window."""
        if not self.cam or not self.cam.is_open():
            messagebox.showinfo("Camera", "Open camera first.")
            return
        
        if not self.preview_on:
            print("[INFO] Preview enabled")
            self.preview_on = True
            self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
            self.preview_thread.start()
        else:
            print("[INFO] Preview disabled")
            self.preview_on = False

    # ------------------------------------------------------------------
    def _preview_loop(self):
        """Preview thread loop with FPS overlay."""
        cv2.namedWindow("Preview", cv2.WINDOW_AUTOSIZE)
        last_time = time.time()
        frame_count = 0
        fps = 0.0
        
        try:
            while self.preview_on and self.cam and self.cam.is_open():
                try:
                    # Get frame from queue (with timeout)
                    frame = self.frame_queue.get(timeout=0.1)
                    if frame is not None:
                        
                        # Calculate FPS (moving average)
                        frame_count += 1
                        now = time.time()
                        if frame_count % 10 == 0:
                            elapsed = now - last_time
                            fps = 10.0 / elapsed if elapsed > 0 else 0
                            last_time = now
                        
                        # Draw FPS overlay
                        try:
                            cv2.putText(
                                frame, 
                                f"{fps:.1f} FPS", 
                                (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, 
                                (0, 0, 255), 
                                2
                            )
                            cv2.imshow("Preview", frame)
                        except Exception as e:
                            print(f"[WARN] Preview display error: {e}")
                except Empty:
                    # No frame available, check if we should continue
                    if not self.frame_grabber_running:
                        # Frame grabber stopped, exit preview
                        break
                    continue
                
                # ESC key closes preview
                if cv2.waitKey(1) & 0xFF == 27:
                    self.preview_on = False
                    break
        except Exception as e:
            print(f"[ERROR] Preview loop error: {e}")
        finally:
            try:
                cv2.destroyWindow("Preview")
            except:
                pass

    # ------------------------------------------------------------------
    def on_close(self):
        """Cleanup on application close."""
        print("[INFO] Exiting…")
        self.preview_on = False
        self.stop_record()
        self._stop_frame_grabber()
        if self.cam:
            self.cam.release()
        self.root.destroy()


# ============ Main ============
def main():
    """Main entry point."""
    root = tk.Tk()
    app = CaptureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    
    # Handle ESC key
    root.bind("<Escape>", lambda e: app.on_close())
    
    print("[INFO] Camera Recorder ready. Use GUI controls.")
    root.mainloop()


if __name__ == "__main__":
    main()
