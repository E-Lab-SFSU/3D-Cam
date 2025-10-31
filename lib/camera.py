#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera module for cross-platform camera control
Supports UVC (Linux), DirectShow (Windows), and other backends
"""

import cv2
import time
import platform


def get_camera_backend():
    """Get appropriate OpenCV backend for current platform."""
    system = platform.system()
    if system == "Linux":
        return cv2.CAP_V4L2
    elif system == "Windows":
        return cv2.CAP_DSHOW  # DirectShow
    elif system == "Darwin":  # macOS
        return cv2.CAP_AVFOUNDATION
    else:
        return cv2.CAP_ANY  # Auto-detect


def detect_cameras(max_test=10, suppress_warnings=True):
    """
    Detect available cameras by testing indices with multiple backends.
    Returns list of working camera indices.
    
    Args:
        max_test: Maximum camera index to test (default: 10)
        suppress_warnings: If True, suppress OpenCV warnings during detection (default: True)
    """
    import os
    import sys
    import contextlib
    
    # Context manager to suppress stderr (OpenCV warnings)
    @contextlib.contextmanager
    def suppress_stderr():
        if suppress_warnings:
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    yield
                finally:
                    sys.stderr = old_stderr
        else:
            yield
    
    available = []
    preferred_backend = get_camera_backend()
    # On Windows, try MSMF first (more reliable than DSHOW for some cameras)
    import platform
    if platform.system() == "Windows":
        backends_to_try = [cv2.CAP_MSMF, preferred_backend, cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends_to_try = [preferred_backend, cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
    
    print("[INFO] Detecting cameras...")
    with suppress_stderr():
        for idx in range(max_test):
            found = False
            for backend in backends_to_try:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        # Try to read a frame to verify it works
                        ret, _ = cap.read()
                        if ret:
                            # Get camera info (without suppressing stderr for this)
                            backend_name = cap.getBackendName()
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            print(f"[INFO] Camera {idx} found: {width}×{height} (backend: {backend_name})")
                            available.append(idx)
                            found = True
                    cap.release()
                    if found:
                        break
                except Exception:
                    # Silently continue if backend fails
                    try:
                        cap.release()
                    except:
                        pass
                    continue
    
    return available


class Camera:
    """Cross-platform camera interface supporting UVC and other webcams."""
    
    def __init__(self, index=0, fps=30, fourcc="MJPG", width=1280, height=720, backend=None):
        self.index = index
        self.fps = fps
        self.fourcc = fourcc
        self.width = width
        self.height = height
        self.backend = backend if backend is not None else get_camera_backend()
        self.cap = None
        self.w = None
        self.h = None
        self.actual_backend = None

    def open(self):
        """Open camera with specified settings. Tries multiple backends if needed."""
        import platform
        
        # Determine backends to try
        if platform.system() == "Windows":
            # On Windows, try DSHOW first (often more reliable than MSMF for some cameras)
            backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, self.backend, cv2.CAP_V4L2, cv2.CAP_ANY]
        else:
            backends_to_try = [self.backend, cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
        
        # Remove duplicates while preserving order
        seen = set()
        backends_to_try = [x for x in backends_to_try if not (x in seen or seen.add(x))]
        
        # Try each backend and verify it can read frames successfully
        for backend_alt in backends_to_try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.index, backend_alt)
            if self.cap.isOpened():
                # Verify we can actually read valid frames (try a few times)
                valid_frames = 0
                for _ in range(5):  # Try up to 5 frames
                    try:
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            # Validate frame structure
                            if len(frame.shape) >= 2 and frame.shape[0] > 0 and frame.shape[1] > 0:
                                valid_frames += 1
                                if valid_frames >= 2:  # Need at least 2 valid frames
                                    self.backend = backend_alt
                                    break
                    except (cv2.error, Exception):
                        # Frame read failed, continue trying
                        continue
                
                if valid_frames >= 2:
                    break
                else:
                    # Backend didn't work, try next
                    self.cap.release()
                    self.cap = None
        
        if not self.cap or not self.cap.isOpened():
            print(f"[ERROR] Failed to open camera {self.index} with any backend")
            return False
        
        self.actual_backend = self.cap.getBackendName()
        
        # Set properties (may not work on all cameras)
        try:
            # Try to set FOURCC (works better on UVC cameras)
            if self.fourcc and self.fourcc != "AUTO":
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        except:
            pass  # Some cameras don't support this
        
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except:
            pass
        
        try:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        except:
            pass
        
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        except:
            pass
        
        time.sleep(0.5)  # Allow camera to stabilize
        
        # Flush initial frames (camera warm-up)
        for _ in range(10):
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    if len(frame.shape) >= 2 and frame.shape[0] > 0 and frame.shape[1] > 0:
                        pass  # Valid frame, continue flushing
            except:
                pass
        
        # Get actual dimensions (may differ from requested)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        actual_fourcc = ""
        try:
            fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            actual_fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        except:
            actual_fourcc = "unknown"
        
        print(f"[INFO] Camera {self.index} opened: {actual_fourcc} {self.w}×{self.h}@{actual_fps:.0f} FPS (backend: {self.actual_backend})")
        return True

    def read(self, max_retries=3):
        """
        Read a frame from the camera. Returns frame or None.
        
        Args:
            max_retries: Maximum number of retries if frame read fails (default: 3)
        """
        if not self.cap or not self.cap.isOpened():
            return None
        
        # Retry logic for reading frames
        for attempt in range(max_retries):
            try:
                ok, frame = self.cap.read()
                
                # Check if read was successful
                if not ok:
                    if attempt < max_retries - 1:
                        continue  # Retry
                    return None
                
                # Validate frame exists
                if frame is None:
                    if attempt < max_retries - 1:
                        continue  # Retry
                    return None
                
                # Validate frame structure (this might trigger the error if frame is corrupted)
                try:
                    # Check shape first (this is safe)
                    if len(frame.shape) < 2:
                        if attempt < max_retries - 1:
                            continue
                        return None
                    
                    # Check dimensions
                    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
                        if attempt < max_retries - 1:
                            continue
                        return None
                    
                    # Check size and dtype (accessing these validates the Mat structure)
                    if frame.size <= 0:
                        if attempt < max_retries - 1:
                            continue
                        return None
                    
                    # If we get here, frame is valid
                    return frame
                    
                except (cv2.error, AttributeError, ValueError) as e:
                    # Frame structure is invalid/corrupted
                    if attempt < max_retries - 1:
                        # Try reading another frame
                        continue
                    # All retries failed
                    return None
                    
            except cv2.error as e:
                # OpenCV error during read operation itself
                if attempt < max_retries - 1:
                    time.sleep(0.01)  # Small delay before retry
                    continue
                # Don't print warnings for every failed frame - too noisy
                return None
            except Exception as e:
                # Unexpected error
                if attempt < max_retries - 1:
                    continue
                return None
        
        return None

    def release(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
            print("[INFO] Camera released")

    def is_open(self):
        """Check if camera is open."""
        return self.cap is not None and self.cap.isOpened()

