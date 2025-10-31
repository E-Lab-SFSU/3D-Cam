#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera module for cross-platform camera control
Supports UVC (Linux), DirectShow (Windows), and other backends
"""

import cv2
import time
import platform
import threading
import os


def is_raspberry_pi():
    """Check if running on Raspberry Pi."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            return 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo
    except:
        return False


def is_linux():
    """Check if running on Linux."""
    return platform.system() == "Linux"


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


def read_frame_with_timeout(cap, timeout=2.0):
    """
    Try to read a frame from VideoCapture with a timeout.
    Returns (success, frame) or (False, None) if timeout.
    """
    result = [None, None]  # [ret, frame]
    exception = [None]
    
    def read_thread():
        try:
            result[0], result[1] = cap.read()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=read_thread, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        return False, None
    
    if exception[0]:
        return False, None
    
    return result[0], result[1]


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
    
    # Get preferred backend first (needed for device scanning)
    preferred_backend = get_camera_backend()
    is_raspi = is_raspberry_pi()
    
    # On Linux, pre-scan available /dev/video* devices to speed up detection
    available_devices = []
    if is_linux() and preferred_backend == cv2.CAP_V4L2:
        # Pre-scan /dev/video* devices to avoid opening non-existent ones
        for idx in range(max_test):
            dev_path = f"/dev/video{idx}"
            if os.path.exists(dev_path):
                available_devices.append(idx)
    
    available = []
    
    # Adjust timeout based on platform (Raspberry Pi is slower)
    detection_timeout = 3.0 if is_raspi else 1.5
    
    # On Windows, try MSMF first (more reliable than DSHOW for some cameras)
    if platform.system() == "Windows":
        backends_to_try = [cv2.CAP_MSMF, preferred_backend, cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        # On Linux, prefer V4L2 and limit backends to try
        backends_to_try = [preferred_backend]
    
    print(f"[INFO] Detecting cameras... (OS: {platform.system()}, Raspberry Pi: {is_raspi})")
    if available_devices:
        print(f"[INFO] Found {len(available_devices)} video devices: {available_devices}")
    
    with suppress_stderr():
        # On Linux, only check devices we found, otherwise check all indices
        indices_to_check = available_devices if available_devices else range(max_test)
        
        for idx in indices_to_check:
            
            found = False
            for backend in backends_to_try:
                cap = None
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        # Check if we can get properties (faster than reading frame)
                        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        
                        # Only validate with frame read if properties look valid
                        # Otherwise skip frame read to avoid long timeouts
                        if width > 0 and height > 0:
                            # Try to read a frame with timeout to verify it works
                            # Use platform-appropriate timeout
                            ret, frame = read_frame_with_timeout(cap, timeout=detection_timeout)
                            if ret and frame is not None:
                                # Get camera info (without suppressing stderr for this)
                                backend_name = cap.getBackendName()
                                print(f"[INFO] Camera {idx} found: {int(width)}×{int(height)} (backend: {backend_name})")
                                available.append(idx)
                                found = True
                            # If frame read timed out but properties are valid, still add it
                            # (camera might be busy but still exists)
                            elif width > 0 and height > 0:
                                # Add without frame verification if properties are valid
                                # This helps detect cameras that are slow to respond
                                backend_name = cap.getBackendName()
                                print(f"[INFO] Camera {idx} found: {int(width)}×{int(height)} (backend: {backend_name}, not verified with frame read)")
                                available.append(idx)
                                found = True
                    if cap:
                        cap.release()
                    if found:
                        break
                except Exception:
                    # Silently continue if backend fails
                    if cap:
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
        is_raspi = is_raspberry_pi()
        is_linux_os = is_linux()
        
        # Determine backends to try
        if platform.system() == "Windows":
            # On Windows, try DSHOW first (often more reliable than MSMF for some cameras)
            backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, self.backend, cv2.CAP_V4L2, cv2.CAP_ANY]
        else:
            # On Linux, prefer V4L2 and limit backends
            backends_to_try = [self.backend]
        
        # Remove duplicates while preserving order
        seen = set()
        backends_to_try = [x for x in backends_to_try if not (x in seen or seen.add(x))]
        
        # Try each backend and verify it can read frames successfully
        for backend_alt in backends_to_try:
            if self.cap:
                self.cap.release()
            
            # Open camera with backend
            if is_linux_os and backend_alt == cv2.CAP_V4L2:
                # On Linux/V4L2, use device path directly if possible for better reliability
                dev_path = f"/dev/video{self.index}"
                if os.path.exists(dev_path):
                    # Open by device path (more reliable than index on some systems)
                    self.cap = cv2.VideoCapture(dev_path, backend_alt)
                else:
                    self.cap = cv2.VideoCapture(self.index, backend_alt)
            else:
                self.cap = cv2.VideoCapture(self.index, backend_alt)
            
            if self.cap.isOpened():
                # On Linux/V4L2, set additional properties for better performance
                if is_linux_os and backend_alt == cv2.CAP_V4L2:
                    try:
                        # Set buffer size to 1 to reduce latency and avoid timeouts
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except:
                        pass
                
                # Check if we can get valid properties (faster than reading frames)
                try:
                    width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    properties_valid = width > 0 and height > 0
                except:
                    properties_valid = False
                
                # Verify we can actually read valid frames (try fewer times with timeout)
                # But on Linux/V4L2, be more lenient - if properties are valid, accept it
                valid_frames = 0
                max_attempts = 2 if is_raspi else 3  # Fewer attempts on Pi to speed up
                
                for _ in range(max_attempts):
                    try:
                        # Use timeout for frame reads during opening
                        ret, frame = read_frame_with_timeout(self.cap, timeout=3.0 if is_raspi else 1.5)
                        if ret and frame is not None:
                            # Validate frame structure
                            if len(frame.shape) >= 2 and frame.shape[0] > 0 and frame.shape[1] > 0:
                                valid_frames += 1
                                if valid_frames >= 1:  # Need at least 1 valid frame
                                    self.backend = backend_alt
                                    break
                    except (cv2.error, Exception):
                        # Frame read failed, continue trying
                        continue
                
                # Accept camera if we got valid frames OR if properties are valid (for V4L2 that's slow)
                if valid_frames >= 1 or (properties_valid and is_linux_os and backend_alt == cv2.CAP_V4L2):
                    self.backend = backend_alt
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
        
        # Determine if using V4L2 (check actual backend name or backend type)
        using_v4l2 = is_linux_os and (self.backend == cv2.CAP_V4L2 or 'V4L2' in str(self.actual_backend))
        
        # Set buffer size - critical for V4L2 to avoid timeouts
        try:
            if using_v4l2:
                # V4L2: smaller buffer = lower latency but less tolerance for delays
                # Always use 1 for V4L2 to minimize timeouts
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
        
        # Shorter stabilization time on Linux/V4L2
        stabilization_time = 0.3 if using_v4l2 else 0.5
        time.sleep(stabilization_time)
        
        # Flush initial frames (camera warm-up) - fewer on Linux/V4L2 to avoid timeouts
        flush_count = 5 if using_v4l2 else 10
        for _ in range(flush_count):
            try:
                # Use timeout for flushing to avoid blocking
                ret, frame = read_frame_with_timeout(self.cap, timeout=0.5)
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

