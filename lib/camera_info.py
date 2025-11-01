#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Information Helper
--------------------------
Provides detailed information about connected cameras for use in GUI applications
and preview scripts. Supports cross-platform camera detection and information gathering.

Usage:
    from lib.camera_info import get_camera_info, list_all_cameras
    
    # Get info for all cameras
    cameras = list_all_cameras()
    
    # Get detailed info for a specific camera
    info = get_camera_info(0)
    
Command line usage:
    python lib/camera_info.py [options]
    python -m lib.camera_info [options]
"""

import cv2
import platform
import os
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import subprocess
import sys
import contextlib

# Handle imports when running as script vs module
# Add parent directory to path if running as script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from lib.camera import detect_cameras, get_camera_backend, is_linux, is_raspberry_pi, read_frame_with_timeout


def fourcc_int_to_string(fourcc_int: int) -> str:
    """
    Convert a FOURCC integer code to a readable string.
    Filters out non-printable characters and maps known codes.
    
    Args:
        fourcc_int: FOURCC code as integer
        
    Returns:
        Readable FOURCC string (e.g., "YUYV", "MJPG")
    """
    if fourcc_int == 0:
        return "unknown"
    
    # Try different byte orders and interpretations
    # FOURCC codes can be stored in different endianness
    
    # Method 1: Little-endian (standard FOURCC, bytes 0-3 as chars)
    chars_le = []
    for i in range(4):
        byte_val = (fourcc_int >> (8 * i)) & 0xFF
        if 65 <= byte_val <= 90 or 48 <= byte_val <= 57:  # A-Z, 0-9
            chars_le.append(chr(byte_val))
        elif byte_val == 0:
            break
        else:
            chars_le.append(None)  # Mark as invalid
    
    # Method 2: Big-endian (reverse order)
    chars_be = []
    for i in range(3, -1, -1):
        byte_val = (fourcc_int >> (8 * i)) & 0xFF
        if 65 <= byte_val <= 90 or 48 <= byte_val <= 57:  # A-Z, 0-9
            chars_be.append(chr(byte_val))
        elif byte_val == 0:
            break
        else:
            chars_be.append(None)
    
    # Check which interpretation gives valid result
    valid_le = all(c is not None for c in chars_le) and len(chars_le) == 4
    valid_be = all(c is not None for c in chars_be) and len(chars_be) == 4
    
    if valid_le:
        result = "".join(chars_le)
        # Check against known codes
        if result in ["YUYV", "YUY2", "MJPG", "H264", "YUV2"]:
            return result
    
    if valid_be:
        result = "".join(chars_be)
        if result in ["YUYV", "YUY2", "MJPG", "H264", "YUV2"]:
            return result
    
    # Method 3: Hex pattern matching (for corrupted displays)
    hex_str = f"{fourcc_int:08X}"
    # Check hex string for known patterns (case-insensitive)
    hex_upper = hex_str.upper()
    if "59555956" in hex_upper or "56595559" in hex_upper:  # YUYV patterns
        return "YUYV"
    if "4A504D47" in hex_upper or "47504A4D" in hex_upper:  # MJPG patterns
        return "MJPG"
    if "32315559" in hex_upper or "59553132" in hex_upper:  # YUY2 patterns
        return "YUY2"
    
    # Method 4: Extract any valid ASCII letters/numbers
    chars = []
    for i in range(4):
        byte_val = (fourcc_int >> (8 * i)) & 0xFF
        if 65 <= byte_val <= 90:  # A-Z
            chars.append(chr(byte_val))
        elif 48 <= byte_val <= 57:  # 0-9
            chars.append(chr(byte_val))
        elif byte_val == 0:
            break
    
    if len(chars) >= 3:
        result = "".join(chars)
        # If it matches a known pattern (even if incomplete)
        if "YUY" in result or "YUV" in result:
            return "YUYV"
        if "MJP" in result or "JPG" in result:
            return "MJPG"
        return result
    
    return "unknown"


@dataclass
class CameraInfo:
    """Structured information about a camera."""
    index: int
    device_path: Optional[str] = None  # e.g., "/dev/video0" on Linux
    name: Optional[str] = None  # Camera name/model
    backend: Optional[str] = None  # OpenCV backend name (e.g., "V4L2", "DSHOW")
    is_available: bool = False
    default_width: int = 0
    default_height: int = 0
    default_fps: float = 0.0
    default_fourcc: str = ""
    supported_resolutions: List[Tuple[int, int]] = None  # List of (width, height) tuples
    supported_fps: List[float] = None  # List of supported FPS values
    supported_fourcc: List[str] = None  # List of supported FOURCC codes
    
    def __post_init__(self):
        """Initialize lists if None."""
        if self.supported_resolutions is None:
            self.supported_resolutions = []
        if self.supported_fps is None:
            self.supported_fps = []
        if self.supported_fourcc is None:
            self.supported_fourcc = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy serialization."""
        return asdict(self)
    
    def __str__(self) -> str:
        """String representation."""
        name_str = f" ({self.name})" if self.name else ""
        dev_str = f" [{self.device_path}]" if self.device_path else ""
        status = "✓" if self.is_available else "✗"
        return f"{status} Camera {self.index}{name_str}{dev_str}: {self.default_width}×{self.default_height} @ {self.default_fps:.1f} FPS ({self.backend})"


def get_v4l2_camera_name(device_path: str) -> Optional[str]:
    """
    Get camera name using v4l2-ctl on Linux.
    Falls back to checking /sys filesystem if v4l2-ctl is not available.
    
    Args:
        device_path: Path to video device (e.g., "/dev/video0")
    
    Returns:
        Camera name/model string or None
    """
    if not is_linux():
        return None
    
    # Try v4l2-ctl first (most reliable)
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--device', device_path, '--info'],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Card type' in line or 'Driver name' in line:
                    # Extract name from line like "Card type       : USB2.0 Camera"
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        name = parts[1].strip()
                        if name:
                            return name
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # Fallback: try to read from /sys filesystem
    try:
        # Extract device number from path (e.g., "/dev/video0" -> "0")
        dev_num = device_path.replace('/dev/video', '')
        sys_path = f'/sys/class/video4linux/video{dev_num}/name'
        if os.path.exists(sys_path):
            with open(sys_path, 'r') as f:
                name = f.read().strip()
                if name:
                    return name
    except Exception:
        pass
    
    return None


def get_windows_camera_name(index: int) -> Optional[str]:
    """
    Try to get camera name on Windows using DirectShow or registry.
    This is limited on Windows as OpenCV doesn't expose camera names easily.
    
    Args:
        index: Camera index
    
    Returns:
        Camera name or None
    """
    # On Windows, we can try opening with DirectShow and checking properties
    # but OpenCV doesn't provide a reliable way to get camera names
    # This is a placeholder for future improvements
    return None


def set_camera_control(device_path: str, control_name: str, value: int) -> bool:
    """
    Set a camera control using v4l2-ctl (Linux only).
    
    Args:
        device_path: Path to video device (e.g., "/dev/video0")
        control_name: Name of control (e.g., "brightness", "contrast", "saturation", "gain", "power_line_frequency")
        value: Value to set
    
    Returns:
        True if successful, False otherwise
    """
    if not is_linux():
        return False
    
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--device', device_path, '-c', f'{control_name}={int(value)}'],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


def get_camera_control(device_path: str, control_name: str) -> Optional[int]:
    """
    Get a camera control value using v4l2-ctl (Linux only).
    
    Args:
        device_path: Path to video device (e.g., "/dev/video0")
        control_name: Name of control (e.g., "brightness", "contrast")
    
    Returns:
        Control value or None if failed
    """
    if not is_linux():
        return None
    
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--device', device_path, '-C', control_name],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        if result.returncode == 0:
            # Parse output like "brightness (int)    : min=-64 max=64 step=1 default=0 value=0"
            for line in result.stdout.split('\n'):
                if control_name in line.lower() and 'value=' in line:
                    try:
                        # Extract value after "value="
                        value_part = line.split('value=')[1].strip()
                        # Get number before any whitespace or newline
                        value = int(value_part.split()[0])
                        return value
                    except (ValueError, IndexError):
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    return None


def get_camera_control_range(device_path: str, control_name: str) -> Optional[Dict[str, int]]:
    """
    Get the range (min, max, default, step) for a camera control (Linux only).
    
    Args:
        device_path: Path to video device (e.g., "/dev/video0")
        control_name: Name of control
    
    Returns:
        Dictionary with 'min', 'max', 'default', 'step' keys, or None if failed
    """
    if not is_linux():
        return None
    
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--device', device_path, '-C', control_name],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        if result.returncode == 0:
            range_info = {}
            for line in result.stdout.split('\n'):
                if control_name in line.lower():
                    # Parse line like "brightness (int)    : min=-64 max=64 step=1 default=0 value=0"
                    for key in ['min', 'max', 'default', 'step']:
                        if f'{key}=' in line:
                            try:
                                value_str = line.split(f'{key}=')[1].strip().split()[0]
                                range_info[key] = int(value_str)
                            except (ValueError, IndexError):
                                pass
            if len(range_info) >= 3:  # At least min, max, default
                return range_info
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    return None


def test_resolution(cap: cv2.VideoCapture, width: int, height: int) -> bool:
    """
    Test if a camera supports a specific resolution.
    
    Args:
        cap: OpenCV VideoCapture object
        width: Resolution width
        height: Resolution height
    
    Returns:
        True if resolution is supported, False otherwise
    """
    if not cap or not cap.isOpened():
        return False
    
    try:
        # Save current settings
        old_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        old_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Try to set new resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Check if it was actually set
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Restore old settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, old_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, old_height)
        
        # Check if resolution matches (allow small tolerance for rounding)
        return abs(actual_width - width) <= 2 and abs(actual_height - height) <= 2
    except Exception:
        return False


def get_camera_info(index: int, test_resolutions: bool = False, 
                   resolution_list: Optional[List[Tuple[int, int]]] = None,
                   backend: Optional[int] = None) -> CameraInfo:
    """
    Get detailed information about a specific camera.
    
    Args:
        index: Camera index to query
        test_resolutions: If True, test common resolutions (slower)
        resolution_list: Optional list of (width, height) tuples to test
        backend: Optional OpenCV backend to use (None = auto-detect)
    
    Returns:
        CameraInfo object with camera details
    """
    info = CameraInfo(index=index)
    
    # Determine device path on Linux
    if is_linux():
        dev_path = f"/dev/video{index}"
        if os.path.exists(dev_path):
            info.device_path = dev_path
            info.name = get_v4l2_camera_name(dev_path)
    
    # Get backend to use
    if backend is None:
        backend = get_camera_backend()
    
    # Determine backends to try - try multiple on Linux too
    if platform.system() == "Windows":
        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, backend, cv2.CAP_ANY]
    else:
        # On Linux, try V4L2 first, then CAP_ANY as fallback
        backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
    
    # Remove duplicates
    seen = set()
    backends_to_try = [x for x in backends_to_try if not (x in seen or seen.add(x))]
    
    cap = None
    detection_timeout = 3.0 if is_raspberry_pi() else 1.5
    
    # FOURCC formats to test
    fourcc_formats_to_test = ["MJPG", "YUYV", "H264", "YUV2"]
    
    # Try to open camera with each backend
    for backend_alt in backends_to_try:
        try:
            # On Linux/V4L2, use device path if available
            if is_linux() and backend_alt == cv2.CAP_V4L2 and info.device_path:
                cap = cv2.VideoCapture(info.device_path, backend_alt)
            else:
                cap = cv2.VideoCapture(index, backend_alt)
            
            if cap.isOpened():
                # Get backend name
                info.backend = cap.getBackendName()
                
                # Set buffer size to 1 for V4L2 (helps avoid timeouts)
                if is_linux() and backend_alt == cv2.CAP_V4L2:
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except:
                        pass
                
                # Get default properties
                info.default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info.default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info.default_fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Get default FOURCC
                try:
                    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                    info.default_fourcc = fourcc_int_to_string(fourcc_int)
                    # Add default to supported list if valid
                    if info.default_fourcc and info.default_fourcc != "unknown" and info.default_fourcc not in info.supported_fourcc:
                        info.supported_fourcc.append(info.default_fourcc)
                except Exception:
                    info.default_fourcc = "unknown"
                
                # Test different FOURCC formats
                for fourcc_str in fourcc_formats_to_test:
                    if fourcc_str == info.default_fourcc:
                        continue  # Already added
                    try:
                        old_fourcc = cap.get(cv2.CAP_PROP_FOURCC)
                        fourcc_code = cv2.VideoWriter_fourcc(*fourcc_str)
                        cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
                        # Check if it was set
                        actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                        actual_fourcc = fourcc_int_to_string(actual_fourcc_int)
                        # Restore
                        cap.set(cv2.CAP_PROP_FOURCC, old_fourcc)
                        # If format matches or is similar, it's supported
                        if actual_fourcc == fourcc_str or fourcc_str in actual_fourcc:
                            if fourcc_str not in info.supported_fourcc:
                                info.supported_fourcc.append(fourcc_str)
                    except Exception:
                        pass
                
                # Try to read a frame with default settings
                ret, frame = read_frame_with_timeout(cap, timeout=detection_timeout)
                if ret and frame is not None:
                    info.is_available = True
                else:
                    # If default format didn't work, try MJPEG
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                        time.sleep(0.1)  # Brief pause for format change
                        ret, frame = read_frame_with_timeout(cap, timeout=detection_timeout)
                        if ret and frame is not None:
                            info.is_available = True
                            # Update default if MJPEG works better
                            if "MJPG" not in info.supported_fourcc:
                                info.supported_fourcc.insert(0, "MJPG")
                    except Exception:
                        pass
                    
                    # If still no frame, try YUYV
                    if not info.is_available:
                        try:
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
                            time.sleep(0.1)
                            ret, frame = read_frame_with_timeout(cap, timeout=detection_timeout)
                            if ret and frame is not None:
                                info.is_available = True
                                if "YUYV" not in info.supported_fourcc:
                                    info.supported_fourcc.insert(0, "YUYV")
                        except Exception:
                            pass
                
                # Even if frame read failed, if we have valid properties, mark as partially available
                if not info.is_available and info.default_width > 0 and info.default_height > 0:
                    # Mark as available if properties are valid (camera exists even if frame read is slow)
                    info.is_available = True
                
                # Test resolutions if requested (only if camera is available)
                if info.is_available and test_resolutions:
                    test_list = resolution_list or [
                        (640, 480), (800, 600), (1024, 768),
                        (1280, 720), (1280, 960), (1600, 1200),
                        (1920, 1080), (2560, 1440)
                    ]
                    
                    for width, height in test_list:
                        if test_resolution(cap, width, height):
                            if (width, height) not in info.supported_resolutions:
                                info.supported_resolutions.append((width, height))
                    
                    # Sort resolutions
                    info.supported_resolutions.sort(key=lambda x: (x[0] * x[1], x[0]))
                
                # Test common FPS values (only if camera is available)
                if info.is_available and test_resolutions:
                    test_fps = [15.0, 20.0, 24.0, 25.0, 30.0, 60.0]
                    for fps in test_fps:
                        try:
                            old_fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.set(cv2.CAP_PROP_FPS, fps)
                            actual_fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.set(cv2.CAP_PROP_FPS, old_fps)
                            if abs(actual_fps - fps) < 1.0:  # Allow 1 FPS tolerance
                                if fps not in info.supported_fps:
                                    info.supported_fps.append(fps)
                        except Exception:
                            pass
                    info.supported_fps.sort()
                
                # Get Windows camera name if applicable
                if platform.system() == "Windows" and not info.name:
                    info.name = get_windows_camera_name(index)
                
                # If we successfully opened and got info, break out of backend loop
                if info.is_available:
                    # Keep cap open for potential further testing
                    break
                else:
                    # Release and try next backend
                    if cap:
                        cap.release()
                        cap = None
        except Exception as e:
            # Release on exception and try next backend
            if cap:
                try:
                    cap.release()
                except:
                    pass
                cap = None
            continue
    
    # Clean up if cap is still open and we're done
    if cap:
        cap.release()
    
    return info


def list_all_cameras(max_test: int = 10, detailed: bool = False,
                    test_resolutions: bool = False,
                    resolution_list: Optional[List[Tuple[int, int]]] = None) -> List[CameraInfo]:
    """
    List all available cameras with their information.
    
    Args:
        max_test: Maximum camera index to test (default: 10)
        detailed: If True, get detailed info even for unavailable cameras
        test_resolutions: If True, test supported resolutions (slower)
        resolution_list: Optional list of (width, height) tuples to test
    
    Returns:
        List of CameraInfo objects
    """
    print(f"[INFO] Scanning for cameras (max_index={max_test})...")
    
    # First, detect which cameras are available
    available_indices = detect_cameras(max_test=max_test)
    
    cameras = []
    
    # Get info for available cameras
    for idx in available_indices:
        print(f"[INFO] Gathering info for camera {idx}...")
        info = get_camera_info(idx, test_resolutions=test_resolutions,
                             resolution_list=resolution_list)
        cameras.append(info)
    
    # Optionally check additional indices if detailed mode
    if detailed:
        for idx in range(max_test):
            if idx not in available_indices:
                # Try to get basic info even if camera wasn't detected
                # This is useful to see if device exists but isn't accessible
                info = CameraInfo(index=idx)
                if is_linux():
                    dev_path = f"/dev/video{idx}"
                    if os.path.exists(dev_path):
                        info.device_path = dev_path
                        info.name = get_v4l2_camera_name(dev_path)
                        cameras.append(info)
    
    # Sort by index
    cameras.sort(key=lambda x: x.index)
    
    print(f"[INFO] Found {len([c for c in cameras if c.is_available])} available camera(s)")
    return cameras


def print_camera_summary(cameras: List[CameraInfo]) -> None:
    """
    Print a formatted summary of camera information.
    
    Args:
        cameras: List of CameraInfo objects
    """
    print("\n" + "="*70)
    print("CAMERA INFORMATION SUMMARY")
    print("="*70)
    
    for cam in cameras:
        print(f"\n{cam}")
        
        if cam.is_available:
            print(f"  Default: {cam.default_fourcc} {cam.default_width}×{cam.default_height} @ {cam.default_fps:.1f} FPS")
            
            if cam.supported_fourcc:
                print(f"  Supported Formats ({len(cam.supported_fourcc)}): {', '.join(cam.supported_fourcc)}")
            
            if cam.supported_resolutions:
                print(f"  Supported Resolutions ({len(cam.supported_resolutions)}):")
                for w, h in cam.supported_resolutions:
                    marker = "✓" if (w, h) == (cam.default_width, cam.default_height) else " "
                    print(f"    {marker} {w}×{h}")
            
            if cam.supported_fps:
                print(f"  Supported FPS ({len(cam.supported_fps)}): {', '.join([f'{f:.0f}' for f in cam.supported_fps])}")
        
        print()
    
    print("="*70 + "\n")


# Convenience function for quick access
def get_available_camera_indices() -> List[int]:
    """
    Quick helper to get just the list of available camera indices.
    
    Returns:
        List of camera indices
    """
    cameras = list_all_cameras(detailed=False, test_resolutions=False)
    return [cam.index for cam in cameras if cam.is_available]


if __name__ == "__main__":
    """Test/demo script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Camera Information Helper")
    parser.add_argument("--detailed", action="store_true", help="Show detailed info including unavailable cameras")
    parser.add_argument("--test-resolutions", action="store_true", help="Test supported resolutions (slower)")
    parser.add_argument("--max-test", type=int, default=10, help="Maximum camera index to test")
    args = parser.parse_args()
    
    print(f"[INFO] Camera Information Helper")
    print(f"[INFO] Platform: {platform.system()}")
    print(f"[INFO] Raspberry Pi: {is_raspberry_pi()}")
    print()
    
    cameras = list_all_cameras(
        max_test=args.max_test,
        detailed=args.detailed,
        test_resolutions=args.test_resolutions
    )
    
    print_camera_summary(cameras)
    
    # Also print JSON-like output for programmatic use
    print("\nJSON-like output:")
    for cam in cameras:
        print(f"Camera {cam.index}: {cam.to_dict()}")

