import os
import time
import sys


def ts_name(stem: str, ext: str) -> str:
    """Return stem_YYYYmmdd_HHMMSS.ext"""
    return f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.{ext}"


def path_stem(path: str) -> str:
    """Return directory + filename-without-extension."""
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return os.path.join(os.path.dirname(path), stem)


# New: centralized output folder handling
BASE_OUTPUT_DIR = "pair_detect_output"


def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def make_session_output_dir(video_path: str) -> str:
    """Create and return output dir: pair_detect_output/<video_stem>_<YYYYmmdd_HHMMSS>/"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(BASE_OUTPUT_DIR, f"{base_name}_{stamp}")
    ensure_dir(out_dir)
    return out_dir


def export_paths_for(video_path: str) -> dict:
    """Return dict with session dir and standard file paths inside it."""
    out_dir = make_session_output_dir(video_path)
    return {
        "dir": out_dir,
        "tracked_mp4": os.path.join(out_dir, "tracked_export.mp4"),
        "binary_mp4": os.path.join(out_dir, "binary_overlay_export.mp4"),
        "pairs_csv": os.path.join(out_dir, "pairs.csv"),
    }


# Capture output directory - relative to script location
def get_script_dir():
    """Get the directory where the main script is located."""
    if getattr(sys, 'frozen', False):
        # If running as compiled executable
        return os.path.dirname(sys.executable)
    else:
        # If running as script
        return os.path.dirname(os.path.abspath(sys.argv[0]))

# Get script directory and create capture_output path relative to it
_script_dir = get_script_dir()
CAPTURE_OUTPUT_DIR = os.path.join(_script_dir, "capture_output")


def make_capture_output_path(width: int, height: int, fps: int) -> str:
    """
    Create output path for captured video.
    Returns: capture_output/video_WxH_YYYYmmdd_HHMMSS.mp4
    """
    ensure_dir(CAPTURE_OUTPUT_DIR)
    name = ts_name(f"video_{width}x{height}_{fps}fps", "mp4")
    path = os.path.join(CAPTURE_OUTPUT_DIR, name)
    print(f"[DEBUG] Video will be saved to: {os.path.abspath(path)}")
    return path


def make_capture_frame_path(width: int, height: int) -> str:
    """
    Create output path for captured frame.
    Returns: capture_output/frame_WxH_YYYYmmdd_HHMMSS.png
    """
    ensure_dir(CAPTURE_OUTPUT_DIR)
    name = ts_name(f"frame_{width}x{height}", "png")
    path = os.path.join(CAPTURE_OUTPUT_DIR, name)
    print(f"[DEBUG] Frame will be saved to: {os.path.abspath(path)}")
    return path


