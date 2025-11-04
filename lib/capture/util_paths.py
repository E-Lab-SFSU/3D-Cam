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
    # Get the folder containing the video
    video_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Generate filenames with counter suffix if multiple exports exist
    def get_unique_path(base_suffix: str, ext: str) -> str:
        """Get unique path by appending -1, -2, etc. if file already exists."""
        base_path = os.path.join(video_dir, f"{base_name}{base_suffix}.{ext}")
        if not os.path.exists(base_path):
            return base_path
        
        counter = 1
        while True:
            numbered_path = os.path.join(video_dir, f"{base_name}{base_suffix}-{counter}.{ext}")
            if not os.path.exists(numbered_path):
                return numbered_path
            counter += 1
    
    # Use the same folder as the video for all outputs
    return {
        "dir": video_dir,
        "tracked_mp4": get_unique_path("-grayscale", "mp4"),  # Grayscale output
        "binary_mp4": get_unique_path("-binary", "mp4"),  # Binary output
        "pairs_csv": get_unique_path("-paired-tracked", "csv"),
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

# Get script directory and create inputs_outputs/videos path relative to it
_script_dir = get_script_dir()
CAPTURE_OUTPUT_DIR = os.path.join(_script_dir, "inputs_outputs/videos")


def make_capture_output_path(width: int, height: int, fps: int, prefix: str = None) -> str:
    """
    Create output path for captured video in its own folder.
    Returns: inputs_outputs/<prefix>_YYYYmmdd_HHMMSS/<prefix>_YYYYmmdd_HHMMSS.mp4
    or if no prefix: inputs_outputs/video_WxH_YYYYmmdd_HHMMSS/video_WxH_YYYYmmdd_HHMMSS.mp4
    
    Args:
        width: Video width
        height: Video height
        fps: Video FPS
        prefix: Optional prefix for folder and filename (if None, uses default naming)
    """
    if prefix:
        # Sanitize prefix (remove invalid filename characters)
        import re
        prefix = re.sub(r'[<>:"/\\|?*]', '_', prefix).strip()
        if not prefix:
            prefix = None  # If prefix becomes empty after sanitization, use default
    
    if prefix:
        base_name = ts_name(prefix, "")
    else:
        base_name = ts_name(f"video_{width}x{height}_{fps}fps", "")
    
    # Remove trailing period if present (from ts_name when ext is empty)
    base_name = base_name.rstrip(".")
    # Use clean base_name for folder (no periods after stripping)
    folder_path = os.path.join("inputs_outputs", base_name)
    ensure_dir(folder_path)
    
    # Remove any double periods from filename
    video_filename = base_name.replace("..", ".") + ".mp4"
    path = os.path.join(folder_path, video_filename)
    print(f"[DEBUG] Video will be saved to: {os.path.abspath(path)}")
    return path


def make_capture_frame_path(width: int, height: int, prefix: str = None) -> str:
    """
    Create output path for captured frame in its own folder.
    Returns: inputs_outputs/<prefix>_YYYYmmdd_HHMMSS/<prefix>_YYYYmmdd_HHMMSS.png
    or if no prefix: inputs_outputs/frame_WxH_YYYYmmdd_HHMMSS/frame_WxH_YYYYmmdd_HHMMSS.png
    
    Args:
        width: Frame width
        height: Frame height
        prefix: Optional prefix for folder and filename (if None, uses default naming)
    """
    if prefix:
        # Sanitize prefix (remove invalid filename characters)
        import re
        prefix = re.sub(r'[<>:"/\\|?*]', '_', prefix).strip()
        if not prefix:
            prefix = None  # If prefix becomes empty after sanitization, use default
    
    if prefix:
        base_name = ts_name(prefix, "")
    else:
        base_name = ts_name(f"frame_{width}x{height}", "")
    
    # Remove trailing period if present (from ts_name when ext is empty)
    base_name = base_name.rstrip(".")
    # Use clean base_name for folder (no periods after stripping)
    folder_path = os.path.join("inputs_outputs", base_name)
    ensure_dir(folder_path)
    
    # Remove any double periods from filename
    frame_filename = base_name.replace("..", ".") + ".png"
    path = os.path.join(folder_path, frame_filename)
    print(f"[DEBUG] Frame will be saved to: {os.path.abspath(path)}")
    return path


