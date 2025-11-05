"""
Standard GUI Styles
-------------------
Standardized window geometries, themes, and styling constants for consistent GUI appearance.
"""

import tkinter as tk
from tkinter import ttk
from typing import Tuple, Optional
from enum import Enum


class WindowSize(Enum):
    """Standard window size presets."""
    SMALL = (400, 550)      # Small dialogs, simple tools (e.g., image calibration)
    MEDIUM = (540, 950)     # Medium controls (e.g., pair detector)
    LARGE = (1400, 900)     # Large visualization tools
    XLARGE = (2000, 900)    # Extra large multi-panel tools (e.g., video calibration)
    CAPTURE = (1000, 700)   # Camera capture window


# Standard size mapping for easy access
STANDARD_SIZES = {
    "small": WindowSize.SMALL,
    "medium": WindowSize.MEDIUM,
    "large": WindowSize.LARGE,
    "xlarge": WindowSize.XLARGE,
    "capture": WindowSize.CAPTURE,
}


def apply_standard_theme(root: tk.Tk) -> None:
    """
    Apply the standard 'clam' theme to a tkinter root window.
    
    Args:
        root: The tkinter root window to apply theme to.
    """
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass  # Theme not available, continue without it


def format_window_title(app_name: str, platform: Optional[str] = None, version: Optional[str] = None) -> str:
    """
    Format a standardized window title.
    
    Args:
        app_name: The main application name (e.g., "Image Calibration Tool")
        platform: Optional platform identifier (e.g., "Raspberry Pi", "Windows")
        version: Optional version string (e.g., "v4.5")
    
    Returns:
        Formatted window title string.
    
    Examples:
        >>> format_window_title("Image Calibration Tool")
        "Image Calibration Tool"
        >>> format_window_title("Image Calibration Tool", "Raspberry Pi")
        "Image Calibration Tool - Raspberry Pi"
        >>> format_window_title("Pair Detector", version="v4.5")
        "Pair Detector v4.5"
    """
    title = app_name
    if version:
        title += f" {version}"
    if platform:
        title += f" - {platform}"
    return title


def get_standard_size(size_name: str) -> Tuple[int, int]:
    """
    Get standard window size dimensions.
    
    Args:
        size_name: One of "small", "medium", "large", "xlarge", "capture"
    
    Returns:
        Tuple of (width, height)
    
    Raises:
        KeyError: If size_name is not a valid standard size.
    """
    size_enum = STANDARD_SIZES.get(size_name.lower())
    if size_enum is None:
        raise KeyError(f"Unknown standard size: {size_name}. Valid options: {list(STANDARD_SIZES.keys())}")
    return size_enum.value


# Standard padding constants
STANDARD_PADDING = {
    "small": 3,      # Small padding for compact controls
    "medium": 5,     # Medium padding for most frames
    "large": 10,     # Large padding for main content areas
    "xlarge": 15,    # Extra large padding for prominent sections
}

