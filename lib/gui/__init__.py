"""
Shared GUI Components
---------------------
Common GUI utilities, styles, and widgets for all 3D-Cam applications.
"""

from .styles import (
    WindowSize,
    STANDARD_SIZES,
    apply_standard_theme,
    format_window_title,
    get_standard_size,
    STANDARD_PADDING,
)
from .common import tooltip, ScrollableFrame

__all__ = [
    "WindowSize",
    "STANDARD_SIZES",
    "apply_standard_theme",
    "format_window_title",
    "get_standard_size",
    "STANDARD_PADDING",
    "tooltip",
    "ScrollableFrame",
]

