"""
Calibration Module
------------------
Video calibration calculation logic and coordinate calculation utilities.
"""

from .video_calibrator import VideoCalibrator
from .utils import (
    extract_working_distance,
    extract_pixels_per_mm,
    # New function names
    calculate_radial_distance_from_center_px,
    calculate_radial_distance_from_center_mm,
    calculate_radial_distance_and_xy_from_pair,
    # Backward compatibility aliases
    calculate_b_px,
    calculate_b_mm,
    calculate_xy_mm,
    calculate_b_xy_from_pair
)

__all__ = [
    'VideoCalibrator',
    # Extraction functions
    'extract_working_distance',
    'extract_pixels_per_mm',
    # New function names
    'calculate_radial_distance_from_center_px',
    'calculate_radial_distance_from_center_mm',
    'calculate_radial_distance_and_xy_from_pair',
    # Backward compatibility aliases
    'calculate_b_px',
    'calculate_b_mm',
    'calculate_xy_mm',
    'calculate_b_xy_from_pair'
]

