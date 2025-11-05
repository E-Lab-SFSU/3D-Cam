"""
Calibration Utilities
---------------------
Utilities for extracting calibration parameters from JSON files and 
calculating 3D coordinates from calibration data.

Includes:
- Calibration data extraction (working_distance, pixels_per_mm)
- Coordinate calculations (radial_distance_from_center, X, Y, Z) using calibration parameters
"""

from typing import Optional, Dict, Tuple
import math


def extract_working_distance(cal_data: Dict) -> Optional[float]:
    """
    Extract working distance from calibration data dictionary.
    Checks multiple possible locations in order:
    1. Top level "working_distance_mm"
    2. First entry in "data_points" array
    3. "camera_parameters.working_distance_mm"
    
    Args:
        cal_data: Dictionary containing calibration data
        
    Returns:
        Working distance in mm, or None if not found
    """
    # 1. Top level
    if "working_distance_mm" in cal_data:
        try:
            return float(cal_data["working_distance_mm"])
        except (ValueError, TypeError):
            pass
    
    # 2. From data_points array (first entry)
    if "data_points" in cal_data and isinstance(cal_data["data_points"], list):
        if len(cal_data["data_points"]) > 0:
            first_point = cal_data["data_points"][0]
            if isinstance(first_point, dict) and "working_distance_mm" in first_point:
                try:
                    return float(first_point["working_distance_mm"])
                except (ValueError, TypeError):
                    pass
    
    # 3. From camera_parameters if it exists
    if "camera_parameters" in cal_data:
        cam_params = cal_data["camera_parameters"]
        if isinstance(cam_params, dict) and "working_distance_mm" in cam_params:
            try:
                return float(cam_params["working_distance_mm"])
            except (ValueError, TypeError):
                pass
    
    return None


def extract_pixels_per_mm(cal_data: Dict) -> Optional[float]:
    """
    Extract pixels_per_mm from calibration data dictionary.
    
    Args:
        cal_data: Dictionary containing calibration data
        
    Returns:
        Pixels per mm value, or None if not found
    """
    if "pixels_per_mm" in cal_data:
        try:
            pixels_per_mm = float(cal_data["pixels_per_mm"])
            if pixels_per_mm > 0:
                return pixels_per_mm
        except (ValueError, TypeError):
            pass
    
    return None


def calculate_radial_distance_from_center_px(inner_radius_px: float, outer_radius_px: float) -> Optional[float]:
    """
    Calculate radial distance from optical center axis in pixels.
    
    Formula: radial_distance = (2 * inner_radius * outer_radius) / (inner_radius + outer_radius)
    Where inner_radius = distance from optical center to inner point (direct view)
          outer_radius = distance from optical center to outer point (mirrored view)
    
    Args:
        inner_radius_px: Inner radius (A) in pixels
        outer_radius_px: Outer radius (C) in pixels
        
    Returns:
        Radial distance from center in pixels, or None if calculation is invalid
    """
    if inner_radius_px <= 0 or outer_radius_px <= 0:
        return None
    
    if inner_radius_px + outer_radius_px == 0:
        return None
    
    radial_distance_px = (2 * inner_radius_px * outer_radius_px) / (inner_radius_px + outer_radius_px)
    return radial_distance_px


# Backward compatibility alias
def calculate_b_px(r_a: float, r_c: float) -> Optional[float]:
    """Backward compatibility alias for calculate_radial_distance_from_center_px."""
    return calculate_radial_distance_from_center_px(r_a, r_c)


def calculate_radial_distance_from_center_mm(radial_distance_px: float, pixels_per_mm: float) -> Optional[float]:
    """
    Convert radial distance from pixels to millimeters.
    
    Args:
        radial_distance_px: Radial distance from center in pixels
        pixels_per_mm: Conversion factor (pixels per millimeter)
        
    Returns:
        Radial distance from center in millimeters, or None if conversion is invalid
    """
    if radial_distance_px is None or radial_distance_px <= 0:
        return None
    
    if pixels_per_mm is None or pixels_per_mm <= 0:
        return None
    
    radial_distance_mm = radial_distance_px / pixels_per_mm
    return radial_distance_mm


# Backward compatibility alias
def calculate_b_mm(b_px: float, pixels_per_mm: float) -> Optional[float]:
    """Backward compatibility alias for calculate_radial_distance_from_center_mm."""
    return calculate_radial_distance_from_center_mm(b_px, pixels_per_mm)


def calculate_xy_mm(midpoint_x: float, midpoint_y: float, 
                   x_center: float, y_center: float, 
                   radial_distance_mm: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate X and Y coordinates in millimeters from pair midpoint and radial distance.
    
    Formula:
        dx = midpoint_x - x_center
        dy = midpoint_y - y_center
        theta_rad = atan2(dy, dx)
        X_mm = radial_distance_mm * cos(theta_rad)
        Y_mm = -radial_distance_mm * sin(theta_rad)  # Negative because image Y increases downward
    
    Args:
        midpoint_x: X coordinate of pair midpoint in pixels
        midpoint_y: Y coordinate of pair midpoint in pixels
        x_center: X coordinate of optical center in pixels
        y_center: Y coordinate of optical center in pixels
        radial_distance_mm: Radial distance from center in millimeters
        
    Returns:
        Tuple of (X_mm, Y_mm) in millimeters, or (None, None) if calculation is invalid
    """
    if radial_distance_mm is None or radial_distance_mm <= 0:
        return (None, None)
    
    # Calculate direction vector from center to midpoint
    dx = midpoint_x - x_center
    dy = midpoint_y - y_center
    dist_to_midpoint = math.sqrt(dx*dx + dy*dy)
    
    if dist_to_midpoint < 1e-6:
        return (None, None)  # Midpoint is at center
    
    # Calculate angle in radians (atan2 gives angle from positive x-axis)
    # Note: In image coordinates, Y increases downward, so we negate Y in the final calculation
    theta_rad = math.atan2(dy, dx)
    
    # Convert polar to Cartesian: X = r * cos(theta), Y = -r * sin(theta)
    # Y is negated because image coordinates have Y increasing downward
    x_mm = radial_distance_mm * math.cos(theta_rad)
    y_mm = -radial_distance_mm * math.sin(theta_rad)
    
    return (x_mm, y_mm)


def calculate_radial_distance_and_xy_from_pair(inner_point_x: float, inner_point_y: float, 
                                                outer_point_x: float, outer_point_y: float,
                                                inner_radius_px: float, outer_radius_px: float,
                                                x_center: float, y_center: float,
                                                pixels_per_mm: Optional[float] = None) -> Dict[str, Optional[float]]:
    """
    Calculate radial distance (pixels and mm), X_mm, and Y_mm from a pair's coordinates and radii.
    
    This is a convenience function that combines all calculations for a complete pair.
    
    Args:
        inner_point_x: X coordinate of inner point (direct view) in pixels
        inner_point_y: Y coordinate of inner point (direct view) in pixels
        outer_point_x: X coordinate of outer point (mirrored view) in pixels
        outer_point_y: Y coordinate of outer point (mirrored view) in pixels
        inner_radius_px: Inner radius in pixels
        outer_radius_px: Outer radius in pixels
        x_center: X coordinate of optical center in pixels
        y_center: Y coordinate of optical center in pixels
        pixels_per_mm: Optional conversion factor (required for mm values)
        
    Returns:
        Dictionary with keys: 'radial_distance_px', 'radial_distance_mm', 'x_mm', 'y_mm'
        Values are None if calculation is invalid or pixels_per_mm is not provided
    """
    result = {
        'radial_distance_px': None,
        'radial_distance_mm': None,
        'x_mm': None,
        'y_mm': None
    }
    
    # Calculate radial distance in pixels
    radial_distance_px = calculate_radial_distance_from_center_px(inner_radius_px, outer_radius_px)
    if radial_distance_px is None:
        return result
    
    result['radial_distance_px'] = radial_distance_px
    
    # If pixels_per_mm is provided, calculate mm values
    if pixels_per_mm is not None and pixels_per_mm > 0:
        # Calculate radial distance in mm
        radial_distance_mm = calculate_radial_distance_from_center_mm(radial_distance_px, pixels_per_mm)
        result['radial_distance_mm'] = radial_distance_mm
        
        if radial_distance_mm is not None:
            # Calculate pair midpoint
            midpoint_x = 0.5 * (inner_point_x + outer_point_x)
            midpoint_y = 0.5 * (inner_point_y + outer_point_y)
            
            # Calculate X and Y in mm
            x_mm, y_mm = calculate_xy_mm(midpoint_x, midpoint_y, x_center, y_center, radial_distance_mm)
            result['x_mm'] = x_mm
            result['y_mm'] = y_mm
    
    return result


# Backward compatibility alias
def calculate_b_xy_from_pair(xi: float, yi: float, xj: float, yj: float,
                             r_a: float, r_c: float,
                             x_center: float, y_center: float,
                             pixels_per_mm: Optional[float] = None) -> Dict[str, Optional[float]]:
    """
    Backward compatibility alias for calculate_radial_distance_and_xy_from_pair.
    Note: Returns dict with old keys 'b_px', 'b_mm' for compatibility.
    """
    result = calculate_radial_distance_and_xy_from_pair(
        xi, yi, xj, yj, r_a, r_c, x_center, y_center, pixels_per_mm
    )
    
    # Map new keys to old keys for backward compatibility
    compat_result = {
        'b_px': result.get('radial_distance_px'),
        'b_mm': result.get('radial_distance_mm'),
        'x_mm': result.get('x_mm'),
        'y_mm': result.get('y_mm')
    }
    return compat_result
