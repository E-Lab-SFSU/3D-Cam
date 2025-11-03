"""
Calibration utilities for extracting working distance and other calibration parameters.
XYZ-coordinate calculation utilities for B, X, Y, Z coordinate transformations.
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


def calculate_b_px(r_a: float, r_c: float) -> Optional[float]:
    """
    Calculate B point in pixels from inner and outer radii.
    
    Formula: B = (2*A*C)/(A+C)
    Where A = inner radius (r_a), C = outer radius (r_c)
    
    Args:
        r_a: Inner radius A in pixels
        r_c: Outer radius C in pixels
        
    Returns:
        B in pixels, or None if calculation is invalid
    """
    if r_a <= 0 or r_c <= 0:
        return None
    
    if r_a + r_c == 0:
        return None
    
    b_px = (2 * r_a * r_c) / (r_a + r_c)
    return b_px


def calculate_b_mm(b_px: float, pixels_per_mm: float) -> Optional[float]:
    """
    Convert B from pixels to millimeters.
    
    Args:
        b_px: B value in pixels
        pixels_per_mm: Conversion factor (pixels per millimeter)
        
    Returns:
        B in millimeters, or None if conversion is invalid
    """
    if b_px is None or b_px <= 0:
        return None
    
    if pixels_per_mm is None or pixels_per_mm <= 0:
        return None
    
    b_mm = b_px / pixels_per_mm
    return b_mm


def calculate_xy_mm(midpoint_x: float, midpoint_y: float, 
                   x_center: float, y_center: float, 
                   b_mm: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate X and Y coordinates in millimeters from pair midpoint and B distance.
    
    Formula:
        dx = midpoint_x - x_center
        dy = midpoint_y - y_center
        theta_rad = atan2(dy, dx)
        X_mm = B_mm * cos(theta_rad)
        Y_mm = -B_mm * sin(theta_rad)  # Negative because image Y increases downward
    
    Args:
        midpoint_x: X coordinate of pair midpoint in pixels
        midpoint_y: Y coordinate of pair midpoint in pixels
        x_center: X coordinate of optical center in pixels
        y_center: Y coordinate of optical center in pixels
        b_mm: B distance in millimeters
        
    Returns:
        Tuple of (X_mm, Y_mm) in millimeters, or (None, None) if calculation is invalid
    """
    if b_mm is None or b_mm <= 0:
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
    x_mm = b_mm * math.cos(theta_rad)
    y_mm = -b_mm * math.sin(theta_rad)
    
    return (x_mm, y_mm)


def calculate_b_xy_from_pair(xi: float, yi: float, xj: float, yj: float,
                             r_a: float, r_c: float,
                             x_center: float, y_center: float,
                             pixels_per_mm: Optional[float] = None) -> Dict[str, Optional[float]]:
    """
    Calculate B (pixels and mm), X_mm, and Y_mm from a pair's coordinates and radii.
    
    This is a convenience function that combines all calculations for a complete pair.
    
    Args:
        xi: X coordinate of point A (inner point)
        yi: Y coordinate of point A (inner point)
        xj: X coordinate of point C (outer point)
        yj: Y coordinate of point C (outer point)
        r_a: Inner radius A in pixels
        r_c: Outer radius C in pixels
        x_center: X coordinate of optical center in pixels
        y_center: Y coordinate of optical center in pixels
        pixels_per_mm: Optional conversion factor (required for B_mm, X_mm, Y_mm)
        
    Returns:
        Dictionary with keys: 'b_px', 'b_mm', 'x_mm', 'y_mm'
        Values are None if calculation is invalid or pixels_per_mm is not provided
    """
    result = {
        'b_px': None,
        'b_mm': None,
        'x_mm': None,
        'y_mm': None
    }
    
    # Calculate B in pixels
    b_px = calculate_b_px(r_a, r_c)
    if b_px is None:
        return result
    
    result['b_px'] = b_px
    
    # If pixels_per_mm is provided, calculate mm values
    if pixels_per_mm is not None and pixels_per_mm > 0:
        # Calculate B in mm
        b_mm = calculate_b_mm(b_px, pixels_per_mm)
        result['b_mm'] = b_mm
        
        if b_mm is not None:
            # Calculate pair midpoint
            midpoint_x = 0.5 * (xi + xj)
            midpoint_y = 0.5 * (yi + yj)
            
            # Calculate X and Y in mm
            x_mm, y_mm = calculate_xy_mm(midpoint_x, midpoint_y, x_center, y_center, b_mm)
            result['x_mm'] = x_mm
            result['y_mm'] = y_mm
    
    return result

