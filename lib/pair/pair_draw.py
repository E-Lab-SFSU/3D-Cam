from typing import List, Tuple, Optional
import math
import random
import colorsys
import hashlib

import cv2
import numpy as np

_pair_color_cache: dict = {}
_current_video_seed: Optional[int] = None


def set_video_seed(video_path: str):
    """Set the seed based on video filename for persistent colors per video."""
    global _current_video_seed, _pair_color_cache
    # Hash the filename to get a consistent integer seed
    filename_hash = int(hashlib.md5(video_path.encode('utf-8')).hexdigest()[:8], 16)
    if _current_video_seed != filename_hash:
        _current_video_seed = filename_hash
        _pair_color_cache.clear()  # Clear cache when video changes


def _get_pair_color(track_id: int, video_path: Optional[str] = None) -> Tuple[int, int, int]:
    """
    Return a persistent seeded color for a track_id, seeded by video filename.
    Same video filename will always produce same colors for same track IDs.
    """
    if video_path:
        set_video_seed(video_path)
    
    # Use combined seed: video filename + track_id
    if _current_video_seed is None:
        # Fallback if no video set
        seed = track_id
    else:
        # Combine video seed with track_id for unique colors per track, consistent per video
        seed = (_current_video_seed ^ track_id) & 0xFFFFFFFF
    
    if seed not in _pair_color_cache:
        rng = random.Random(seed)
        # Generate bright, saturated colors
        h = rng.uniform(0, 360)
        s = rng.uniform(0.6, 1.0)
        v = rng.uniform(0.7, 1.0)
        # HSV to RGB, then to BGR
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
        _pair_color_cache[seed] = (int(b * 255), int(g * 255), int(r * 255))
    return _pair_color_cache[seed]


def draw_blob_boxes(dst, blobs: List[dict]):
    for b in blobs:
        x, y, w, h = b["box"]
        cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 255, 0), 1)


def draw_center(dst, cx: int, cy: int):
    cv2.circle(dst, (cx, cy), 6, (0, 255, 255), 2)
    cv2.line(dst, (cx - 10, cy), (cx + 10, cy), (0, 255, 255), 1)
    cv2.line(dst, (cx, cy - 10), (cx, cy + 10), (0, 255, 255), 1)


def draw_pair_centers(dst, pairs: List[Tuple], label_mode: str = "Red/Blue", video_path: Optional[str] = None):
    """Draw small circles at pair midpoints (used for tracking visualization)."""
    for (pid, xi, yi, xj, yj, *_rest) in pairs:
        mx = int(0.5 * (xi + xj))
        my = int(0.5 * (yi + yj))
        if label_mode == "Random":
            col = _get_pair_color(pid, video_path)
        else:
            col = (0, 255, 0)  # Green by default
        cv2.circle(dst, (mx, my), 3, col, -1)


def draw_pair_lines(dst, pairs: List[Tuple], show_labels: bool = False, label_mode: str = "Red/Blue", video_path: Optional[str] = None, show_pair_points: bool = True):
    """
    label_mode: "None", "Red/Blue", "Random"
    show_labels: if True, show #A/#C text labels
    show_pair_points: if True, draw circles at pair endpoints (A and C points)
    """
    for (pid, xi, yi, xj, yj, *_rest) in pairs:
        line_color = (255, 255, 255)
        colA = colC = (255, 255, 255)

        if label_mode == "Red/Blue":
            colA = (0, 0, 255)  # Red
            colC = (255, 0, 0)  # Blue
            line_color = (255, 255, 255)
        elif label_mode == "Random":
            pair_col = _get_pair_color(pid, video_path)
            line_color = pair_col
            colA = colC = pair_col

        cv2.line(dst, (xi, yi), (xj, yj), line_color, 1)
        if show_pair_points:
            cv2.circle(dst, (xi, yi), 3, colA, -1)
            cv2.circle(dst, (xj, yj), 3, colC, -1)
        if show_labels:
            # Calculate midpoint for consistent label positioning
            mx = int(0.5 * (xi + xj))
            my = int(0.5 * (yi + yj))
            # Place A/C labels at bottom row (will be stacked with other labels)
            # Line height is approximately 12 pixels for font_scale 0.4
            line_height = 12
            bottom_y = my + 8  # Bottom row position
            # Place labels at midpoint, bottom row
            cv2.putText(dst, f"{pid}A/{pid}C", (mx + 4, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colA, 1)


def draw_pair_rays_toward_center(dst, pairs: List[Tuple], frame_width: int, xCenter: int, yCenter: int, label_mode: str = "Red/Blue", video_path: Optional[str] = None):
    ray_len = int(frame_width * (2 / 3))
    for (pid, xi, yi, xj, yj, *_rest) in pairs:
        AB = np.array([xj - xi, yj - yi], dtype=float)
        L = np.linalg.norm(AB)
        if L < 1e-6:
            continue
        u = AB / L
        mid = np.array([(xi + xj) / 2, (yi + yj) / 2], dtype=float)

        AC = np.array([xCenter - xi, yCenter - yi], dtype=float)
        t = AC.dot(u)
        P = np.array([xi, yi], dtype=float) + t * u
        sgn = np.sign((P - mid).dot(u)) or 1.0

        endA = np.array([xi, yi], dtype=float)
        endC = np.array([xj, yj], dtype=float)
        if (endA - P).dot(u) * sgn > (endC - P).dot(u) * sgn:
            anchor = endA
        else:
            anchor = endC

        end = (anchor + sgn * ray_len * u).astype(int)
        ray_color = (255, 255, 255)
        if label_mode == "Random":
            ray_color = _get_pair_color(pid, video_path)
        cv2.line(dst, (int(anchor[0]), int(anchor[1])), (int(end[0]), int(end[1])), ray_color, 1)


def draw_z_values(dst, pairs: List[Tuple], 
                  working_distance_mm: Optional[float] = None,
                  magic_constant: Optional[float] = None,
                  magic_offset: Optional[float] = None,
                  label_mode: str = "Red/Blue",
                  video_path: Optional[str] = None):
    """
    Draw Z value text labels on pairs. Shows "??" if Z cannot be calculated.
    
    Args:
        dst: Destination image (BGR)
        pairs: List of pair tuples (pid, xi, yi, xj, yj, th_i, r_i, th_j, r_j, score, ...)
        working_distance_mm: Working distance in mm for Z calculation
        magic_constant: Magic constant for Z calculation (optional)
        magic_offset: Magic offset for Z calculation (optional)
        label_mode: Color mode for text ("None", "Red/Blue", "Random")
        video_path: Optional video path for consistent colors
    """
    if working_distance_mm is None or working_distance_mm <= 0:
        return  # Cannot calculate Zprime without working distance
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    
    for pair in pairs:
        # Unpack pair: (pid, xi, yi, xj, yj, th_i, r_i, th_j, r_j, score, ...)
        if len(pair) < 10:
            continue
        pid, xi, yi, xj, yj, th_i, r_i, th_j, r_j = pair[0:9]
        
        r_a = r_i  # A = inner radius
        r_c = r_j  # C = outer radius
        
        if r_a <= 0 or r_c <= 0:
            continue
        
        # Calculate Zprime = working_distance * (C-A)/(A+C)
        zprime_val = working_distance_mm * (r_c - r_a) / (r_a + r_c)
        
        # Calculate Z = Zprime * magic_constant + magic_offset
        # Only if magic_constant and magic_offset are available
        z_val = None
        if magic_constant is not None and magic_offset is not None:
            z_val = zprime_val * magic_constant + magic_offset
        
        # Determine text color based on label_mode
        text_color = (255, 255, 255)  # White default
        if label_mode == "Red/Blue":
            text_color = (255, 255, 255)  # White
        elif label_mode == "Random":
            text_color = _get_pair_color(pid, video_path)
        
        # Draw Z value at the midpoint of the pair line
        # Stack labels vertically: A/C at bottom, Z above, X/Y at top
        mx = int(0.5 * (xi + xj))
        my = int(0.5 * (yi + yj))
        line_height = 12  # Approximate line height for font_scale 0.4
        bottom_y = my + 8  # Bottom row (A/C labels go here)
        
        if z_val is not None:
            z_text = f"Z:{z_val:.2f}"
        else:
            z_text = "Z:??"
        # Place Z value one line above A/C labels
        cv2.putText(dst, z_text, (mx + 4, bottom_y - line_height), font, font_scale, text_color, thickness)


def draw_xy_values(dst, pairs: List[Tuple],
                   xCenter: int, yCenter: int,
                   working_distance_mm: Optional[float] = None,
                   pixels_per_mm: Optional[float] = None,
                   label_mode: str = "Red/Blue",
                   video_path: Optional[str] = None):
    """
    Draw X and Y value text labels on pairs. Shows "??" if X or Y cannot be calculated.
    
    Args:
        dst: Destination image (BGR)
        pairs: List of pair tuples (pid, xi, yi, xj, yj, th_i, r_i, th_j, r_j, score, ...)
        xCenter: X coordinate of optical center
        yCenter: Y coordinate of optical center
        working_distance_mm: Working distance in mm for B calculation (optional)
        pixels_per_mm: Pixels per mm conversion factor for X/Y calculation (optional)
        label_mode: Color mode for text ("None", "Red/Blue", "Random")
        video_path: Optional video path for consistent colors
    """
    if pixels_per_mm is None or pixels_per_mm <= 0:
        return  # Cannot convert to mm without pixels_per_mm
    
    if working_distance_mm is None or working_distance_mm <= 0:
        return  # Cannot calculate B without working distance
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1
    
    for pair in pairs:
        # Unpack pair: (pid, xi, yi, xj, yj, th_i, r_i, th_j, r_j, score, ...)
        if len(pair) < 10:
            continue
        pid, xi, yi, xj, yj, th_i, r_i, th_j, r_j = pair[0:9]
        
        r_a = r_i  # A = inner radius
        r_c = r_j  # C = outer radius
        
        if r_a <= 0 or r_c <= 0:
            continue
        
        # Calculate B in pixels: B = (2*A*C)/(A+C)
        b_px = (2 * r_a * r_c) / (r_a + r_c)
        
        # Calculate pair midpoint
        mx = 0.5 * (xi + xj)
        my = 0.5 * (yi + yj)
        
        # Calculate direction vector from center to pair midpoint
        dx = mx - xCenter
        dy = my - yCenter
        dist_to_midpoint = math.sqrt(dx*dx + dy*dy)
        
        if dist_to_midpoint < 1e-6:
            continue  # Skip if midpoint is at center
        
        # Calculate angle in radians (atan2 gives angle from positive x-axis)
        # Note: In image coordinates, Y increases downward, so we negate Y
        theta_rad = math.atan2(dy, dx)
        
        # Convert B from pixels to mm
        b_mm = b_px / pixels_per_mm
        
        # Convert polar to Cartesian: X = r * cos(theta), Y = -r * sin(theta)
        # Y is negated because image coordinates have Y increasing downward
        x_mm = b_mm * math.cos(theta_rad)
        y_mm = -b_mm * math.sin(theta_rad)
        
        # Determine text color based on label_mode
        text_color = (255, 255, 255)  # White default
        if label_mode == "Red/Blue":
            text_color = (255, 255, 255)  # White
        elif label_mode == "Random":
            text_color = _get_pair_color(pid, video_path)
        
        # Draw X and Y values at the midpoint of the pair line
        # Stack labels vertically from bottom to top: A/C (bottom), Z, Y, X (top)
        # Convert coordinates to integers for OpenCV
        mx_int = int(mx)
        my_int = int(my)
        line_height = 11  # Approximate line height for font_scale 0.35
        z_line_height = 12  # Line height for Z value (font_scale 0.4)
        bottom_y = my_int + 8  # Bottom row (A/C labels go here)
        
        # Stack order from bottom: A/C (bottom_y), Z (one line up), Y (two lines up), X (three lines up)
        # Place Y value above Z value
        y_y_pos = bottom_y - z_line_height - line_height  # Two lines above bottom (above Z)
        # Place X value above Y value
        x_y_pos = bottom_y - z_line_height - 2 * line_height  # Three lines above bottom (top)
        
        cv2.putText(dst, f"X:{x_mm:.2f}", (mx_int + 4, x_y_pos), font, font_scale, text_color, thickness)
        cv2.putText(dst, f"Y:{y_mm:.2f}", (mx_int + 4, y_y_pos), font, font_scale, text_color, thickness)


def draw_real_point(dst, pairs: List[Tuple],
                    xCenter: int, yCenter: int,
                    working_distance_mm: Optional[float] = None,
                    pixels_per_mm: Optional[float] = None,
                    label_mode: str = "Red/Blue",
                    video_path: Optional[str] = None):
    """
    Draw B point at B pixels distance from optical center along ray toward pair.
    The point represents the calculated B value: B = (2*A*C)/(A+C) in pixels.
    
    Args:
        dst: Destination image (BGR)
        pairs: List of pair tuples (pid, xi, yi, xj, yj, th_i, r_i, th_j, r_j, score, ...)
        xCenter: X coordinate of optical center
        yCenter: Y coordinate of optical center
        working_distance_mm: Working distance in mm (not directly used, but kept for consistency)
        pixels_per_mm: Not used, kept for backward compatibility
        label_mode: Color mode for point ("None", "Red/Blue", "Random")
        video_path: Optional video path for consistent colors
    """
    if working_distance_mm is None or working_distance_mm <= 0:
        return  # Keep check for consistency, but B can be calculated without it
    
    for pair in pairs:
        # Unpack pair: (pid, xi, yi, xj, yj, th_i, r_i, th_j, r_j, score, ...)
        if len(pair) < 10:
            continue
        pid, xi, yi, xj, yj, th_i, r_i, th_j, r_j = pair[0:9]
        
        r_a = r_i  # A = inner radius
        r_c = r_j  # C = outer radius
        
        if r_a <= 0 or r_c <= 0:
            continue
        
        # Calculate B in pixels: B = (2*A*C)/(A+C)
        b_px = (2 * r_a * r_c) / (r_a + r_c)
        
        # Calculate pair midpoint
        mx = 0.5 * (xi + xj)
        my = 0.5 * (yi + yj)
        
        # Calculate direction vector from center to pair midpoint
        dx = mx - xCenter
        dy = my - yCenter
        dist_to_midpoint = math.sqrt(dx*dx + dy*dy)
        
        if dist_to_midpoint < 1e-6:
            continue  # Skip if midpoint is at center
        
        # Normalize direction vector
        ux = dx / dist_to_midpoint
        uy = dy / dist_to_midpoint
        
        # Calculate B point: B pixels away from center along the ray
        # Calculate point coordinates
        real_x = xCenter + b_px * ux
        real_y = yCenter + b_px * uy
        
        # Determine point color based on label_mode
        point_color = (255, 255, 255)  # White default
        if label_mode == "Red/Blue":
            point_color = (255, 255, 255)  # White
        elif label_mode == "Random":
            point_color = _get_pair_color(pid, video_path)
        
        # Draw point (small filled circle)
        cv2.circle(dst, (int(real_x), int(real_y)), 4, point_color, -1)


def draw_stats_overlay(dst, pairs_before_tracking: List[Tuple], pairs_after_tracking: List[Tuple], 
                       total_pairs: int, total_tracks: int, 
                       show_current_stats: bool, show_total_stats: bool):
    """
    Draw stats overlay in top left corner in white text.
    
    Args:
        dst: Destination image (BGR)
        pairs_before_tracking: Pairs before tracking (with pair IDs)
        pairs_after_tracking: Pairs after tracking (with track IDs)
        total_pairs: Cumulative count of all pairs seen
        total_tracks: Total number of unique tracks ever created
        show_current_stats: Whether to show current frame stats
        show_total_stats: Whether to show total stats
    """
    if not show_current_stats and not show_total_stats:
        return
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)  # White
    line_height = 20
    x_offset = 10
    y_start = 25
    
    y_pos = y_start
    
    if show_current_stats:
        # Count unique pairs in frame (before tracking)
        unique_pairs_in_frame = len(set(pid for pid, *_ in pairs_before_tracking))
        
        cv2.putText(dst, f"Current Stats:", (x_offset, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        cv2.putText(dst, f"  pair count: {unique_pairs_in_frame}", (x_offset, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
    
    if show_total_stats:
        if show_current_stats:
            y_pos += 5  # Small gap between sections
        cv2.putText(dst, f"Total Stats:", (x_offset, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        cv2.putText(dst, f"  total pairs: {total_pairs}", (x_offset, y_pos), font, font_scale, color, thickness)
        y_pos += line_height
        cv2.putText(dst, f"  total tracks: {total_tracks}", (x_offset, y_pos), font, font_scale, color, thickness)


