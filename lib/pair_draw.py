from typing import List, Tuple, Optional
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


def draw_pair_lines(dst, pairs: List[Tuple], show_labels: bool = False, label_mode: str = "Red/Blue", video_path: Optional[str] = None):
    """
    label_mode: "None", "Red/Blue", "Random"
    show_labels: if True, show #A/#B text labels
    """
    for (pid, xi, yi, xj, yj, *_rest) in pairs:
        line_color = (255, 255, 255)
        colA = colB = (255, 255, 255)

        if label_mode == "Red/Blue":
            colA = (0, 0, 255)  # Red
            colB = (255, 0, 0)  # Blue
            line_color = (255, 255, 255)
        elif label_mode == "Random":
            pair_col = _get_pair_color(pid, video_path)
            line_color = pair_col
            colA = colB = pair_col

        cv2.line(dst, (xi, yi), (xj, yj), line_color, 1)
        cv2.circle(dst, (xi, yi), 3, colA, -1)
        cv2.circle(dst, (xj, yj), 3, colB, -1)
        if show_labels:
            cv2.putText(dst, f"{pid}A", (xi + 4, yi - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colA, 1)
            cv2.putText(dst, f"{pid}B", (xj + 4, yj - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colB, 1)


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
        endB = np.array([xj, yj], dtype=float)
        if (endA - P).dot(u) * sgn > (endB - P).dot(u) * sgn:
            anchor = endA
        else:
            anchor = endB

        end = (anchor + sgn * ray_len * u).astype(int)
        ray_color = (255, 255, 255)
        if label_mode == "Random":
            ray_color = _get_pair_color(pid, video_path)
        cv2.line(dst, (int(anchor[0]), int(anchor[1])), (int(end[0]), int(end[1])), ray_color, 1)


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


