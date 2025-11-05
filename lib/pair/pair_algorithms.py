import math
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed as skimage_watershed
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    peak_local_max = None
    skimage_watershed = None


def angdiff(a: float, b: float) -> float:
    return abs(((a - b + 180) % 360) - 180)


def line_distance_to_point(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = math.hypot(vx, vy)
    if denom < 1e-9:
        return math.hypot(px - ax, py - ay)
    return abs(vx * wy - vy * wx) / denom


def polar_from_center(x: float, y: float, cx: float, cy: float) -> Tuple[float, float]:
    dx, dy = x - cx, y - cy
    r = math.hypot(dx, dy)
    th = math.degrees(math.atan2(dy, dx))
    if th < 0:
        th += 360
    return th, r


def detect(binary: np.ndarray, cx: int, cy: int, params: Dict) -> List[Dict]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: List[Dict] = []
    minA, maxA, maxW = int(params["minArea"]), int(params["maxArea"]), int(params["maxW"])
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Use actual contour area (white pixels), not bounding box area
        area = cv2.contourArea(c)
        if area < minA or area > maxA:
            continue
        if w > maxW or h > maxW:
            continue
        xc, yc = x + w // 2, y + h // 2
        th, r = polar_from_center(xc, yc, cx, cy)
        blobs.append(dict(theta=th, r=r, xc=xc, yc=yc, area=area, box=(x, y, w, h)))

    return blobs


def detect_watershed(binary: np.ndarray, cx: int, cy: int, params: Dict) -> Tuple[List[Dict], np.ndarray]:
    """
    Detect blobs using watershed segmentation to separate overlapping objects.
    Better than standard contour detection for touching/overlapping blobs.
    
    Args:
        binary: Binary image (0/255)
        cx, cy: Optical center coordinates
        params: Detection parameters (minArea, maxArea, maxW, etc.)
    
    Returns:
        Tuple of:
        - List of blob dictionaries with same format as detect()
        - Markers array for debug visualization (same shape as binary, uint8)
    """
    # Noise removal with morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area (dilated to ensure we capture background)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area using distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Finding unknown region (boundary between sure foreground and sure background)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling (connected components of sure foreground)
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Mark the region of unknown with zero (watershed will determine these boundaries)
    markers[unknown == 255] = 0
    
    # Apply watershed algorithm
    # Convert binary to 3-channel for watershed (requires color image)
    binary_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(binary_3ch, markers)
    
    # Extract blobs from markers
    blobs: List[Dict] = []
    minA, maxA, maxW = int(params["minArea"]), int(params["maxArea"]), int(params["maxW"])
    
    # Get unique marker IDs (excluding background -1 and unknown 0, 1)
    unique_markers = np.unique(markers)
    for marker_id in unique_markers:
        if marker_id <= 1:  # Skip background (-1) and unknown/background (0, 1)
            continue
        
        # Create mask for this marker
        mask = (markers == marker_id).astype(np.uint8) * 255
        
        # Find contours in this mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Use largest contour for this marker
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        # Filter by size
        if area < minA or area > maxA:
            continue
        if w > maxW or h > maxW:
            continue
        
        xc, yc = x + w // 2, y + h // 2
        th, r = polar_from_center(xc, yc, cx, cy)
        blobs.append(dict(theta=th, r=r, xc=xc, yc=yc, area=area, box=(x, y, w, h)))
    
    # Convert markers to uint8 for visualization (clamp negative values)
    markers_vis = np.clip(markers, 0, 255).astype(np.uint8)
    
    return blobs, markers_vis


def detect_watershed_robust(
    binary: np.ndarray, 
    cx: int, 
    cy: int, 
    params: Dict,
    return_debug: bool = False
) -> Tuple[List[Dict], Dict]:
    """
    Watershed-based blob detection following OpenCV/PyImageSearch tutorial approach.
    Uses Euclidean Distance Transform, peak detection, and scikit-image watershed.
    Designed for separating overlapping/touching blobs while preserving isolated ones.
    
    Args:
        binary: Binary image (0/255) - can be grayscale or BGR
        cx, cy: Optical center coordinates
        params: Detection parameters including watershed-specific params:
            - minArea, maxArea, maxW: Standard blob filtering
            - ws_min_distance: Minimum distance between peaks in EDT (default: 20)
            - ws_use_pyrmeanshift: Use pyramid mean shift filtering (default: False)
            - ws_pyrmeanshift_sp: Spatial window radius for mean shift (default: 21)
            - ws_pyrmeanshift_sr: Color window radius for mean shift (default: 51)
            - ws_use_otsu: Use Otsu's thresholding (default: False, if binary already thresholded)
        return_debug: If True, return debug dictionary with intermediate images
    
    Returns:
        Tuple of:
        - List of blob dictionaries (same format as detect())
        - Debug dictionary (if return_debug=True) containing:
            - 'input': Original input image
            - 'pyrmeanshift': After pyramid mean shift filtering (if enabled)
            - 'binary': Binary thresholded image
            - 'dist_transform': Euclidean Distance Transform (normalized for visualization)
            - 'local_max': Local maxima (peaks) in distance transform
            - 'markers': Marker labels from connected components
            - 'markers_colored': Colored visualization of markers
            - 'labels': Final watershed labels
            - 'labels_colored': Colored visualization of final labels
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image is required for watershed detection. Install with: pip install scikit-image")
    
    # Extract parameters with defaults
    minA = int(params.get("minArea", 4))
    maxA = int(params.get("maxArea", 5000))
    maxW = int(params.get("maxW", 100))
    
    ws_min_distance = int(params.get("ws_min_distance", 20))
    ws_use_pyrmeanshift = bool(params.get("ws_use_pyrmeanshift", False))
    ws_pyrmeanshift_sp = int(params.get("ws_pyrmeanshift_sp", 21))
    ws_pyrmeanshift_sr = int(params.get("ws_pyrmeanshift_sr", 51))
    ws_use_otsu = bool(params.get("ws_use_otsu", False))
    
    debug_images = {} if return_debug else None
    
    # Step 1: Convert input to grayscale if needed, store original for debug
    if len(binary.shape) == 3:
        if return_debug:
            debug_images['input'] = binary.copy()
        gray_input = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    else:
        if return_debug:
            debug_images['input'] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        gray_input = binary.copy()
    
    # Step 2: Optional pyramid mean shift filtering (from tutorial)
    if ws_use_pyrmeanshift:
        # Convert grayscale to BGR for mean shift (requires color input)
        bgr_input = cv2.cvtColor(gray_input, cv2.COLOR_GRAY2BGR)
        shifted = cv2.pyrMeanShiftFiltering(bgr_input, ws_pyrmeanshift_sp, ws_pyrmeanshift_sr)
        gray_shifted = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        if return_debug:
            debug_images['pyrmeanshift'] = shifted.copy()
    else:
        gray_shifted = gray_input
    
    # Step 3: Apply Otsu's thresholding if requested (or if binary isn't already thresholded)
    if ws_use_otsu or np.max(gray_shifted) > 1:  # Check if already binary
        _, thresh = cv2.threshold(gray_shifted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        # Already binary, just ensure it's 0/255
        thresh = (gray_shifted > 127).astype(np.uint8) * 255
    
    if return_debug:
        debug_images['binary'] = thresh.copy()
    
    # Step 4: Compute Euclidean Distance Transform (EDT)
    # This computes the distance from each foreground pixel to the nearest background pixel
    D = ndimage.distance_transform_edt(thresh)
    
    if return_debug:
        # Normalize for visualization (0-255)
        D_vis = cv2.normalize(D, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        debug_images['dist_transform'] = D_vis
    
    # Step 5: Find local maxima (peaks) in the distance transform
    # These peaks represent the centers of objects
    # peak_local_max returns coordinates (n_peaks, 2), so we convert to boolean mask
    peak_coords = peak_local_max(D, min_distance=ws_min_distance, labels=thresh)
    localMax = np.zeros(D.shape, dtype=bool)
    if peak_coords.size > 0:
        # peak_coords is (n_peaks, 2) with (row, col) coordinates
        localMax[tuple(peak_coords.T)] = True
    
    if return_debug:
        # Visualize local maxima as white points on black background
        local_max_vis = (localMax.astype(np.uint8) * 255)
        debug_images['local_max'] = local_max_vis
    
    # Step 6: Perform connected component analysis on local maxima
    # This creates markers for the watershed algorithm
    markers, num_labels = ndimage.label(localMax, structure=np.ones((3, 3)))
    
    if return_debug:
        # Create colored visualization of markers
        markers_colored = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
        unique_markers = np.unique(markers)
        num_objects = len([m for m in unique_markers if m > 0])  # Exclude background (0)
        for marker_id in unique_markers:
            if marker_id == 0:  # Background
                continue
            mask = (markers == marker_id)
            # Assign color based on marker ID (use HSV coloring for distinct colors)
            hue = int(((marker_id - 1) * 180 / max(1, num_objects)) % 180)
            markers_colored[mask] = [hue, 255, 255]
        markers_colored = cv2.cvtColor(markers_colored, cv2.COLOR_HSV2BGR)
        markers_colored[markers == 0] = [0, 0, 0]  # Background in black
        # Convert markers to uint8 for display (normalize to 0-255 range)
        markers_uint8 = cv2.normalize(markers.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        debug_images['markers'] = markers_uint8
        debug_images['markers_colored'] = markers_colored
    
    # Step 7: Apply watershed algorithm
    # Note: watershed assumes markers are local minima, so we use -D (negative distance)
    # This makes peaks (high distance values) become valleys (low values in -D)
    labels = skimage_watershed(-D, markers, mask=thresh)
    
    if return_debug:
        # Create colored visualization of final labels
        labels_colored = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        unique_labels = np.unique(labels)
        num_segments = len([l for l in unique_labels if l > 0])  # Exclude background (0)
        for label_id in unique_labels:
            if label_id == 0:  # Background
                continue
            mask = (labels == label_id)
            # Assign color based on label ID
            hue = int(((label_id - 1) * 180 / max(1, num_segments)) % 180)
            labels_colored[mask] = [hue, 255, 255]
        labels_colored = cv2.cvtColor(labels_colored, cv2.COLOR_HSV2BGR)
        labels_colored[labels == 0] = [0, 0, 0]  # Background in black
        # Convert labels to uint8 for display (normalize to 0-255 range)
        labels_uint8 = cv2.normalize(labels.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        debug_images['labels'] = labels_uint8
        debug_images['labels_colored'] = labels_colored
    
    # Step 8: Extract blobs from watershed labels
    # Loop over each unique label and extract contour
    blobs: List[Dict] = []
    
    for label in np.unique(labels):
        # Skip background (label 0)
        if label == 0:
            continue
        
        # Create mask for this label
        mask = np.zeros(gray_input.shape, dtype="uint8")
        mask[labels == label] = 255
        
        # Find contours in the mask and extract the largest one
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Use largest contour (as in tutorial)
        c = max(contours, key=cv2.contourArea)
        
        # Get bounding box and area
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        # Filter by size constraints
        if area < minA or area > maxA:
            continue
        if w > maxW or h > maxW:
            continue
        
        # Calculate center point
        xc = x + w // 2
        yc = y + h // 2
        
        # Calculate polar coordinates from optical center
        th, r = polar_from_center(xc, yc, cx, cy)
        
        # Add blob
        blobs.append(dict(theta=th, r=r, xc=xc, yc=yc, area=area, box=(x, y, w, h)))
    
    if return_debug:
        return blobs, debug_images
    else:
        return blobs, {}


def pair_scored(blobs: List[Dict], params: Dict, xCenter: int, yCenter: int, center_valid: bool) -> List[Tuple]:
    if len(blobs) < 2 or not center_valid:
        return []

    used = [False] * len(blobs)
    pairs: List[Tuple] = []
    pid = 1

    w_theta = float(params["w_theta"])
    w_area = float(params["w_area"])
    w_center = float(params["w_center"])
    smin = float(params["Smin"])
    dmr = max(1e-6, float(params["maxDMR"]))
    rgap = float(params["maxRadGap"])    
    coff = max(1.0, float(params["maxCenterOff"]))

    for i, b1 in enumerate(blobs):
        if used[i]:
            continue
        best_j, best_score = -1, -1.0
        for j, b2 in enumerate(blobs):
            if j == i or used[j]:
                continue

            if abs(b1["r"] - b2["r"]) > rgap:
                continue
            dθ = angdiff(b1["theta"], b2["theta"])
            if dθ > dmr:
                continue

            S_theta = 1.0 - (dθ / dmr)
            S_area = min(b1["area"], b2["area"]) / max(b1["area"], b2["area"])
            d_center = line_distance_to_point(b1["xc"], b1["yc"], b2["xc"], b2["yc"], xCenter, yCenter)
            # Hard gate: reject pairs whose AC-to-center distance exceeds pixel threshold
            if d_center > coff:
                continue
            S_center = 1.0 - (d_center / coff)
            if S_center < 0:
                S_center = 0.0

            score = w_theta * S_theta + w_area * S_area + w_center * S_center

            if score > best_score:
                best_score, best_j = score, j

        if best_j >= 0 and best_score >= smin:
            b2 = blobs[best_j]
            # Ensure first point has smaller r value (A), second point has larger r value (C)
            if b1["r"] > b2["r"]:
                b1, b2 = b2, b1
            pairs.append((pid, b1["xc"], b1["yc"], b2["xc"], b2["yc"],
                          b1["theta"], b1["r"], b2["theta"], b2["r"], best_score,
                          b1["area"], b2["area"]))
            used[i] = used[best_j] = True
            pid += 1

    return pairs


def pair_scored_symmetric(blobs: List[Dict], params: Dict, xCenter: int, yCenter: int, center_valid: bool) -> List[Tuple]:
    if len(blobs) < 2 or not center_valid:
        return []

    n = len(blobs)
    best_match = [-1] * n
    best_score = [0.0] * n

    for i, b1 in enumerate(blobs):
        for j, b2 in enumerate(blobs):
            if j == i:
                continue
            if abs(b1["r"] - b2["r"]) > params["maxRadGap"]:
                continue
            dθ = angdiff(b1["theta"], b2["theta"])
            if dθ > params["maxDMR"]:
                continue

            Sθ = 1 - (dθ / params["maxDMR"])            
            SA = min(b1["area"], b2["area"]) / max(b1["area"], b2["area"])
            dC = line_distance_to_point(b1["xc"], b1["yc"], b2["xc"], b2["yc"], xCenter, yCenter)
            # Hard gate: reject if beyond pixel threshold
            if dC > params["maxCenterOff"]:
                continue
            SC = max(0, 1 - (dC / params["maxCenterOff"]))
            S = (params["w_theta"] * Sθ + params["w_area"] * SA + params["w_center"] * SC)

            if S > best_score[i]:
                best_score[i], best_match[i] = S, j

    pairs: List[Tuple] = []
    used = set()
    pid = 1
    for i in range(n):
        j = best_match[i]
        if j >= 0 and best_match[j] == i and i not in used and j not in used:
            if best_score[i] >= params["Smin"]:
                b1, b2 = blobs[i], blobs[j]
                # Ensure first point has smaller r value (A), second point has larger r value (C)
                if b1["r"] > b2["r"]:
                    b1, b2 = b2, b1
                pairs.append((pid, b1["xc"], b1["yc"], b2["xc"], b2["yc"],
                              b1["theta"], b1["r"], b2["theta"], b2["r"], best_score[i],
                              b1["area"], b2["area"]))
                used.update({i, j})
                pid += 1

    return pairs


def pair_scored_hungarian(blobs: List[Dict], params: Dict, xCenter: int, yCenter: int, center_valid: bool) -> List[Tuple]:
    if len(blobs) < 2 or not center_valid:
        return []

    N = len(blobs)
    w_theta = float(params["w_theta"])
    w_area = float(params["w_area"])
    w_center = float(params["w_center"])
    smin = float(params["Smin"])
    dmr = max(1e-6, float(params["maxDMR"]))
    rgap = float(params["maxRadGap"])    
    coff = max(1.0, float(params["maxCenterOff"]))

    S = np.zeros((N, N), dtype=float)
    for i, b1 in enumerate(blobs):
        for j, b2 in enumerate(blobs):
            if j == i:
                continue
            if abs(b1["r"] - b2["r"]) > rgap:
                continue
            dθ = angdiff(b1["theta"], b2["theta"])
            if dθ > dmr:
                continue

            S_theta = 1.0 - (dθ / dmr)
            S_area = min(b1["area"], b2["area"]) / max(b1["area"], b2["area"])
            d_center = line_distance_to_point(b1["xc"], b1["yc"], b2["xc"], b2["yc"], xCenter, yCenter)
            # Hard gate: zero out invalid pairs
            if d_center > coff:
                continue
            S_center = 1.0 - (d_center / coff)
            if S_center < 0:
                S_center = 0.0

            S[i, j] = w_theta * S_theta + w_area * S_area + w_center * S_center

    row_ind, col_ind = linear_sum_assignment(-S)

    pairs: List[Tuple] = []
    used = set()
    pid = 1
    for i, j in zip(row_ind, col_ind):
        if i >= j or (i in used) or (j in used):
            continue
        score = float(S[i, j])
        if score >= smin and score > 0:
            b1, b2 = blobs[i], blobs[j]
            # Ensure first point has smaller r value (A), second point has larger r value (C)
            if b1["r"] > b2["r"]:
                b1, b2 = b2, b1
            pairs.append((pid, b1["xc"], b1["yc"], b2["xc"], b2["yc"],
                          b1["theta"], b1["r"], b2["theta"], b2["r"], round(score, 4),
                          b1["area"], b2["area"]))
            used.update({i, j})
            pid += 1

    return pairs


def detect_yolo(frame: np.ndarray, model, cx: int, cy: int, params: Dict, 
                confidence_threshold: float = 0.25, class_filter: Optional[List[int]] = None) -> List[Dict]:
    """
    Detect objects using YOLO model and convert them to blob format.
    
    Args:
        frame: BGR image frame (numpy array)
        model: YOLO model instance (from ultralytics)
        cx, cy: Optical center coordinates
        params: Detection parameters dictionary (same format as detect())
        confidence_threshold: Minimum confidence for YOLO detections
        class_filter: Optional list of class IDs to filter (None = all classes)
    
    Returns:
        List of blob dictionaries with same format as detect(): 
        [dict(theta=th, r=r, xc=xc, yc=yc, area=area, box=(x, y, w, h)), ...]
    """
    # Run YOLO inference
    results = model(frame, verbose=False, conf=confidence_threshold)
    
    blobs: List[Dict] = []
    minA = int(params.get("minArea", 0))
    maxA = int(params.get("maxArea", 1000000))
    maxW = int(params.get("maxW", 10000))
    
    # Get threshold parameters for area calculation
    threshold_value = int(params.get("threshold", 127))
    invert_threshold = params.get("invert_threshold", 0)
    blur_size = int(params.get("blur", 1))
    
    # Convert frame to grayscale for thresholding
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Process each detection
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Filter by class if specified
            if class_filter is not None and int(box.cls[0]) not in class_filter:
                continue
            
            # Get bounding box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calculate bounding box dimensions
            w = x2 - x1
            h = y2 - y1
            
            # Filter by max width/height
            if w > maxW or h > maxW:
                continue
            
            # Center point of bounding box
            xc = (x1 + x2) // 2
            yc = (y1 + y2) // 2
            
            # Extract region within bounding box (with bounds checking)
            y1_clamped = max(0, y1)
            y2_clamped = min(gray.shape[0], y2)
            x1_clamped = max(0, x1)
            x2_clamped = min(gray.shape[1], x2)
            
            if y2_clamped <= y1_clamped or x2_clamped <= x1_clamped:
                continue
            
            roi = gray[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
            
            if roi.size == 0:
                continue
            
            # Apply blur if specified
            if blur_size > 1 and blur_size % 2 == 1:
                roi = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
            
            # Apply thresholding to ROI
            if invert_threshold:
                _, binary_roi = cv2.threshold(roi, threshold_value, 255, cv2.THRESH_BINARY_INV)
            else:
                _, binary_roi = cv2.threshold(roi, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Count white pixels (area)
            area = cv2.countNonZero(binary_roi)
            
            # Filter by area constraints
            if area < minA or area > maxA:
                continue
            
            # Calculate polar coordinates from optical center
            th, r = polar_from_center(xc, yc, cx, cy)
            
            # Add blob (using bounding box for box parameter)
            blobs.append(dict(
                theta=th, 
                r=r, 
                xc=xc, 
                yc=yc, 
                area=area, 
                box=(x1, y1, w, h)
            ))
    
    return blobs


