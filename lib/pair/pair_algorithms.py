import math
from typing import List, Tuple, Dict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


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
        area = w * h
        if area < minA or area > maxA:
            continue
        if w > maxW or h > maxW:
            continue
        xc, yc = x + w // 2, y + h // 2
        th, r = polar_from_center(xc, yc, cx, cy)
        blobs.append(dict(theta=th, r=r, xc=xc, yc=yc, area=area, box=(x, y, w, h)))
    return blobs


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


