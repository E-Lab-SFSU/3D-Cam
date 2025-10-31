import json
import os
from typing import Dict, Tuple


def save_preset_file(
    preset_path: str,
    params: Dict,
    overlays: Dict,
    overlay_targets: Dict,
    center: Tuple[int, int, bool],
    video_path: str,
) -> bool:
    data = {
        "params": params,
        "overlays": overlays,
        "overlay_targets": overlay_targets,
        "center": {"x": center[0], "y": center[1], "valid": bool(center[2])},
        "video_path": video_path,
    }
    try:
        with open(preset_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def load_preset_file(
    preset_path: str,
    params: Dict,
    overlays: Dict,
    overlay_targets: Dict,
    video_path: str,
):
    if not os.path.exists(preset_path):
        return params, overlays, overlay_targets, (None, None, False), video_path, False
    try:
        with open(preset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out_params = dict(params)
        out_overlays = dict(overlays)
        out_targets = dict(overlay_targets)

        if "params" in data:
            for k, v in data["params"].items():
                if k in out_params:
                    out_params[k] = v
        if "overlays" in data:
            for k, v in data["overlays"].items():
                if k in out_overlays:
                    out_overlays[k] = v
            # Migration: convert old "color_ab" to "label_mode"
            if "color_ab" in data["overlays"] and "label_mode" not in data["overlays"]:
                out_overlays["label_mode"] = "Color A/B" if data["overlays"]["color_ab"] else "None"
        if "overlay_targets" in data:
            for k, v in data["overlay_targets"].items():
                if k in out_targets:
                    out_targets[k] = v

        cx = cy = None
        valid = False
        if "center" in data:
            c = data["center"]
            cx = c.get("x")
            cy = c.get("y")
            valid = bool(c.get("valid", False) and cx is not None and cy is not None)

        new_video_path = data.get("video_path", video_path)
        return out_params, out_overlays, out_targets, (cx, cy, valid), new_video_path, True
    except Exception:
        return params, overlays, overlay_targets, (None, None, False), video_path, False


