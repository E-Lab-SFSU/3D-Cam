"""
Video Calibration Logic
-----------------------
Core calibration calculation logic for video calibration.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime


class VideoCalibrator:
    """Core calibration calculation logic."""
    
    def __init__(self):
        self.z_calibration_scale_factor: Optional[float] = None
        self.z_calibration_offset_mm: Optional[float] = None
        self.calibration_data: Optional[Dict] = None
    
    def calculate_calibration(
        self,
        data_points: List[Tuple[str, float, float, float, float]],
        all_input_metrics: List[Tuple[str, Dict]],
        all_chosen_metrics: List[Tuple[str, Dict]],
        all_z_filter_stats: Optional[List[Tuple[str, Dict]]] = None
    ) -> Dict:
        """
        Calculate z_calibration_scale_factor and z_calibration_offset_mm using linear regression.
        
        Args:
            data_points: List of (csv_path, mm_height, working_distance, avg_geometric_z_mm, avg_b)
            all_input_metrics: List of (csv_name, metrics_dict) for input dataset
            all_chosen_metrics: List of (csv_name, metrics_dict) for chosen dataset
            all_z_filter_stats: Optional list of (csv_name, z_filter_stats_dict)
        
        Returns:
            Dictionary with calibration results including:
            - z_calibration_scale_factor
            - z_calibration_offset_mm
            - r_squared
            - calibration_data (full structure)
        """
        if len(data_points) < 2:
            raise ValueError(
                f"Need at least 2 valid data points. Currently have: {len(data_points)}"
            )
        
        # Extract values
        geometric_z_values = np.array([z for _, _, _, z, _ in data_points])
        b_values = np.array([b for _, _, _, _, b in data_points])
        z_values = np.array([h for _, h, _, _, _ in data_points])  # Z = calibrated mm height input
        
        # Linear regression: Z = Geometric_Z_mm * z_calibration_scale_factor + z_calibration_offset_mm
        coeffs = np.polyfit(geometric_z_values, z_values, 1)
        self.z_calibration_scale_factor = coeffs[0]  # slope
        self.z_calibration_offset_mm = coeffs[1]    # intercept
        
        # Calculate R² for quality assessment
        predicted_z = self.z_calibration_scale_factor * geometric_z_values + self.z_calibration_offset_mm
        ss_res = np.sum((z_values - predicted_z) ** 2)
        ss_tot = np.sum((z_values - np.mean(z_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate average working distance from data points
        working_distances = [wd for _, _, wd, _, _ in data_points]
        avg_working_distance_mm = np.mean(working_distances) if len(working_distances) > 0 else None
        
        # Calculate aggregate metrics
        total_input_count = sum(m["count"] for _, m in all_input_metrics)
        total_chosen_count = sum(m["count"] for _, m in all_chosen_metrics)
        
        # Create dictionary for looking up input metrics by video name
        input_video_dict = {name: m for name, m in all_input_metrics}
        
        self.calibration_data = {
            "z_calibration_scale_factor": float(self.z_calibration_scale_factor),
            "z_calibration_offset_mm": float(self.z_calibration_offset_mm),
            "r_squared": float(r_squared),
            "data_points": [
                {
                    "csv_path": csv_path,
                    "z_mm": float(h),  # Z = calibrated input value
                    "working_distance_mm": float(wd),
                    "avg_Geometric_Z_mm": float(z),
                    "avg_radial_distance_from_center_px": float(b)
                }
                for csv_path, h, wd, z, b in data_points
            ],
            "formula": "Z = Geometric_Z_mm * z_calibration_scale_factor + z_calibration_offset_mm",
            "description": "Z is the calibrated mm height input, Geometric_Z_mm is calculated from pair data",
            "geometric_z_formula": "Geometric_Z_mm = working_distance * (C-A)/(A+C)",
            "radial_distance_formula": "radial_distance_from_center = (2*A*C)/(A+C)",
            "metrics": {
                "input_dataset": {
                    "total_pairs": total_input_count,
                    "per_video": [
                        {
                            "video": name,
                            "count": m["count"],
                            "geometric_z_mean": m.get("geometric_z", {}).get("mean", m.get("zprime", {}).get("mean", 0)),
                            "geometric_z_std": m.get("geometric_z", {}).get("std", m.get("zprime", {}).get("std", 0)),
                            "geometric_z_min": m.get("geometric_z", {}).get("min", m.get("zprime", {}).get("min", 0)),
                            "geometric_z_max": m.get("geometric_z", {}).get("max", m.get("zprime", {}).get("max", 0)),
                            "b_mean": m["b"]["mean"],
                            "b_std": m["b"]["std"],
                            "b_min": m["b"]["min"],
                            "b_max": m["b"]["max"],
                            "score_mean": m["score"]["mean"],
                            "score_std": m["score"]["std"],
                            "score_min": m["score"]["min"],
                            "score_max": m["score"]["max"]
                        }
                        for name, m in all_input_metrics
                    ]
                },
                "chosen_dataset": {
                    "total_pairs": total_chosen_count,
                    "percent_of_input": float(100 * total_chosen_count / max(1, total_input_count)),
                    "per_video": [
                        {
                            "video": name,
                            "count": m["count"],
                            "percent_of_input": float(100 * m["count"] / max(1, input_video_dict.get(name, {}).get("count", 0))) if name in input_video_dict else 0.0,
                            "geometric_z_mean": m.get("geometric_z", {}).get("mean", m.get("zprime", {}).get("mean", 0)),
                            "geometric_z_std": m.get("geometric_z", {}).get("std", m.get("zprime", {}).get("std", 0)),
                            "geometric_z_min": m.get("geometric_z", {}).get("min", m.get("zprime", {}).get("min", 0)),
                            "geometric_z_max": m.get("geometric_z", {}).get("max", m.get("zprime", {}).get("max", 0)),
                            "b_mean": m["b"]["mean"],
                            "b_std": m["b"]["std"],
                            "b_min": m["b"]["min"],
                            "b_max": m["b"]["max"],
                            "score_mean": m["score"]["mean"],
                            "score_std": m["score"]["std"],
                            "score_min": m["score"]["min"],
                            "score_max": m["score"]["max"]
                        }
                        for name, m in all_chosen_metrics
                    ]
                }
            }
        }
        
        # Add average working distance to calibration data if available
        if avg_working_distance_mm is not None:
            self.calibration_data["working_distance_mm"] = float(avg_working_distance_mm)
        
        # Backward compatibility: add old keys for compatibility with existing code
        self.calibration_data["magic_constant"] = float(self.z_calibration_scale_factor)
        self.calibration_data["magic_offset"] = float(self.z_calibration_offset_mm)
        
        return {
            "z_calibration_scale_factor": self.z_calibration_scale_factor,
            "z_calibration_offset_mm": self.z_calibration_offset_mm,
            "r_squared": r_squared,
            "avg_b": float(np.mean(b_values)) if len(b_values) > 0 else 0.0,
            "total_input_count": total_input_count,
            "total_chosen_count": total_chosen_count,
            "calibration_data": self.calibration_data,
            "z_filter_summary": self._build_z_filter_summary(all_z_filter_stats) if all_z_filter_stats else "",
            # Backward compatibility aliases
            "magic_constant": self.z_calibration_scale_factor,
            "magic_offset": self.z_calibration_offset_mm
        }
    
    def _build_z_filter_summary(self, all_z_filter_stats: List[Tuple[str, Dict]]) -> str:
        """Build summary string for Z filtering statistics."""
        if not all_z_filter_stats:
            return ""
        total_omitted = sum(stats['omitted'] for _, stats in all_z_filter_stats)
        return f"\nZ Filtering: Omitted {total_omitted} outlier pairs across all videos"
    
    def auto_save_calibration(self, calibrations_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Automatically save calibration data to the calibrations folder.
        
        Args:
            calibrations_dir: Optional path to calibrations directory. Defaults to "calibrations" folder.
        
        Returns:
            Path to saved file, or None if save failed.
        """
        if self.calibration_data is None:
            return None
        
        if calibrations_dir is None:
            calibrations_dir = Path("calibrations")
        calibrations_dir.mkdir(exist_ok=True)
        
        # Generate timestamped filename combining all CSV filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_names = []
        if self.calibration_data.get("data_points"):
            for point in self.calibration_data["data_points"]:
                csv_path = point.get("csv_path", "")
                if csv_path:
                    # Extract CSV filename without extension
                    csv_name = Path(csv_path).stem
                    csv_names.append(csv_name)
        
        # Combine CSV names with underscores
        if csv_names:
            combined_names = "_".join(csv_names)
            # Limit filename length to avoid filesystem issues
            if len(combined_names) > 100:
                combined_names = combined_names[:100]
            prefix = combined_names + "_"
        else:
            prefix = ""
        
        filename = f"{prefix}video_calibration_{timestamp}.json"
        file_path = calibrations_dir / filename
        
        try:
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.calibration_data, f, indent=2)
            return file_path
        except Exception as e:
            print(f"[ERROR] Failed to auto-save calibration: {e}")
            return None

