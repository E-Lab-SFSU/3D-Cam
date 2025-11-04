"""
Calibration GUI Components
---------------------------
GUI components for video calibration tool, including VideoEntry and GUI builders.
"""

import csv
import os
import tkinter as tk
from tkinter import ttk, filedialog
from typing import List, Tuple, Dict, Optional
import numpy as np


class VideoEntry:
    """Container for a single video calibration entry."""
    def __init__(self, frame, row, app):
        self.frame = frame
        self.row = row
        self.app = app
        self.csv_var = tk.StringVar()
        self.height_var = tk.StringVar()
        self.csv_path = ""
        self.mm_height: Optional[float] = None
        
        # Create widgets
        ttk.Label(frame, text=f"CSV {row + 1}:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        
        csv_label = ttk.Label(frame, text="No CSV selected", foreground="gray")
        csv_label.grid(row=0, column=1, sticky="w", padx=5)
        self.csv_label = csv_label
        
        csv_btn = ttk.Button(frame, text="📂 Browse", 
                            command=lambda: self.select_csv())
        csv_btn.grid(row=0, column=2, padx=5)
        
        ttk.Label(frame, text="Height (mm):").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        height_entry = ttk.Entry(frame, textvariable=self.height_var, width=15)
        height_entry.grid(row=1, column=1, sticky="w", padx=5, pady=(5, 0))
        self.height_entry = height_entry
        
        remove_btn = ttk.Button(frame, text="✖ Remove", 
                               command=lambda: self.remove())
        remove_btn.grid(row=1, column=2, padx=5, pady=(5, 0))
        self.remove_btn = remove_btn
    
    def select_csv(self):
        """Open file dialog to select CSV file."""
        csv_file = filedialog.askopenfilename(
            title=f"Select CSV File for Calibration {self.row + 1}",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="inputs_outputs" if os.path.exists("inputs_outputs") else "."
        )
        
        if csv_file:
            self.csv_path = csv_file
            csv_name = os.path.basename(csv_file)
            self.csv_label.config(text=csv_name, foreground="black")
            print(f"[INFO] CSV {self.row + 1}: Selected {csv_name}")
            # Update live metrics when CSV is loaded
            if self.app:
                # Load CSV data for visualization
                csv_name = os.path.basename(csv_file)
                self.app.load_csv_for_visualization(csv_file, csv_name)
                self.app.update_live_metrics()
                self.app.update_3d_visualization()
    
    def get_data(self) -> Optional[Tuple[str, float, float, float, float, Dict, Dict, Optional[Dict]]]:
        """
        Get CSV path, mm height, working distance, average Zprime, and average B from highest quality pairs.
        Also returns metrics for input and chosen datasets.
        Returns None if invalid.
        Returns: (csv_path, mm_height, working_distance_mm, avg_zprime, avg_b, input_metrics, chosen_metrics, z_filter_stats)
        """
        if not self.csv_path or not os.path.exists(self.csv_path):
            return None
        
        if not self.app:
            return None
        
        try:
            mm_val = float(self.height_var.get())
            if mm_val <= 0:
                return None
            
            # Get working distance from app's global field
            working_dist_val = float(self.app.working_dist_var.get())
            if working_dist_val <= 0:
                return None
        except (ValueError, tk.TclError):
            return None
        
        # Load pairs from the selected CSV file
        if not os.path.exists(self.csv_path):
            return None
        
        try:
            pairs_data = []
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Try new A/C notation first, fall back to old notation for backward compatibility
                        r_a = float(row.get("A_px", row.get("Radius_A_px", 0)))  # Inner radius (A)
                        r_c = float(row.get("C_px", row.get("Radius_B_px", 0)))  # Outer radius (C)
                        score = float(row.get("Pair_Score", 0))
                        track_id = int(row.get("Track_ID", 0))
                        
                        if r_a > 0 and r_c > 0 and score > 0:
                            frame_num = int(row.get('Frame_Number', 0))
                            pairs_data.append({
                                "r_a": r_a,
                                "r_c": r_c,
                                "score": score,
                                "track_id": track_id,
                                "frame": frame_num  # Store frame number for visualization
                            })
                    except (ValueError, KeyError):
                        continue
            
            if len(pairs_data) == 0:
                return None
            
            # STEP 1: First, isolate near-mean pairs (Z-based filtering)
            # Calculate Zprime, Zdoubleprime for ALL pairs first
            all_pairs_zprimes = []
            all_pairs_zdoubleprimes = []
            all_pairs_z_values = []  # Store (pair_idx, zprime, zdoubleprime, z_mm)
            
            for idx, p in enumerate(pairs_data):
                r_a = p["r_a"]
                r_c = p["r_c"]
                if r_a + r_c > 0:
                    zprime = working_dist_val * (r_c - r_a) / (r_a + r_c)
                    zdoubleprime = (r_c - r_a) / (r_a + r_c)
                    all_pairs_zprimes.append(zprime)
                    all_pairs_zdoubleprimes.append(zdoubleprime)
                    all_pairs_z_values.append((idx, zprime, zdoubleprime, None))
            
            # Try to load Z_mm from CSV if available
            if hasattr(self, 'app') and self.app:
                try:
                    with open(self.csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        z_mm_map = {}
                        for row in reader:
                            try:
                                track_id = int(row.get('Track_ID', 0))
                                z_mm_str = row.get('Z_mm', '').strip()
                                if z_mm_str and track_id not in z_mm_map:
                                    z_mm_val = float(z_mm_str)
                                    if z_mm_val > 0:
                                        z_mm_map[track_id] = z_mm_val
                            except (ValueError, KeyError):
                                continue
                        
                        for i, (idx, zp, zdp, _) in enumerate(all_pairs_z_values):
                            pair_track_id = pairs_data[idx].get("track_id", 0)
                            if pair_track_id in z_mm_map:
                                all_pairs_z_values[i] = (idx, zp, zdp, z_mm_map[pair_track_id])
                except Exception:
                    pass
            
            # Apply Z-based filtering if enabled (isolate near-mean pairs)
            near_mean_pairs_indices = list(range(len(pairs_data)))
            z_filter_stats = None
            
            if hasattr(self, 'app') and self.app and self.app.enable_z_filter_var.get():
                filter_type = self.app.z_filter_type_var.get()
                threshold_std = self.app.z_filter_threshold_var.get()
                
                # Select which Z values to use for filtering
                z_values_for_filter = []
                if filter_type == "Zprime":
                    z_values_for_filter = [zv[1] for zv in all_pairs_z_values]
                elif filter_type == "Zdoubleprime":
                    z_values_for_filter = [zv[2] for zv in all_pairs_z_values]
                elif filter_type == "Z_mm":
                    z_values_for_filter = [zv[3] for zv in all_pairs_z_values if zv[3] is not None]
                
                if len(z_values_for_filter) > 0 and len(z_values_for_filter) == len(all_pairs_z_values):
                    # Calculate mean and std of Z values
                    mean_z = np.mean(z_values_for_filter)
                    std_z = np.std(z_values_for_filter)
                    
                    if std_z > 0:
                        # Filter pairs: keep those within threshold_std standard deviations
                        filtered_indices = []
                        for i, zv in enumerate(all_pairs_z_values):
                            idx, zp, zdp, zmm = zv
                            
                            z_to_check = None
                            if filter_type == "Zprime":
                                z_to_check = zp
                            elif filter_type == "Zdoubleprime":
                                z_to_check = zdp
                            elif filter_type == "Z_mm":
                                z_to_check = zmm
                            
                            if z_to_check is not None:
                                z_score = abs(z_to_check - mean_z) / std_z
                                if z_score <= threshold_std:
                                    filtered_indices.append(idx)
                        
                        if len(filtered_indices) > 0:
                            near_mean_pairs_indices = filtered_indices
                            z_filter_stats = {
                                "filter_type": filter_type,
                                "threshold_std": float(threshold_std),
                                "mean_z": float(mean_z),
                                "std_z": float(std_z),
                                "pairs_before": len(pairs_data),
                                "pairs_after": len(filtered_indices),
                                "omitted": len(pairs_data) - len(filtered_indices)
                            }
            
            # STEP 2: Then, choose high quality pairs from near-mean pairs
            min_score_threshold = 0.9
            if hasattr(self, 'app') and self.app:
                min_score_threshold = self.app.min_score_var.get()
            
            # Filter by quality score
            quality_pairs = []
            if hasattr(self, 'app') and self.app and self.app.enable_quality_filter_var.get():
                quality_pairs = [pairs_data[i] for i in near_mean_pairs_indices 
                               if pairs_data[i]["score"] >= min_score_threshold]
            else:
                quality_pairs = [pairs_data[i] for i in near_mean_pairs_indices]
            
            if len(quality_pairs) == 0:
                return None
            
            # Calculate Zprime, Zdoubleprime, and B for each quality pair
            zprimes = []
            zdoubleprimes = []
            b_values = []
            
            for idx, p in enumerate(quality_pairs):
                r_a = p["r_a"]  # A is the inner radius (smaller)
                r_c = p["r_c"]  # C is the outer radius (larger)
                if r_a + r_c > 0:
                    zprime = working_dist_val * (r_c - r_a) / (r_a + r_c)
                    zdoubleprime = (r_c - r_a) / (r_a + r_c)  # working_distance = 1
                    zprimes.append(zprime)
                    zdoubleprimes.append(zdoubleprime)
                    # Calculate B = (2*A*C)/(A+C)
                    b_val = (2 * r_a * r_c) / (r_a + r_c)
                    b_values.append(b_val)
            
            # Final quality pairs are already filtered
            final_quality_pairs = quality_pairs
            
            # Store calibration pairs for visualization
            if hasattr(self, 'app') and self.app:
                csv_name = os.path.basename(self.csv_path)
                # Track which frames are calibration pairs
                calibration_frames_map = {}  # {track_id: set of frames}
                
                # Collect frames for all calibration pairs
                for p in final_quality_pairs:
                    track_id = p.get("track_id", 0)
                    frame = p.get("frame", 0)
                    if track_id > 0 and frame > 0:
                        if track_id not in calibration_frames_map:
                            calibration_frames_map[track_id] = set()
                        calibration_frames_map[track_id].add(frame)
                
                # Store in app's visualization
                self.app.set_calibration_pairs(csv_name, calibration_frames_map)
                
                # Load full CSV data for visualization if not already loaded
                if csv_name not in self.app.viz_data:
                    self.app.load_csv_for_visualization(self.csv_path, csv_name)
            
            if len(zprimes) == 0:
                return None
            
            avg_zprime = np.mean(zprimes)
            avg_b = np.mean(b_values)
            
            # Calculate metrics for chosen/quality pairs
            chosen_metrics = {
                "count": len(final_quality_pairs),
                "zprime": {
                    "mean": float(np.mean(zprimes)),
                    "std": float(np.std(zprimes)),
                    "min": float(np.min(zprimes)),
                    "max": float(np.max(zprimes))
                },
                "b": {
                    "mean": float(np.mean(b_values)),
                    "std": float(np.std(b_values)),
                    "min": float(np.min(b_values)),
                    "max": float(np.max(b_values))
                },
                "score": {
                    "mean": float(np.mean([p["score"] for p in final_quality_pairs])),
                    "std": float(np.std([p["score"] for p in final_quality_pairs])),
                    "min": float(np.min([p["score"] for p in final_quality_pairs])),
                    "max": float(np.max([p["score"] for p in final_quality_pairs]))
                }
            }
            
            # Add Z filtering statistics to chosen_metrics if available
            if z_filter_stats:
                chosen_metrics["z_filter"] = z_filter_stats
            
            # Calculate metrics for all input pairs
            all_zprimes = []
            all_b_values = []
            all_scores = []
            for p in pairs_data:
                r_a = p["r_a"]
                r_c = p["r_c"]
                if r_a + r_c > 0:
                    zprime = working_dist_val * (r_c - r_a) / (r_a + r_c)
                    all_zprimes.append(zprime)
                    b_val = (2 * r_a * r_c) / (r_a + r_c)
                    all_b_values.append(b_val)
                    all_scores.append(p["score"])
            
            input_metrics = {
                "count": len(pairs_data),
                "zprime": {
                    "mean": float(np.mean(all_zprimes)) if all_zprimes else 0.0,
                    "std": float(np.std(all_zprimes)) if all_zprimes else 0.0,
                    "min": float(np.min(all_zprimes)) if all_zprimes else 0.0,
                    "max": float(np.max(all_zprimes)) if all_zprimes else 0.0
                },
                "b": {
                    "mean": float(np.mean(all_b_values)) if all_b_values else 0.0,
                    "std": float(np.std(all_b_values)) if all_b_values else 0.0,
                    "min": float(np.min(all_b_values)) if all_b_values else 0.0,
                    "max": float(np.max(all_b_values)) if all_b_values else 0.0
                },
                "score": {
                    "mean": float(np.mean(all_scores)) if all_scores else 0.0,
                    "std": float(np.std(all_scores)) if all_scores else 0.0,
                    "min": float(np.min(all_scores)) if all_scores else 0.0,
                    "max": float(np.max(all_scores)) if all_scores else 0.0
                }
            }
            
            return (self.csv_path, mm_val, working_dist_val, avg_zprime, avg_b, input_metrics, chosen_metrics, z_filter_stats)
        
        except Exception as e:
            print(f"[ERROR] Failed to read {self.csv_path}: {e}")
            return None
    
    def remove(self):
        """Remove this entry from the GUI."""
        if self.app and len(self.app.video_entries) > 2:
            self.app.remove_video_entry(self)


def display_metrics(metrics_text, input_metrics_list, chosen_metrics_list, total_input, total_chosen, z_filter_stats_list=None):
    """Display detailed metrics for input and chosen datasets."""
    metrics_text.config(state="normal")
    metrics_text.delete(1.0, tk.END)
    
    # Calculate percentages for each video
    input_video_dict = {name: m for name, m in input_metrics_list}
    z_filter_dict = {name: stats for name, stats in (z_filter_stats_list or [])}
    
    # Header
    metrics_text.insert(tk.END, "="*80 + "\n")
    metrics_text.insert(tk.END, "DATASET METRICS\n")
    metrics_text.insert(tk.END, "="*80 + "\n\n")
    
    # Input dataset totals
    metrics_text.insert(tk.END, f"INPUT DATASET (Total: {total_input} pairs)\n")
    metrics_text.insert(tk.END, "-"*80 + "\n")
    
    # Per-video input metrics
    for name, m in input_metrics_list:
        metrics_text.insert(tk.END, f"\n  {name}:\n")
        metrics_text.insert(tk.END, f"    Pairs: {m['count']}\n")
        metrics_text.insert(tk.END, f"    Zprime: mean={m['zprime']['mean']:.4f}, std={m['zprime']['std']:.4f}, "
                                   f"range=[{m['zprime']['min']:.4f}, {m['zprime']['max']:.4f}]\n")
        metrics_text.insert(tk.END, f"    B:      mean={m['b']['mean']:.4f}, std={m['b']['std']:.4f}, "
                                   f"range=[{m['b']['min']:.4f}, {m['b']['max']:.4f}]\n")
        metrics_text.insert(tk.END, f"    Score:  mean={m['score']['mean']:.4f}, std={m['score']['std']:.4f}, "
                                   f"range=[{m['score']['min']:.4f}, {m['score']['max']:.4f}]\n")
    
    # Chosen dataset totals
    overall_percent = 100 * total_chosen / max(1, total_input)
    metrics_text.insert(tk.END, f"\n{'='*80}\n")
    metrics_text.insert(tk.END, f"CHOSEN DATASET (Total: {total_chosen} pairs, {overall_percent:.1f}% of input)\n")
    metrics_text.insert(tk.END, "-"*80 + "\n")
    
    # Per-video chosen metrics with percentages
    for name, m in chosen_metrics_list:
        input_count = input_video_dict.get(name, {}).get("count", 0)
        video_percent = 100 * m['count'] / max(1, input_count) if input_count > 0 else 0.0
        metrics_text.insert(tk.END, f"\n  {name}:\n")
        metrics_text.insert(tk.END, f"    Pairs: {m['count']} ({video_percent:.1f}% of input)\n")
        metrics_text.insert(tk.END, f"    Zprime: mean={m['zprime']['mean']:.4f}, std={m['zprime']['std']:.4f}, "
                                   f"range=[{m['zprime']['min']:.4f}, {m['zprime']['max']:.4f}]\n")
        metrics_text.insert(tk.END, f"    B:      mean={m['b']['mean']:.4f}, std={m['b']['std']:.4f}, "
                                   f"range=[{m['b']['min']:.4f}, {m['b']['max']:.4f}]\n")
        metrics_text.insert(tk.END, f"    Score:  mean={m['score']['mean']:.4f}, std={m['score']['std']:.4f}, "
                                   f"range=[{m['score']['min']:.4f}, {m['score']['max']:.4f}]\n")
        
        # Add Z filtering info if available
        if name in z_filter_dict:
            zf = z_filter_dict[name]
            metrics_text.insert(tk.END, f"\n    Z Filtering ({zf['filter_type']}):\n")
            metrics_text.insert(tk.END, f"      Target Z (mean): {zf['mean_z']:.4f}\n")
            metrics_text.insert(tk.END, f"      Std Dev: {zf['std_z']:.4f}\n")
            metrics_text.insert(tk.END, f"      Threshold: ±{zf['threshold_std']:.2f}σ\n")
            metrics_text.insert(tk.END, f"      Pairs: {zf['pairs_before']} → {zf['pairs_after']} (omitted {zf['omitted']})\n")
    
    metrics_text.config(state="disabled")

