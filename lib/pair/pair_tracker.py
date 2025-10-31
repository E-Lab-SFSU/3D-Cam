from typing import List, Tuple, Dict, Optional
import math
import json
import os


PairTuple = Tuple[int, int, int, int, int, float, float, float, float, float, float, float]
# (pid, xi, yi, xj, yj, th_i, r_i, th_j, r_j, score, area_i, area_j)


class PairTracker:
    """
    Persistent tracker with velocity-based modeling and size/length morphing support.
    Matches pairs by predicted position, velocity smoothness, size smoothness, and length smoothness.
    Allows pairs to gradually change size (average blob area) and length (distance between points)
    throughout frames, preferring smooth transitions in matching decisions.
    Supports tracking of both moving and stationary objects (still tracking).
    """

    @staticmethod
    def _load_config(config_path: str = "tracker_config.json") -> Dict:
        """Load configuration from JSON file with default fallback."""
        default_config = {
            "tracking": {
                "max_match_dist_px": 25.0,
                "max_misses": 10
            },
            "smoothness": {
                "velocity": {
                    "angle_weight": 0.6,
                    "magnitude_weight": 0.4,
                    "magnitude_cv_multiplier": 5.0,
                    "penalty_scale": 10.0,
                    "max_history": 5
                },
                "size": {
                    "min_ratio": 0.7,
                    "max_ratio": 1.4,
                    "ratio_decay_range": 0.35,
                    "cv_multiplier": 3.0,
                    "cv_blend": 0.7,
                    "penalty_scale": 5.0,
                    "max_history": 5
                },
                "length": {
                    "min_ratio": 0.8,
                    "max_ratio": 1.25,
                    "ratio_decay_range": 0.25,
                    "cv_multiplier": 3.0,
                    "cv_blend": 0.7,
                    "penalty_scale": 5.0,
                    "max_history": 5
                }
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    def merge_dict(default, user):
                        result = default.copy()
                        for key, value in user.items():
                            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                                result[key] = merge_dict(result[key], value)
                            else:
                                result[key] = value
                        return result
                    return merge_dict(default_config, config)
            except Exception as e:
                print(f"[WARN] Failed to load tracker config from {config_path}: {e}. Using defaults.")
                return default_config
        return default_config

    def __init__(self, max_match_dist_px: float = None, max_misses: int = None,
                 config_path: str = "tracker_config.json"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize tracking parameters (use provided values or config defaults)
        self.max_match_dist_px = float(max_match_dist_px if max_match_dist_px is not None 
                                      else self.config["tracking"]["max_match_dist_px"])
        self.max_misses = int(max_misses if max_misses is not None 
                             else self.config["tracking"]["max_misses"])
        
        # Get smoothness configs
        self.vel_cfg = self.config["smoothness"]["velocity"]
        self.size_cfg = self.config["smoothness"]["size"]
        self.length_cfg = self.config["smoothness"]["length"]
        
        self.next_id: int = 1
        self.tracks: Dict[int, Dict] = {}

    @staticmethod
    def _midpoint(pair: PairTuple) -> Tuple[float, float]:
        _pid, xi, yi, xj, yj, *_ = pair
        return (0.5 * (xi + xj), 0.5 * (yi + yj))
    
    @staticmethod
    def _pair_size(pair: PairTuple) -> float:
        """Calculate the average area of the two blobs in a pair."""
        # Handle both old format (10 elements) and new format (12 elements with areas)
        if len(pair) >= 12:
            _pid, xi, yi, xj, yj, _th_i, _r_i, _th_j, _r_j, _score, area_i, area_j = pair
            return (area_i + area_j) / 2.0
        else:
            # Fallback: return 0 if areas not available (old format)
            return 0.0
    
    @staticmethod
    def _pair_length(pair: PairTuple) -> float:
        """Calculate the length (distance) between the two points of a pair."""
        _pid, xi, yi, xj, yj, *_ = pair
        dx = xj - xi
        dy = yj - yi
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    @staticmethod
    def _angle_between_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """Return angle in degrees between two 2D vectors (0-180)."""
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]*v1[0] + v1[1]*v1[1])
        mag2 = math.sqrt(v2[0]*v2[0] + v2[1]*v2[1])
        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0.0
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp for acos
        return math.degrees(math.acos(cos_angle))
    
    def _velocity_smoothness_score(self, new_vel: Tuple[float, float], vel_history: List[Tuple[float, float]]) -> float:
        """
        Calculate velocity smoothness score (0-1, higher = smoother).
        Based on angle consistency and magnitude consistency with previous velocities.
        """
        if len(vel_history) == 0:
            return 1.0  # No history, assume smooth
        
        # Calculate scores based on consistency
        angle_score = 1.0
        magnitude_score = 1.0
        
        new_mag = math.sqrt(new_vel[0]*new_vel[0] + new_vel[1]*new_vel[1])
        
        # Get most recent velocity for comparison
        prev_vel = vel_history[-1]
        prev_mag = math.sqrt(prev_vel[0]*prev_vel[0] + prev_vel[1]*prev_vel[1])
        
        if new_mag > 1e-6 and prev_mag > 1e-6:
            # Angle consistency: smaller angle difference = smoother
            angle_diff = self._angle_between_vectors(new_vel, prev_vel)
            # Normalize to 0-1: 0° = 1.0, 180° = 0.0
            angle_score = 1.0 - (angle_diff / 180.0)
            
            # Magnitude consistency: similar speeds = smoother
            mag_ratio = min(new_mag, prev_mag) / max(new_mag, prev_mag) if max(new_mag, prev_mag) > 0 else 1.0
            magnitude_score = mag_ratio
        
        # If we have more history, also check linearity across multiple frames
        if len(vel_history) >= 2:
            # Calculate average angle difference and magnitude variance
            angles = []
            magnitudes = []
            
            for v in vel_history + [new_vel]:
                mag = math.sqrt(v[0]*v[0] + v[1]*v[1])
                if mag > 1e-6:
                    magnitudes.append(mag)
                    angles.append(math.degrees(math.atan2(v[1], v[0])))
            
            if len(angles) >= 2:
                # Calculate magnitude coefficient of variation (lower = smoother)
                if len(magnitudes) >= 2:
                    avg_mag = sum(magnitudes) / len(magnitudes)
                    mag_variance = sum((m - avg_mag)**2 for m in magnitudes) / len(magnitudes)
                    if avg_mag > 1e-6:
                        mag_cv = math.sqrt(mag_variance) / avg_mag
                        # Normalize: 0 variance = 1.0, high variance -> 0
                        magnitude_score = max(magnitude_score, 
                                            1.0 / (1.0 + mag_cv * self.vel_cfg["magnitude_cv_multiplier"]))
                
                # Calculate angle consistency (circular variance)
                if len(angles) >= 2:
                    # Convert to unit vectors and average
                    sin_sum = sum(math.sin(math.radians(a)) for a in angles)
                    cos_sum = sum(math.cos(math.radians(a)) for a in angles)
                    mean_sin = sin_sum / len(angles)
                    mean_cos = cos_sum / len(angles)
                    # Resultant length (0-1): 1 = all same direction, 0 = scattered
                    resultant_length = math.sqrt(mean_sin*mean_sin + mean_cos*mean_cos)
                    angle_score = max(angle_score, resultant_length)
        
        # Combine scores: weighted average favoring angle consistency
        smoothness = (self.vel_cfg["angle_weight"] * angle_score + 
                     self.vel_cfg["magnitude_weight"] * magnitude_score)
        return smoothness
    
    def _size_change_smoothness_score(self, new_size: float, size_history: List[float]) -> float:
        """
        Calculate size change smoothness score (0-1, higher = smoother).
        Based on gradual average blob area changes rather than abrupt jumps.
        Areas can change significantly (e.g., blob expansion/contraction), so we allow
        larger relative changes while still preferring smooth transitions.
        """
        if len(size_history) == 0:
            return 1.0  # No history, assume smooth
        
        prev_size = size_history[-1]
        
        if prev_size < 1e-6:
            return 1.0 if new_size < 1e-6 else 0.5  # Handle zero-size pairs
        
        # Calculate relative change
        size_ratio = new_size / prev_size if prev_size > 0 else 1.0
        
        # Score based on how close to 1.0 the ratio is (1.0 = no change, perfectly smooth)
        # Allow gradual changes using config-specified ratio range
        min_ratio = self.size_cfg["min_ratio"]
        max_ratio = self.size_cfg["max_ratio"]
        if min_ratio <= size_ratio <= max_ratio:
            # Within reasonable range, score based on how close to 1.0
            distance_from_one = abs(size_ratio - 1.0)
            smoothness = 1.0 - (distance_from_one / self.size_cfg["ratio_decay_range"])
            smoothness = max(0.0, smoothness)
        else:
            # Outside reasonable range, penalize more
            if size_ratio < min_ratio:
                # Shrinking significantly
                excess = min_ratio - size_ratio
                smoothness = max(0.0, 1.0 - (excess / min_ratio))
            else:
                # Growing significantly
                excess = size_ratio - max_ratio
                smoothness = max(0.0, 1.0 / (1.0 + excess))
        
        # If we have more history, check consistency across multiple frames
        if len(size_history) >= 2:
            # Calculate coefficient of variation for size consistency
            sizes = size_history + [new_size]
            avg_size = sum(sizes) / len(sizes)
            if avg_size > 1e-6:
                size_variance = sum((s - avg_size)**2 for s in sizes) / len(sizes)
                size_cv = math.sqrt(size_variance) / avg_size
                # Lower CV = smoother, but allow gradual trends
                cv_score = 1.0 / (1.0 + size_cv * self.size_cfg["cv_multiplier"])
                smoothness = max(smoothness, cv_score * self.size_cfg["cv_blend"])
        
        return smoothness
    
    def _length_change_smoothness_score(self, new_length: float, length_history: List[float]) -> float:
        """
        Calculate length change smoothness score (0-1, higher = smoother).
        Based on gradual pair length (distance between points) changes rather than abrupt jumps.
        """
        if len(length_history) == 0:
            return 1.0  # No history, assume smooth
        
        prev_length = length_history[-1]
        
        if prev_length < 1e-6:
            return 1.0 if new_length < 1e-6 else 0.5  # Handle zero-length pairs
        
        # Calculate relative change
        length_ratio = new_length / prev_length if prev_length > 0 else 1.0
        
        # Score based on how close to 1.0 the ratio is (1.0 = no change, perfectly smooth)
        # Allow gradual changes using config-specified ratio range
        min_ratio = self.length_cfg["min_ratio"]
        max_ratio = self.length_cfg["max_ratio"]
        if min_ratio <= length_ratio <= max_ratio:
            # Within reasonable range, score based on how close to 1.0
            distance_from_one = abs(length_ratio - 1.0)
            smoothness = 1.0 - (distance_from_one / self.length_cfg["ratio_decay_range"])
            smoothness = max(0.0, smoothness)
        else:
            # Outside reasonable range, penalize more
            if length_ratio < min_ratio:
                # Shrinking significantly
                excess = min_ratio - length_ratio
                smoothness = max(0.0, 1.0 - (excess / min_ratio))
            else:
                # Growing significantly
                excess = length_ratio - max_ratio
                smoothness = max(0.0, 1.0 / (1.0 + excess))
        
        # If we have more history, check consistency across multiple frames
        if len(length_history) >= 2:
            # Calculate coefficient of variation for length consistency
            lengths = length_history + [new_length]
            avg_length = sum(lengths) / len(lengths)
            if avg_length > 1e-6:
                length_variance = sum((l - avg_length)**2 for l in lengths) / len(lengths)
                length_cv = math.sqrt(length_variance) / avg_length
                # Lower CV = smoother, but allow gradual trends
                cv_score = 1.0 / (1.0 + length_cv * self.length_cfg["cv_multiplier"])
                smoothness = max(smoothness, cv_score * self.length_cfg["cv_blend"])
        
        return smoothness

    def reset(self):
        self.next_id = 1
        self.tracks.clear()

    def update(self, pairs: List[PairTuple]) -> List[PairTuple]:
        """
        Given current frame pairs, return same tuples but with the first element
        replaced by a stable track_id. Internal state is updated.
        """
        if not pairs:
            # increment misses, retire old
            to_delete = []
            for tid, t in self.tracks.items():
                t["misses"] += 1
                if t["misses"] > self.max_misses:
                    to_delete.append(tid)
            for tid in to_delete:
                self.tracks.pop(tid, None)
            return []

        # Prepare current pair midpoints (tracking is based on midpoint positions only)
        cur_mids = [self._midpoint(p) for p in pairs]
        # Also prepare pair sizes and lengths for morphing tracking
        cur_sizes = [self._pair_size(p) for p in pairs]
        cur_lengths = [self._pair_length(p) for p in pairs]

        # Build candidate edges with motion-based scoring
        # Score combines distance, velocity smoothness, and size change smoothness
        edges: List[Tuple[int, int, float]] = []  # (tid, pair_idx, composite_score)
        for tid, t in self.tracks.items():
            prev_mid = t["mid"]
            vel = t.get("vel", (0.0, 0.0))
            vel_history = t.get("vel_history", [])
            
            # Predict position: prev + velocity (no acceleration)
            predicted = (prev_mid[0] + vel[0], prev_mid[1] + vel[1])
            
            for idx, mid in enumerate(cur_mids):
                # Distance from predicted position
                d2_pred = self._dist2(predicted, mid)
                # Also check distance from last position (fallback for new/stationary tracks)
                d2_last = self._dist2(prev_mid, mid)
                d2 = min(d2_pred, d2_last)
                
                if d2 > self.max_match_dist_px * self.max_match_dist_px:
                    continue
                
                # Calculate what the new velocity would be if this match is made
                candidate_new_vel = (mid[0] - prev_mid[0], mid[1] - prev_mid[1])
                
                # Calculate velocity smoothness score (0-1, higher = smoother)
                smoothness = self._velocity_smoothness_score(candidate_new_vel, vel_history)
                
                # Calculate size change smoothness score
                candidate_new_size = cur_sizes[idx]
                size_history = t.get("size_history", [])
                size_smoothness = self._size_change_smoothness_score(candidate_new_size, size_history)
                
                # Calculate length change smoothness score
                candidate_new_length = cur_lengths[idx]
                length_history = t.get("length_history", [])
                length_smoothness = self._length_change_smoothness_score(candidate_new_length, length_history)
                
                # Composite score: combine distance, velocity smoothness, size smoothness, and length smoothness
                # Lower is better, so we invert smoothness scores
                # Normalize distance to similar scale (use sqrt for pixel distance)
                dist_normalized = math.sqrt(d2)  # Distance in pixels
                smoothness_penalty = (1.0 - smoothness) * self.vel_cfg["penalty_scale"]
                size_penalty = (1.0 - size_smoothness) * self.size_cfg["penalty_scale"]
                length_penalty = (1.0 - length_smoothness) * self.length_cfg["penalty_scale"]
                composite_score = dist_normalized + smoothness_penalty + size_penalty + length_penalty
                
                edges.append((tid, idx, composite_score))

        # Greedy match by ascending composite score (distance + velocity smoothness + size smoothness + length smoothness)
        edges.sort(key=lambda e: e[2])
        taken_tracks = set()
        taken_pairs = set()
        assignment: Dict[int, int] = {}  # pair_index -> track_id
        for tid, idx, _d2 in edges:
            if tid in taken_tracks or idx in taken_pairs:
                continue
            taken_tracks.add(tid)
            taken_pairs.add(idx)
            assignment[idx] = tid

        # Assign existing or create new tracks
        enriched: List[PairTuple] = []
        for idx, pair in enumerate(pairs):
            mid = cur_mids[idx]
            if idx in assignment:
                tid = assignment[idx]
                tr = self.tracks[tid]
                prev_mid = tr["mid"]
                
                # Get previous velocity before updating
                prev_vel = tr.get("vel", (0.0, 0.0))
                
                # Update velocity: new_vel = current - previous (single frame velocity)
                new_vel = (mid[0] - prev_mid[0], mid[1] - prev_mid[1])
                
                # Maintain position history (kept for potential future use, but no longer needed for turn checking)
                pos_history = tr.get("pos_history", [])
                pos_history.append(prev_mid)
                # Keep only the last few frames of history
                max_history = 10  # Simple fixed history size
                if len(pos_history) > max_history:
                    pos_history = pos_history[-max_history:]
                tr["pos_history"] = pos_history
                
                # Maintain velocity history for smoothness scoring
                vel_history = tr.get("vel_history", [])
                vel_history.append(prev_vel)
                # Keep only the configured number of velocities for smoothness calculation
                max_vel_history = self.vel_cfg["max_history"]
                if len(vel_history) > max_vel_history:
                    vel_history = vel_history[-max_vel_history:]
                tr["vel_history"] = vel_history
                
                # Maintain size history for size morphing
                prev_size = tr.get("size", 0.0)
                current_pair_size = cur_sizes[idx]
                size_history = tr.get("size_history", [])
                size_history.append(prev_size)
                # Keep only the configured number of sizes for smoothness calculation
                max_size_history = self.size_cfg["max_history"]
                if len(size_history) > max_size_history:
                    size_history = size_history[-max_size_history:]
                tr["size_history"] = size_history
                tr["size"] = current_pair_size
                
                # Maintain length history for length morphing
                prev_length = tr.get("length", 0.0)
                current_pair_length = cur_lengths[idx]
                length_history = tr.get("length_history", [])
                length_history.append(prev_length)
                # Keep only the configured number of lengths for smoothness calculation
                max_length_history = self.length_cfg["max_history"]
                if len(length_history) > max_length_history:
                    length_history = length_history[-max_length_history:]
                tr["length_history"] = length_history
                tr["length"] = current_pair_length
                
                # Update current state
                tr["mid"] = mid
                tr["vel"] = new_vel
                tr["misses"] = 0
                tr["hits"] += 1
            else:
                tid = self.next_id
                self.next_id += 1
                current_pair_size = cur_sizes[idx]
                current_pair_length = cur_lengths[idx]
                self.tracks[tid] = {
                    "mid": mid,
                    "vel": (0.0, 0.0),
                    "size": current_pair_size,  # Initialize with current pair size
                    "length": current_pair_length,  # Initialize with current pair length
                    "pos_history": [],  # Start with empty history
                    "vel_history": [],  # Start with empty velocity history
                    "size_history": [],  # Start with empty size history
                    "length_history": [],  # Start with empty length history
                    "misses": 0,
                    "hits": 1,
                }
            # Replace first element (pid) with stable tid
            enriched.append((tid, *pair[1:]))

        # Age unmatched tracks
        to_delete = []
        for tid, t in self.tracks.items():
            if tid in taken_tracks:
                continue
            t["misses"] += 1
            if t["misses"] > self.max_misses:
                to_delete.append(tid)
        for tid in to_delete:
            self.tracks.pop(tid, None)

        return enriched

