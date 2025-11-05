#!/usr/bin/env python3
"""
JSON Name Migration Script
--------------------------
Migrates old naming conventions to new clearer names in all JSON files:
- magic_constant → z_calibration_scale_factor
- magic_offset → z_calibration_offset_mm
- Zprime / zprime / avg_zprime → Geometric_Z_mm / avg_Geometric_Z_mm
- B / avg_b → radial_distance_from_center_px / avg_radial_distance_from_center_px
- A, C → inner_radius_px, outer_radius_px (in formulas/comments)

Run this script to update all existing calibration and preset JSON files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import sys

def migrate_calibration_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate calibration JSON structure."""
    migrated = data.copy()
    
    # Top-level keys
    if "magic_constant" in migrated:
        migrated["z_calibration_scale_factor"] = migrated.pop("magic_constant")
    
    if "magic_offset" in migrated:
        migrated["z_calibration_offset_mm"] = migrated.pop("magic_offset")
    
    # Update formula strings
    if "formula" in migrated:
        migrated["formula"] = migrated["formula"].replace("Zprime", "Geometric_Z_mm")
        migrated["formula"] = migrated["formula"].replace("magic_constant", "z_calibration_scale_factor")
        migrated["formula"] = migrated["formula"].replace("magic_offset", "z_calibration_offset_mm")
    
    if "zprime_formula" in migrated:
        formula = migrated.pop("zprime_formula")
        formula = formula.replace("Zprime", "Geometric_Z_mm")
        formula = formula.replace("(C-A)", "(outer_radius - inner_radius)")
        formula = formula.replace("(A+C)", "(inner_radius + outer_radius)")
        formula = formula.replace("C-A", "outer_radius - inner_radius")
        formula = formula.replace("A+C", "inner_radius + outer_radius")
        migrated["geometric_z_formula"] = formula
    
    if "b_formula" in migrated:
        formula = migrated.pop("b_formula")
        formula = formula.replace("B =", "radial_distance_from_center =")
        formula = formula.replace("B=", "radial_distance_from_center=")
        formula = formula.replace("(2*A*C)", "(2 * inner_radius * outer_radius)")
        formula = formula.replace("(A+C)", "(inner_radius + outer_radius)")
        formula = formula.replace("2*A*C", "2 * inner_radius * outer_radius")
        formula = formula.replace("A+C", "inner_radius + outer_radius")
        formula = formula.replace("A", "inner_radius")
        formula = formula.replace("C", "outer_radius")
        migrated["radial_distance_formula"] = formula
    
    # Update description
    if "description" in migrated:
        migrated["description"] = migrated["description"].replace("Zprime", "Geometric_Z_mm")
    
    # Migrate data_points array
    if "data_points" in migrated and isinstance(migrated["data_points"], list):
        for point in migrated["data_points"]:
            if isinstance(point, dict):
                if "avg_zprime" in point:
                    point["avg_Geometric_Z_mm"] = point.pop("avg_zprime")
                if "avg_b" in point:
                    point["avg_radial_distance_from_center_px"] = point.pop("avg_b")
    
    return migrated

def migrate_preset_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate pair_detect preset JSON structure."""
    migrated = data.copy()
    
    # Migrate calibration section
    if "calibration" in migrated:
        cal = migrated["calibration"]
        if "magic_constant" in cal:
            cal["z_calibration_scale_factor"] = cal.pop("magic_constant")
        if "magic_offset" in cal:
            cal["z_calibration_offset_mm"] = cal.pop("magic_offset")
    
    return migrated

def migrate_file(file_path: Path) -> bool:
    """Migrate a single JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine file type and migrate accordingly
        if "magic_constant" in data or "zprime_formula" in data or "b_formula" in data:
            migrated_data = migrate_calibration_json(data)
            file_type = "calibration"
        elif "calibration" in data or "params" in data:
            migrated_data = migrate_preset_json(data)
            file_type = "preset"
        else:
            # Check if it has any of the old keys
            has_old_keys = any(key in data for key in ["magic_constant", "magic_offset", "avg_zprime", "avg_b"])
            if has_old_keys:
                migrated_data = migrate_calibration_json(data)
                file_type = "calibration"
            else:
                print(f"[SKIP] {file_path}: Unknown JSON structure, skipping")
                return False
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(migrated_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] {file_type.upper()}: {file_path}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] {file_path}: Invalid JSON - {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return False

def main():
    """Find and migrate all JSON files."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Directories to search
    search_dirs = [
        project_root / "calibrations",
        project_root / "inputs_outputs",
        project_root,  # Root for detect_pairs_default.json
    ]
    
    json_files = []
    for search_dir in search_dirs:
        if search_dir.exists():
            json_files.extend(search_dir.rglob("*.json"))
    
    if not json_files:
        print("No JSON files found to migrate.")
        return
    
    print(f"Found {len(json_files)} JSON file(s) to migrate...\n")
    
    success_count = 0
    for json_file in sorted(json_files):
        if migrate_file(json_file):
            success_count += 1
    
    print(f"\nMigration complete: {success_count}/{len(json_files)} files migrated successfully.")

if __name__ == "__main__":
    main()
