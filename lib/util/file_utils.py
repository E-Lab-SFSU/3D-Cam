#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Utilities
--------------
Shared functions for finding latest files, calibration files, and other file operations.
"""

from pathlib import Path
from typing import Optional


def find_latest_file(directory: str, pattern: str) -> Optional[str]:
    """
    Find the latest file matching a pattern in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match (e.g., "*.json", "*.csv")
    
    Returns:
        Path to the latest file as string, or None if no files found.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return None
    
    files = list(dir_path.glob(pattern))
    if not files:
        return None
    
    # Sort by modification time (newest first)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return str(files[0])


def find_latest_calibration_file() -> Optional[str]:
    """
    Find the latest calibration JSON file in the calibrations folder.
    
    Returns:
        Path to the latest calibration file, or None if no file is found.
    """
    calibrations_dir = Path("calibrations")
    if not calibrations_dir.exists():
        return None
    
    json_files = list(calibrations_dir.glob("*.json"))
    if not json_files:
        return None
    
    # Sort by modification time (newest first)
    json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return str(json_files[0])


def find_latest_image_calibration_file() -> Optional[str]:
    """
    Find the latest image calibration JSON file in the calibrations folder.
    
    Returns:
        Path to the latest image calibration file, or None if no file is found.
    """
    calibrations_dir = Path("calibrations")
    if not calibrations_dir.exists():
        return None
    
    json_files = list(calibrations_dir.glob("image_calibration_*.json"))
    if not json_files:
        return None
    
    # Sort by modification time (newest first)
    json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return str(json_files[0])

