#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Utilities
-------------
Shared functions for CSV file operations, including auto-loading latest CSV files.
"""

from pathlib import Path
from typing import Optional, Callable


def find_latest_csv(directory: str = "inputs_outputs", pattern: str = "*.csv") -> Optional[Path]:
    """
    Find the latest CSV file in a directory (recursively).
    
    Args:
        directory: Directory to search (default: "inputs_outputs")
        pattern: File pattern to match (default: "*.csv")
    
    Returns:
        Path to the latest CSV file, or None if no files found.
    """
    output_dir = Path(directory)
    if not output_dir.exists():
        return None
    
    # Find all CSV files in subdirectories
    csv_files = list(output_dir.rglob(pattern))
    if not csv_files:
        return None
    
    # Sort by modification time (newest first)
    csv_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return csv_files[0]


def auto_load_latest_csv(load_callback: Callable[[str], None], directory: str = "inputs_outputs", pattern: str = "*.csv") -> bool:
    """
    Automatically load the latest CSV file from a directory.
    
    Args:
        load_callback: Function to call with the CSV file path (as string)
        directory: Directory to search (default: "inputs_outputs")
        pattern: File pattern to match (default: "*.csv")
    
    Returns:
        True if a file was loaded successfully, False otherwise.
    """
    latest_csv = find_latest_csv(directory, pattern)
    if latest_csv is None:
        return False
    
    try:
        load_callback(str(latest_csv))
        return True
    except Exception as e:
        print(f"Failed to auto-load {latest_csv}: {e}")
        return False

