#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON Utilities
---------------
Centralized JSON file operations with consistent error handling, encoding, and logging.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging

# Standard JSON formatting constants
JSON_INDENT = 2
JSON_ENCODING = 'utf-8'

# Standard JSON keys used across the codebase
JSON_KEY_CALIBRATION = "calibration"
JSON_KEY_PARAMS = "params"
JSON_KEY_OVERLAYS = "overlays"

logger = logging.getLogger(__name__)


def save_json(file_path: str, data: Dict[str, Any], indent: int = JSON_INDENT, 
              ensure_ascii: bool = False, create_dirs: bool = True) -> bool:
    """
    Save data to JSON file with consistent encoding and error handling.
    
    Args:
        file_path: Path to JSON file (as string)
        data: Dictionary to save
        indent: JSON indentation (default: JSON_INDENT = 2)
        ensure_ascii: Whether to escape non-ASCII characters (default: False)
        create_dirs: Whether to create parent directories if they don't exist (default: True)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding=JSON_ENCODING) as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        print(f"[ERROR] Failed to save JSON to {file_path}: {e}")
        return False


def load_json(file_path: str, default: Optional[Dict] = None) -> Optional[Dict]:
    """
    Load data from JSON file with consistent encoding and error handling.
    
    Args:
        file_path: Path to JSON file (as string)
        default: Default value to return if file doesn't exist or fails to load
    
    Returns:
        Dictionary from JSON, or default if file doesn't exist/fails
    """
    path = Path(file_path)
    if not path.exists():
        return default
    
    try:
        with open(path, 'r', encoding=JSON_ENCODING) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load JSON from {file_path}: {e}")
        print(f"[WARN] Failed to load JSON from {file_path}: {e}")
        return default


def update_json(file_path: str, updates: Dict[str, Any], create_if_missing: bool = True) -> bool:
    """
    Update a JSON file by merging updates into existing data.
    
    Args:
        file_path: Path to JSON file
        updates: Dictionary of updates to merge
        create_if_missing: If True, create file with updates if it doesn't exist
    
    Returns:
        True if successful, False otherwise
    
    Example:
        # Update calibration section
        update_json("preset.json", {"calibration": {"pixels_per_mm": 5.2}})
    """
    if create_if_missing or Path(file_path).exists():
        existing_data = load_json(file_path, default={})
        if existing_data is None:
            existing_data = {}
        
        # Deep merge updates
        def deep_merge(base: Dict, updates: Dict) -> Dict:
            result = base.copy()
            for key, value in updates.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_data = deep_merge(existing_data, updates)
        return save_json(file_path, merged_data)
    
    return False


def get_json_value(file_path: str, key_path: str, default: Any = None) -> Any:
    """
    Get a value from JSON file using dot-notation key path.
    
    Args:
        file_path: Path to JSON file
        key_path: Dot-separated key path (e.g., "calibration.pixels_per_mm")
        default: Default value if key not found
    
    Returns:
        Value at key path, or default
    
    Example:
        pixels_per_mm = get_json_value("cal.json", "pixels_per_mm", 0.0)
        magic_constant = get_json_value("cal.json", "calibration.magic_constant")
    """
    data = load_json(file_path)
    if data is None:
        return default
    
    keys = key_path.split('.')
    value = data
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
