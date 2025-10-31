#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for camera capture operations
"""

import subprocess
import platform


def safe_run(cmd, timeout=2):
    """
    Safely run a subprocess command.
    
    Args:
        cmd: Command to run (list or string)
        timeout: Timeout in seconds (default: 2)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
        return True
    except Exception as e:
        print(f"[WARN] Command failed: {e}")
        return False

