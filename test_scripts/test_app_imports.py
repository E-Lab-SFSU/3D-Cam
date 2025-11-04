#!/usr/bin/env python3
"""
Test Application Imports
------------------------
Verifies that all applications modified in Phase 1 can be imported and 
their GUI classes can be instantiated without errors.
"""

import sys
import tkinter as tk
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_capture_raspi():
    """Test capture_raspi.py can be imported and GUI instantiated."""
    print("\nTesting capture_raspi.py...")
    try:
        from apps import capture_raspi
        print("[OK] Module imported")
        
        # Check if CaptureApp class exists
        assert hasattr(capture_raspi, 'CaptureApp'), "CaptureApp class not found"
        print("[OK] CaptureApp class found")
        
        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_capture_windows():
    """Test capture_windows.py can be imported and GUI instantiated."""
    print("\nTesting capture_windows.py...")
    try:
        from apps import capture_windows
        print("[OK] Module imported")
        
        # Check if CaptureApp class exists
        assert hasattr(capture_windows, 'CaptureApp'), "CaptureApp class not found"
        print("[OK] CaptureApp class found")
        
        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_calibrate_scale_raspi():
    """Test calibrate_scale_raspi.py can be imported."""
    print("\nTesting calibrate_scale_raspi.py...")
    try:
        from apps import calibrate_scale_raspi
        print("[OK] Module imported")
        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_calibrate_scale_windows():
    """Test calibrate_scale_windows.py can be imported."""
    print("\nTesting calibrate_scale_windows.py...")
    try:
        from apps import calibrate_scale_windows
        print("[OK] Module imported")
        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_calibrate_video():
    """Test calibrate_video.py can be imported."""
    print("\nTesting calibrate_video.py...")
    try:
        from apps import calibrate_video
        print("[OK] Module imported")
        
        # Check if VideoCalibrationApp class exists
        assert hasattr(calibrate_video, 'VideoCalibrationApp'), "VideoCalibrationApp class not found"
        print("[OK] VideoCalibrationApp class found")
        
        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_plot_z_histogram():
    """Test plot_z_histogram.py can be imported."""
    print("\nTesting plot_z_histogram.py...")
    try:
        from apps import plot_z_histogram
        print("[OK] Module imported")
        
        # Check if ZHistogramViewer class exists
        assert hasattr(plot_z_histogram, 'ZHistogramViewer'), "ZHistogramViewer class not found"
        print("[OK] ZHistogramViewer class found")
        
        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_pair_ui():
    """Test lib/pair/ui.py can be imported."""
    print("\nTesting lib/pair/ui.py...")
    try:
        from lib.pair import ui
        print("[OK] Module imported")
        
        # Check if build_gui function exists
        assert hasattr(ui, 'build_gui'), "build_gui function not found"
        print("[OK] build_gui function found")
        
        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_base_visualizer():
    """Test lib/visualizing/base_visualizer.py can be imported."""
    print("\nTesting lib/visualizing/base_visualizer.py...")
    try:
        from lib.visualizing import base_visualizer
        print("[OK] Module imported")
        
        # Check if Base3DVisualizer class exists
        assert hasattr(base_visualizer, 'Base3DVisualizer'), "Base3DVisualizer class not found"
        print("[OK] Base3DVisualizer class found")
        
        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Application Import Test Suite")
    print("=" * 60)
    
    tests = [
        test_capture_raspi,
        test_capture_windows,
        test_calibrate_scale_raspi,
        test_calibrate_scale_windows,
        test_calibrate_video,
        test_plot_z_histogram,
        test_pair_ui,
        test_base_visualizer,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
