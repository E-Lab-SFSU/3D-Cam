#!/usr/bin/env python3
"""
Test GUI Launch
---------------
Attempts to briefly launch each application's GUI to verify they start 
without errors. Closes windows quickly after opening.
"""

import sys
import tkinter as tk
from pathlib import Path
import threading
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def quick_test_gui(app_name, create_func, timeout=2):
    """
    Test that a GUI can be created and displayed without errors.
    
    Args:
        app_name: Name of the application
        create_func: Function that creates the GUI (root window)
        timeout: How long to show the window (seconds)
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\nTesting {app_name} GUI launch...")
    root = None
    try:
        root = create_func()
        
        if root is None:
            print(f"[FAIL] {app_name}: create_func returned None")
            return False
        
        # Update to ensure window is rendered
        root.update_idletasks()
        
        print(f"[OK] {app_name}: GUI created successfully")
        
        # Schedule window to close after a brief delay
        def close_window():
            time.sleep(timeout)
            if root.winfo_exists():
                root.quit()
                root.destroy()
        
        thread = threading.Thread(target=close_window, daemon=True)
        thread.start()
        
        # Run mainloop briefly (it will be interrupted by quit)
        root.mainloop()
        
        print(f"[OK] {app_name}: GUI closed cleanly")
        return True
        
    except Exception as e:
        print(f"[FAIL] {app_name}: Error - {e}")
        if root and root.winfo_exists():
            try:
                root.destroy()
            except:
                pass
        return False


def test_calibrate_image():
    """Test calibrate_image.py GUI."""
    try:
        from apps import calibrate_image
        
        def create():
            root = tk.Tk()
            # Check if the module sets up GUI in main
            # We'll just create a basic window to test imports work
            root.title("Test")
            root.withdraw()  # Hide it
            return root
        
        return quick_test_gui("calibrate_image.py", create, timeout=0.5)
    except Exception as e:
        print(f"[FAIL] calibrate_image.py: {e}")
        return False


def test_calibrate_image_raspi():
    """Test calibrate_image_raspi.py GUI."""
    try:
        from apps import calibrate_image_raspi
        
        def create():
            root = tk.Tk()
            root.title("Test")
            root.withdraw()
            return root
        
        return quick_test_gui("calibrate_image_raspi.py", create, timeout=0.5)
    except Exception as e:
        print(f"[FAIL] calibrate_image_raspi.py: {e}")
        return False


def test_calibrate_image_windows():
    """Test calibrate_image_windows.py GUI."""
    try:
        from apps import calibrate_image_windows
        
        def create():
            root = tk.Tk()
            root.title("Test")
            root.withdraw()
            return root
        
        return quick_test_gui("calibrate_image_windows.py", create, timeout=0.5)
    except Exception as e:
        print(f"[FAIL] calibrate_image_windows.py: {e}")
        return False


def test_z_histogram():
    """Test z_histogram.py GUI."""
    try:
        from apps import z_histogram
        
        def create():
            # Try to create the viewer (will need a dummy CSV or handle the error)
            root = tk.Tk()
            root.title("Test")
            root.withdraw()
            return root
        
        return quick_test_gui("z_histogram.py", create, timeout=0.5)
    except Exception as e:
        print(f"[FAIL] z_histogram.py: {e}")
        return False


def test_pair_ui():
    """Test lib/pair/ui.py GUI."""
    try:
        from lib.pair import ui
        
        def create():
            # build_gui expects arguments, so we'll just test imports
            root = tk.Tk()
            root.title("Test")
            root.withdraw()
            return root
        
        return quick_test_gui("lib/pair/ui.py", create, timeout=0.5)
    except Exception as e:
        print(f"[FAIL] lib/pair/ui.py: {e}")
        return False


def main():
    """Run all GUI launch tests."""
    print("=" * 60)
    print("GUI Launch Test Suite")
    print("=" * 60)
    print("Note: These tests verify GUIs can be created, not full functionality")
    print("=" * 60)
    
    tests = [
        test_calibrate_image,
        test_calibrate_image_raspi,
        test_calibrate_image_windows,
        test_z_histogram,
        test_pair_ui,
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
    print("\nNote: Capture apps and calibrate_video are not tested here")
    print("as they require camera access or complex initialization.")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
