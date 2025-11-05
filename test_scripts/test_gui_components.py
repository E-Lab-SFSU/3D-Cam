#!/usr/bin/env python3
"""
Test GUI Components
-------------------
Verifies that the new shared GUI components from Phase 1 refactoring work correctly.
"""

import sys
import tkinter as tk
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all GUI components can be imported."""
    print("Testing GUI component imports...")
    try:
        from lib.gui import (
            WindowSize,
            STANDARD_SIZES,
            apply_standard_theme,
            format_window_title,
            STANDARD_PADDING,
            tooltip,
        )
        print("[OK] All GUI components imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_window_sizes():
    """Test window size enum and helper function."""
    print("\nTesting window sizes...")
    try:
        from lib.gui import WindowSize, STANDARD_SIZES, get_standard_size
        
        # Test enum values
        assert WindowSize.SMALL.value == (400, 550)
        assert WindowSize.CAPTURE.value == (1000, 700)
        print("[OK] Window size enum values correct")
        
        # Test standard sizes dict
        assert "small" in STANDARD_SIZES
        assert "capture" in STANDARD_SIZES
        print("[OK] STANDARD_SIZES dictionary correct")
        
        # Test get_standard_size function
        width, height = get_standard_size("small")
        assert width == 400 and height == 550
        print("[OK] get_standard_size function works")
        
        return True
    except Exception as e:
        print(f"[FAIL] Window size test failed: {e}")
        return False


def test_theme_application():
    """Test theme application."""
    print("\nTesting theme application...")
    try:
        from lib.gui import apply_standard_theme
        
        root = tk.Tk()
        root.withdraw()  # Hide window
        
        apply_standard_theme(root)
        print("[OK] Theme applied without errors")
        
        root.destroy()
        return True
    except Exception as e:
        print(f"[FAIL] Theme application test failed: {e}")
        return False


def test_title_formatting():
    """Test window title formatting."""
    print("\nTesting title formatting...")
    try:
        from lib.gui import format_window_title
        
        # Test basic title
        title1 = format_window_title("Test App")
        assert title1 == "Test App"
        print(f"[OK] Basic title: '{title1}'")
        
        # Test with platform
        title2 = format_window_title("Test App", platform="Windows")
        assert title2 == "Test App - Windows"
        print(f"[OK] Title with platform: '{title2}'")
        
        # Test with version
        title3 = format_window_title("Test App", version="v1.0")
        assert title3 == "Test App v1.0"
        print(f"[OK] Title with version: '{title3}'")
        
        # Test with both
        title4 = format_window_title("Test App", platform="Raspberry Pi", version="v2.0")
        assert title4 == "Test App v2.0 - Raspberry Pi"
        print(f"[OK] Title with both: '{title4}'")
        
        return True
    except Exception as e:
        print(f"[FAIL] Title formatting test failed: {e}")
        return False


def test_tooltip():
    """Test tooltip function."""
    print("\nTesting tooltip function...")
    try:
        from lib.gui import tooltip
        
        root = tk.Tk()
        root.withdraw()  # Hide window
        
        button = tk.Button(root, text="Test")
        tooltip(button, "Test tooltip")
        print("[OK] Tooltip attached without errors")
        
        root.destroy()
        return True
    except Exception as e:
        print(f"[FAIL] Tooltip test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("GUI Components Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_window_sizes,
        test_theme_application,
        test_title_formatting,
        test_tooltip,
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
