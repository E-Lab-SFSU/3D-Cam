#!/usr/bin/env python3
"""
Run All Tests
-------------
Master test script that runs all Phase 1 GUI consistency test suites.
"""

import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_test(test_script):
    """Run a test script and return success status."""
    script_path = Path(__file__).parent / test_script
    print(f"\n{'=' * 60}")
    print(f"Running {test_script}...")
    print('=' * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent.parent),
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"[FAIL] Failed to run {test_script}: {e}")
        return False


def main():
    """Run all test suites."""
    print("=" * 60)
    print("Phase 1 GUI Consistency - Complete Test Suite")
    print("=" * 60)
    
    test_scripts = [
        "test_gui_components.py",
        "test_app_imports.py",
        "test_gui_launch.py",
    ]
    
    results = {}
    
    for test_script in test_scripts:
        results[test_script] = run_test(test_script)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_script, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status}: {test_script}")
    
    print("=" * 60)
    print(f"Overall: {passed}/{total} test suites passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
