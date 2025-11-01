#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Performance Test Script
-------------------------------
Tests different formats, resolutions, and backends to find optimal camera settings.
Runs comprehensive tests and reports FPS for each combination.
"""

import cv2
import time
import platform
from collections import defaultdict


def test_format_resolution(camera_index, fourcc, width, height, backend, duration=2.0):
    """
    Test a specific format/resolution combination and return FPS.
    
    Returns:
        (success, fps, actual_width, actual_height, actual_fourcc)
    """
    print(f"  Testing {fourcc} {width}x{height} (backend: {backend})...", end=" ", flush=True)
    
    cap = None
    try:
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            print("FAILED (can't open)")
            return False, 0, 0, 0, "unknown"
        
        # Set format
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 0)  # Try max speed
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for faster capture
        
        # Small delay for camera to adjust
        time.sleep(0.2)
        
        # Get actual settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps_set = cap.get(cv2.CAP_PROP_FPS)
        
        # Get actual FOURCC
        try:
            from lib.camera_info import fourcc_int_to_string
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            actual_fourcc = fourcc_int_to_string(fourcc_int)
        except:
            actual_fourcc = fourcc
        
        # Verify settings were applied
        if actual_width != width or actual_height != height:
            print(f"FAILED (got {actual_width}x{actual_height} instead)")
            cap.release()
            return False, 0, actual_width, actual_height, actual_fourcc
        
        # Flush a few frames to stabilize
        for _ in range(5):
            cap.read()
        
        # Measure actual FPS
        frame_count = 0
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time:
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_count += 1
            else:
                # If we get too many failures, break early
                if frame_count > 0 and time.time() - start_time > 0.5:
                    break
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        cap.release()
        
        if fps > 0:
            print(f"✓ {fps:.1f} FPS ({actual_fourcc}, {actual_width}x{actual_height})")
            return True, fps, actual_width, actual_height, actual_fourcc
        else:
            print(f"FAILED (no frames)")
            return False, 0, actual_width, actual_height, actual_fourcc
            
    except Exception as e:
        print(f"ERROR: {e}")
        if cap:
            try:
                cap.release()
            except:
                pass
        return False, 0, 0, 0, "unknown"


def main():
    print("=" * 80)
    print("Camera Performance Test")
    print("=" * 80)
    
    # Get camera index
    import sys
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except:
            pass
    
    print(f"\nTesting Camera {camera_index}")
    print("-" * 80)
    
    # Determine backends to test
    if platform.system() == "Windows":
        backends_to_test = [
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_MSMF, "MSMF"),
        ]
    else:
        backends_to_test = [
            (cv2.CAP_V4L2, "V4L2"),
        ]
    
    # Formats and resolutions to test
    formats_to_test = [
        ("YUYV", [(640, 480), (1280, 720), (1920, 1080)]),
        ("YUY2", [(640, 480), (1280, 720), (1920, 1080)]),
        ("MJPG", [(640, 480), (1280, 720), (1920, 1080)]),
        ("H264", [(1280, 720), (1920, 1080)]),  # H264 typically only at higher res
    ]
    
    results = defaultdict(list)  # Key: (format, width, height), Value: list of (backend, fps)
    
    for backend_code, backend_name in backends_to_test:
        print(f"\n{backend_name} Backend:")
        print("-" * 80)
        
        for fourcc, resolutions in formats_to_test:
            print(f"\n{fourcc}:")
            
            for width, height in resolutions:
                success, fps, act_w, act_h, act_fourcc = test_format_resolution(
                    camera_index, fourcc, width, height, backend_code, duration=2.0
                )
                
                if success and fps > 0:
                    results[(fourcc, act_w, act_h)].append((backend_name, fps, act_fourcc))
        
        # Small delay between backends
        time.sleep(0.5)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    if not results:
        print("\nNo successful tests! Camera may not be accessible or configured incorrectly.")
        return
    
    # Sort by FPS (descending)
    sorted_results = sorted(
        results.items(),
        key=lambda x: max(fps for _, fps, _ in x[1]),
        reverse=True
    )
    
    print(f"\n{'Format':<8} {'Resolution':<15} {'Backend':<8} {'FPS':<8} {'Notes':<20}")
    print("-" * 80)
    
    for (fourcc, width, height), backend_results in sorted_results:
        for backend_name, fps, actual_fourcc in sorted(backend_results, key=lambda x: x[1], reverse=True):
            notes = ""
            if actual_fourcc != fourcc:
                notes = f"(actual: {actual_fourcc})"
            print(f"{fourcc:<8} {width}x{height:<10} {backend_name:<8} {fps:>6.1f} {notes:<20}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Find best for 1920x1080
    best_1080p = None
    best_1080p_fps = 0
    for (fourcc, width, height), backend_results in results.items():
        if width == 1920 and height == 1080:
            max_fps = max(fps for _, fps, _ in backend_results)
            if max_fps > best_1080p_fps:
                best_1080p_fps = max_fps
                best_1080p = (fourcc, backend_results[0][2], max_fps)
    
    if best_1080p:
        print(f"\nBest 1920x1080: {best_1080p[0]} at {best_1080p[2]:.1f} FPS")
        print(f"  Use format: {best_1080p[0]}")
        print(f"  Actual format may be: {best_1080p[1]}")
    
    # Find best overall FPS
    best_overall = sorted_results[0]
    best_fourcc, best_res = best_overall[0]
    best_fps = max(fps for _, fps, _ in best_overall[1])
    print(f"\nHighest FPS: {best_fourcc} {best_res[0]}x{best_res[1]} at {best_fps:.1f} FPS")
    
    # Find best balance
    high_res_results = [
        (k, v) for k, v in results.items()
        if k[1] >= 1280 and k[2] >= 720 and max(fps for _, fps, _ in v) >= 15
    ]
    if high_res_results:
        best_balance = max(high_res_results, key=lambda x: max(fps for _, fps, _ in x[1]))
        print(f"\nBest balance (quality/speed): {best_balance[0][0]} {best_balance[0][1]}x{best_balance[0][2]} at {max(fps for _, fps, _ in best_balance[1]):.1f} FPS")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

