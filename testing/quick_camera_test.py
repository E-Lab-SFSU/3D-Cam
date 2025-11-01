#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Camera Test - Fast test for specific resolution
Tests if 1920x1080 works with different formats
"""

import cv2
import time
import platform
import sys


def quick_test(camera_index=0, fourcc="MJPG", width=1920, height=1080):
    """Quick test of a specific format/resolution combo."""
    
    print(f"Testing {fourcc} at {width}x{height}...")
    
    # Try both backends on Windows
    if platform.system() == "Windows":
        backends = [
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_MSMF, "MSMF"),
        ]
    else:
        backends = [(cv2.CAP_V4L2, "V4L2")]
    
    best_fps = 0
    best_backend = None
    
    for backend_code, backend_name in backends:
        print(f"  {backend_name}: ", end="", flush=True)
        
        try:
            cap = cv2.VideoCapture(camera_index, backend_code)
            if not cap.isOpened():
                print("FAILED (can't open)")
                continue
        except Exception as e:
            print(f"FAILED (error: {e})")
            continue
        
        try:
            # Set format - order matters!
            fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
            
            # Try setting properties in optimal order for Windows
            # Set FOURCC first, then dimensions, then FPS
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
            time.sleep(0.1)  # Small delay after FOURCC
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            time.sleep(0.1)  # Small delay after dimensions
            
            # For MJPG, try different FPS settings - some cameras need explicit FPS
            if fourcc == "MJPG":
                # Try 30 FPS first, then max if that doesn't work well
                cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                cap.set(cv2.CAP_PROP_FPS, 0)  # Max speed for uncompressed
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for faster capture
            
            time.sleep(0.2)
            
            # Check actual settings
            act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if act_w != width or act_h != height:
                print(f"FAILED (got {act_w}x{act_h})")
                cap.release()
                continue
            
            # Flush frames and use grab()+retrieve() for MJPG to skip stale frames
            for _ in range(10):
                cap.grab()
            cap.retrieve()  # Get one decoded frame
            
            # Measure FPS - use grab()+retrieve() pattern for MJPG to get latest frames
            frame_count = 0
            start = time.time()
            test_duration = 3.0
            
            while time.time() - start < test_duration:
                # For MJPG, use grab()+retrieve() to skip stale frames
                if fourcc == "MJPG":
                    if not cap.grab():
                        continue
                    # Skip up to 2 stale frames
                    cap.grab()  # Skip one more if available
                    ret, frame = cap.retrieve()
                else:
                    ret, frame = cap.read()
                    
                if ret and frame is not None:
                    frame_count += 1
            
            elapsed = time.time() - start
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            print(f"{fps:.1f} FPS")
            
            if fps > best_fps:
                best_fps = fps
                best_backend = backend_name
            
            cap.release()
            
        except Exception as e:
            print(f"ERROR: {type(e).__name__}")
            try:
                cap.release()
            except:
                pass
    
    if best_fps > 0:
        print(f"\n✓ Best result: {best_backend} at {best_fps:.1f} FPS")
        return True, best_fps, best_backend
    else:
        print("\n✗ No working configuration found")
        return False, 0, None


if __name__ == "__main__":
    camera_index = 0
    if len(sys.argv) > 1:
        camera_index = int(sys.argv[1])
    
    print("=" * 60)
    print(f"Quick Camera Test - Camera {camera_index}")
    print("=" * 60)
    
    # Test common formats at 1920x1080
    formats = ["MJPG", "YUYV", "YUY2", "H264"]
    
    results = []
    for fmt in formats:
        print(f"\n{fmt}:")
        success, fps, backend = quick_test(camera_index, fmt, 1920, 1080)
        if success:
            results.append((fmt, fps, backend))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"\nBest format for 1920x1080: {results[0][0]} at {results[0][1]:.1f} FPS ({results[0][2]})")
        
        print("\nAll working formats:")
        for fmt, fps, backend in results:
            print(f"  {fmt:6} {fps:5.1f} FPS ({backend})")
    else:
        print("\nNo formats worked at 1920x1080")
        print("Camera may not support this resolution, or format names are different")

