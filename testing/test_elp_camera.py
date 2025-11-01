#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELP Camera Specific Test
Tests settings specific to ELP USB cameras
"""

import cv2
import time
import sys

camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

print("=" * 70)
print(f"ELP Camera Test - Camera {camera_index}")
print("Testing different FPS settings and property orders")
print("=" * 70)

configs = [
    ("FPS=30, buffer=1", 30, 1),
    ("FPS=30, buffer=2", 30, 2),
    ("FPS=25, buffer=1", 25, 1),
    ("FPS=20, buffer=1", 20, 1),
    ("FPS=15, buffer=1", 15, 1),
]

for desc, fps, buf_size in configs:
    print(f"\n{desc}:")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("  FAILED to open")
        continue
    
    try:
        # Set properties in optimal order for ELP cameras
        fourcc_code = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        time.sleep(0.1)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        time.sleep(0.1)
        
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buf_size)
        
        # ELP-specific optimizations
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
        except:
            pass
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except:
            pass
        try:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        except:
            pass
        
        time.sleep(0.3)
        
        # Check actual settings
        act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        act_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if act_w != 1920 or act_h != 1080:
            print(f"  FAILED: got {act_w}x{act_h}")
            cap.release()
            continue
        
        print(f"  Actual FPS setting: {act_fps:.1f}")
        
        # Flush frames using grab+retrieve
        for _ in range(15):
            cap.grab()
        cap.retrieve()
        
        # Measure FPS with grab+retrieve pattern
        count = 0
        start = time.time()
        duration = 4.0
        
        while time.time() - start < duration:
            if cap.grab():
                # Skip stale frames
                cap.grab()  # Skip one
                ret, _ = cap.retrieve()
                if ret:
                    count += 1
        
        elapsed = time.time() - start
        measured_fps = count / elapsed if elapsed > 0 else 0
        
        print(f"  Measured FPS: {measured_fps:.1f}")
        
        cap.release()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        try:
            cap.release()
        except:
            pass

print("\n" + "=" * 70)
print("Test complete - use the configuration with highest measured FPS")

