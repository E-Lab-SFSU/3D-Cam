#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct MJPG 1920x1080 FPS Test
Tests different methods to get higher FPS
"""

import cv2
import time
import sys

camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

print("=" * 70)
print(f"MJPG 1920x1080 FPS Test - Camera {camera_index}")
print("=" * 70)

# Test 1: Standard read()
print("\nTest 1: Standard read() with FPS=0 (max speed)")
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 0)  # Max speed
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    time.sleep(0.3)
    
    # Flush
    for _ in range(10):
        cap.read()
    
    # Measure
    count = 0
    start = time.time()
    while time.time() - start < 3.0:
        ret, _ = cap.read()
        if ret:
            count += 1
    
    fps1 = count / 3.0
    print(f"  Result: {fps1:.1f} FPS")
    cap.release()

# Test 2: read() with FPS=30
print("\nTest 2: Standard read() with FPS=30")
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    time.sleep(0.3)
    
    for _ in range(10):
        cap.read()
    
    count = 0
    start = time.time()
    while time.time() - start < 3.0:
        ret, _ = cap.read()
        if ret:
            count += 1
    
    fps2 = count / 3.0
    print(f"  Result: {fps2:.1f} FPS")
    cap.release()

# Test 3: grab()+retrieve() pattern
print("\nTest 3: grab()+retrieve() pattern (skip stale frames)")
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Need buffer for skipping
    
    time.sleep(0.3)
    
    for _ in range(10):
        cap.grab()
    cap.retrieve()
    
    count = 0
    start = time.time()
    while time.time() - start < 3.0:
        if cap.grab():
            cap.grab()  # Skip one stale frame
            ret, _ = cap.retrieve()
            if ret:
                count += 1
    
    fps3 = count / 3.0
    print(f"  Result: {fps3:.1f} FPS")
    cap.release()

# Test 4: Try different buffer sizes
print("\nTest 4: Buffer size = 2, FPS=30")
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    time.sleep(0.3)
    
    for _ in range(10):
        cap.read()
    
    count = 0
    start = time.time()
    while time.time() - start < 3.0:
        ret, _ = cap.read()
        if ret:
            count += 1
    
    fps4 = count / 3.0
    print(f"  Result: {fps4:.1f} FPS")
    cap.release()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Test 1 (FPS=0):        {fps1:.1f} FPS")
print(f"Test 2 (FPS=30):       {fps2:.1f} FPS")
print(f"Test 3 (grab+retrieve): {fps3:.1f} FPS")
print(f"Test 4 (buffer=2):     {fps4:.1f} FPS")

best = max(fps1, fps2, fps3, fps4)
print(f"\nBest: {best:.1f} FPS")

