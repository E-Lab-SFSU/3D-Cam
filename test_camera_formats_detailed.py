#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed Camera Format Test
Tests different approaches to get higher FPS at 1920x1080
"""

import cv2
import time
import platform
import sys


def test_with_settings(camera_index, fourcc, width, height, backend_name, backend_code, 
                       use_grab_retrieve=False, fps_setting=0, buffer_size=1):
    """Test with specific settings."""
    
    cap = cv2.VideoCapture(camera_index, backend_code)
    if not cap.isOpened():
        return None
    
    try:
        # Set format
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        time.sleep(0.1)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        time.sleep(0.1)
        
        cap.set(cv2.CAP_PROP_FPS, fps_setting)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        
        time.sleep(0.2)
        
        # Check settings
        act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if act_w != width or act_h != height:
            cap.release()
            return None
        
        # Flush frames
        for _ in range(10):
            if use_grab_retrieve:
                cap.grab()
            else:
                cap.read()
        
        if use_grab_retrieve:
            cap.retrieve()
        
        # Measure FPS
        frame_count = 0
        start = time.time()
        duration = 3.0
        
        while time.time() - start < duration:
            if use_grab_retrieve:
                if not cap.grab():
                    continue
                # Skip stale frames for MJPG
                if fourcc == "MJPG":
                    cap.grab()  # Skip one more
                ret, frame = cap.retrieve()
            else:
                ret, frame = cap.read()
            
            if ret and frame is not None:
                frame_count += 1
        
        elapsed = time.time() - start
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        cap.release()
        return fps
        
    except Exception as e:
        cap.release()
        return None


def main():
    camera_index = 0
    if len(sys.argv) > 1:
        camera_index = int(sys.argv[1])
    
    print("=" * 70)
    print(f"Detailed Camera Format Test - Camera {camera_index}")
    print("=" * 70)
    
    if platform.system() == "Windows":
        backends = [(cv2.CAP_DSHOW, "DSHOW")]
    else:
        backends = [(cv2.CAP_V4L2, "V4L2")]
    
    # Test configurations
    configs = [
        ("MJPG", 1920, 1080, False, 0, 1, "MJPG, max FPS, buffer=1"),
        ("MJPG", 1920, 1080, True, 0, 1, "MJPG, grab+retrieve, max FPS"),
        ("MJPG", 1920, 1080, True, 30, 1, "MJPG, grab+retrieve, 30 FPS"),
        ("MJPG", 1920, 1080, False, 30, 1, "MJPG, 30 FPS set"),
        ("MJPG", 1920, 1080, False, 60, 1, "MJPG, 60 FPS set"),
        ("MJPG", 1280, 720, False, 0, 1, "MJPG, 1280x720, max FPS"),
    ]
    
    print("\nTesting configurations...\n")
    
    for fourcc, width, height, use_gr, fps_set, buf_size, desc in configs:
        print(f"{desc}:")
        
        for backend_code, backend_name in backends:
            print(f"  {backend_name}: ", end="", flush=True)
            fps = test_with_settings(
                camera_index, fourcc, width, height, backend_name, backend_code,
                use_grab_retrieve=use_gr, fps_setting=fps_set, buffer_size=buf_size
            )
            
            if fps:
                print(f"{fps:.1f} FPS")
            else:
                print("FAILED")
        
        print()
    
    print("=" * 70)
    print("\nTry the configuration with the highest FPS!")


if __name__ == "__main__":
    main()

