#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi Picamera2 Preview
------------------------------

Minimal Picamera2 preview loop that mirrors the working snippet provided by the
user, but leverages the reusable controller in `lib.capture.picamera2_controller`.
"""

import argparse
import sys
import time
from contextlib import suppress
from pathlib import Path

# Ensure project root is on sys.path when running as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lib.capture import Picamera2Controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Picamera2 preview helper with FPS display."
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "drm", "qtgl", "null"],
        help="Preview backend to use.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target framerate (FrameRate control).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    controller = Picamera2Controller(
        resolution=(1920, 1080),
        framerate=args.fps,
        backend=args.backend,
    )

    try:
        controller.start()
    except Exception as exc:
        controller.stop()
        raise RuntimeError(
            f"Camera/preview start failed: {exc}"
        ) from exc

    print("Camera preview started. Press Ctrl+C to exit.")
    last_update = time.time()

    try:
        while True:
            time.sleep(0.1)
            now = time.time()
            if now - last_update >= 1.0:
                fps = controller.get_fps()
                print(f"\rFPS: {fps:.1f}", end="", flush=True)
                last_update = now
    except KeyboardInterrupt:
        print()
    finally:
        with suppress(Exception):
            controller.stop()


if __name__ == "__main__":
    main()
