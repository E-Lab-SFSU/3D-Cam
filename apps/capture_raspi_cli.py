#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi Picamera2 CLI
--------------------------

Interactive command-line interface for Picamera2 that provides a live preview,
start/stop recording, still capture, and camera control adjustments.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when running as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lib.capture import Picamera2Controller, build_output_path, sanitize_name


TEMP_DIR = Path("inputs_outputs") / "_tmp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Picamera2 CLI with preview, recording, and capture."
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
    parser.add_argument(
        "--bitrate",
        type=int,
        default=12_000_000,
        help="Recording bitrate in bits per second (default: 12 Mbps).",
    )
    return parser.parse_args()


def ensure_temp_dir() -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_DIR


def prompt_filename(default_stem: str) -> str:
    """Prompt the user for a filename stem."""
    raw = input(f"Enter filename (default '{default_stem}'): ").strip()
    return sanitize_name(raw) if raw else default_stem


def summarize_commands() -> None:
    print(
        "\nCommands:\n"
        "  help                Show this help message\n"
        "  status              Display controller status information\n"
        "  record start        Begin recording (H.264 @ 1080p)\n"
        "  record stop         Stop recording and save to inputs_outputs/<name>/<name>.mp4\n"
        "  capture             Capture a still image to inputs_outputs/<name>/<name>.jpg\n"
        "  controls            List available camera controls\n"
        "  set <name> <value>  Set a camera control (e.g., set AnalogueGain 1.5)\n"
        "  quit / exit         Stop preview and exit\n"
    )


def list_controls(controller: Picamera2Controller) -> None:
    controls = controller.list_controls()
    print(f"Found {len(controls)} controls:")
    for name, meta in sorted(controls.items()):
        print(
            f"  {name}: default={meta['default']} range=({meta['min']}, {meta['max']}) type={meta['type']}"
        )


def run_cli(controller: Picamera2Controller, bitrate: int) -> None:
    summarize_commands()
    last_fps_update = time.time()

    while True:
        try:
            if time.time() - last_fps_update >= 1.0:
                fps = controller.get_fps()
                status = controller.status()
                print(
                    f"\rPreview: {status['preview_backend']} | Recording: {status['recording']} | FPS: {fps:.1f}   ",
                    end="",
                    flush=True,
                )
                last_fps_update = time.time()

            cmd = input("\n> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not cmd:
            continue

        parts = cmd.split()
        action = parts[0].lower()

        if action in {"quit", "exit"}:
            break

        if action == "help":
            summarize_commands()
            continue

        if action == "status":
            info = controller.status()
            info["fps"] = controller.get_fps()
            print(info)
            continue

        if action == "controls":
            list_controls(controller)
            continue

        if action == "set":
            if len(parts) < 3:
                print("Usage: set <control_name> <value>")
                continue
            name = parts[1]
            value = " ".join(parts[2:])
            try:
                # Attempt to parse numeric types automatically
                if value.isdigit():
                    parsed: object = int(value)
                else:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value
                controller.set_control(name, parsed)
            except Exception as exc:  # pragma: no cover - hardware dependent
                print(f"[ERROR] Failed to set control '{name}': {exc}")
            continue

        if action == "record":
            if len(parts) == 1:
                print("Usage: record <start|stop>")
                continue
            sub = parts[1].lower()
            if sub == "start":
                if controller.is_recording:
                    print("Recording already in progress.")
                    continue

                ensure_temp_dir()
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                temp_path = TEMP_DIR / f"temp_video_{timestamp}.mp4"
                try:
                    controller.start_recording(temp_path, bitrate=bitrate)
                except Exception as exc:  # pragma: no cover - hardware dependent
                    print(f"[ERROR] Failed to start recording: {exc}")
                else:
                    print(f"Recording... temporary file: {temp_path}")
                continue

            if sub == "stop":
                if not controller.is_recording:
                    print("No active recording.")
                    continue

                try:
                    temp_path = controller.stop_recording()
                except Exception as exc:  # pragma: no cover - hardware dependent
                    print(f"[ERROR] Failed to stop recording: {exc}")
                    continue

                default_stem = temp_path.stem.replace("temp_video_", "video_")
                stem = prompt_filename(default_stem)
                final_path = build_output_path(stem, "mp4")

                try:
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(temp_path), final_path)
                    print(f"[INFO] Saved recording to: {final_path.resolve()}")
                except Exception as exc:  # pragma: no cover - hardware dependent
                    print(f"[ERROR] Failed to save recording: {exc}")
                continue

            print("Usage: record <start|stop>")
            continue

        if action == "capture":
            ensure_temp_dir()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            temp_path = TEMP_DIR / f"temp_image_{timestamp}.jpg"
            try:
                controller.capture_image(temp_path)
            except Exception as exc:  # pragma: no cover - hardware dependent
                print(f"[ERROR] Failed to capture still: {exc}")
                continue

            default_stem = temp_path.stem.replace("temp_image_", "image_")
            stem = prompt_filename(default_stem)
            final_path = build_output_path(stem, "jpg")

            try:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(temp_path), final_path)
                print(f"[INFO] Saved still image to: {final_path.resolve()}")
            except Exception as exc:  # pragma: no cover - hardware dependent
                print(f"[ERROR] Failed to save still image: {exc}")
            continue

        print(f"Unknown command: {cmd!r}. Type 'help' for options.")

    # Cleanup temporary directory if empty.
    with suppress_cleanup():
        if TEMP_DIR.is_dir() and not any(TEMP_DIR.iterdir()):
            TEMP_DIR.rmdir()


class suppress_cleanup:
    """Context manager to swallow cleanup errors."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return True  # suppress all exceptions


def main() -> None:
    args = parse_args()

    try:
        controller = Picamera2Controller(
            resolution=(1920, 1080),
            framerate=args.fps,
            backend=args.backend,
        )
    except ImportError as exc:
        print(exc)
        sys.exit(1)

    try:
        controller.start()
    except Exception as exc:
        controller.stop()
        print(f"[ERROR] Failed to start preview: {exc}")
        sys.exit(2)

    print("Picamera2 preview running. Use the CLI to control capture. Ctrl+C to exit.")

    try:
        run_cli(controller, bitrate=args.bitrate)
    finally:
        controller.stop()


if __name__ == "__main__":
    main()


