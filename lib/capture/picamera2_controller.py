#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Picamera2 Controller (clean version)
------------------------------------
Thin wrapper for Picamera2 preview, FPS tracking, recording, and still capture.
Minimal backend logic like bare working version.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from picamera2 import Picamera2, Preview
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import FfmpegOutput
except ImportError as exc:
    Picamera2 = None  # type: ignore
    Preview = None
    H264Encoder = None
    FfmpegOutput = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

DEFAULT_RECORDING_BITRATE = 12_000_000
DEFAULT_FRAME_RATE = 30
DEFAULT_RESOLUTION = (1920, 1080)


def _has_desktop_session() -> bool:
    """Return True if running under X11 or Wayland."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _start_best_preview(picam2: Picamera2, backend: str = "auto") -> str:
    """
    Start a preview using a simple rule:
      - If desktop session → try QTGL then DRM
      - Else → try DRM then QTGL
      - NULL only if explicitly requested
    """
    if Preview is None:
        raise RuntimeError("Picamera2 Preview is unavailable.")

    backends = {
        "qtgl": Preview.QTGL,
        "drm": Preview.DRM,
        "null": Preview.NULL,
    }

    if backend not in ("auto", "qtgl", "drm", "null"):
        raise ValueError("backend must be one of: auto, qtgl, drm, null")

    # Pick order based on simple desktop detection
    if backend == "auto":
        order = ["qtgl", "drm"] if _has_desktop_session() else ["drm", "qtgl"]
    else:
        order = [backend]

    last_exc: Optional[Exception] = None
    for name in order:
        try:
            picam2.start_preview(backends[name])
            return name
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(f"Failed to start preview with {order}: {last_exc}")


class FPSTracker:
    def __init__(self, window_size: int = 30):
        self.timestamps = deque(maxlen=window_size)

    def update(self) -> None:
        self.timestamps.append(time.time())

    def get_fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / dt if dt > 0 else 0.0


@dataclass
class RecordingState:
    filepath: Path
    encoder: H264Encoder
    output: FfmpegOutput


class Picamera2Controller:
    def __init__(
        self,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        framerate: int = DEFAULT_FRAME_RATE,
        backend: str = "auto",
    ) -> None:

        if Picamera2 is None:
            raise ImportError(
                "Picamera2 library not found. Install picamera2 on Raspberry Pi."
            ) from _IMPORT_ERROR

        self.resolution = resolution
        self.framerate = framerate
        self.backend = backend

        self.picam2 = Picamera2()
        self._preview_config = self.picam2.create_preview_configuration(
            main={"size": resolution}
        )
        self._still_config = self.picam2.create_still_configuration(
            main={"size": resolution}
        )
        self.picam2.configure(self._preview_config)

        self.fps_tracker = FPSTracker()
        self.picam2.post_callback = self._frame_callback

        self._preview_backend: Optional[str] = None
        self._camera_started = False
        self._recording_state: Optional[RecordingState] = None

    # Start / Stop
    def start(self) -> None:
        if self._preview_backend is None:
            self._preview_backend = _start_best_preview(self.picam2, self.backend)
            print(f"[INFO] Preview backend: {self._preview_backend}")

        if not self._camera_started:
            if self.framerate:
                try:
                    self.picam2.set_controls({"FrameRate": self.framerate})
                except Exception:
                    pass
            self.picam2.start()
            self._camera_started = True
            print("[INFO] Camera started.")

    def stop(self) -> None:
        if self.is_recording:
            self.stop_recording()
        if self._camera_started:
            self.picam2.stop()
            self._camera_started = False
            print("[INFO] Camera stopped.")
        if self._preview_backend:
            self.picam2.stop_preview()
            self._preview_backend = None
            print("[INFO] Preview stopped.")

    @property
    def is_recording(self) -> bool:
        return self._recording_state is not None

    # Recording
    def start_recording(self, filepath: Path, bitrate: int = DEFAULT_RECORDING_BITRATE) -> None:
        if self.is_recording:
            raise RuntimeError("Recording already in progress.")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        encoder = H264Encoder(bitrate=bitrate)
        output = FfmpegOutput(str(filepath))
        self.picam2.start_recording(encoder, output)
        self._recording_state = RecordingState(filepath, encoder, output)
        print(f"[INFO] Recording -> {filepath}")

    def stop_recording(self) -> Path:
        if not self.is_recording:
            raise RuntimeError("No active recording to stop.")
        self.picam2.stop_recording()
        filepath = self._recording_state.filepath
        self._recording_state = None
        print(f"[INFO] Recording saved: {filepath}")
        return filepath

    # Stills
    def capture_image(self, filepath: Path) -> Path:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.picam2.switch_mode_and_capture_file(self._still_config, str(filepath))
        self.picam2.configure(self._preview_config)
        self.picam2.start()
        print(f"[INFO] Still captured: {filepath}")
        return filepath

    # Camera controls
    def get_fps(self) -> float:
        return self.fps_tracker.get_fps()

    def status(self) -> Dict[str, Any]:
        return {
            "preview_backend": self._preview_backend,
            "camera_started": self._camera_started,
            "recording": self.is_recording,
            "resolution": self.resolution,
            "framerate": self.framerate,
            "fps": self.get_fps(),
        }

    def _frame_callback(self, request: Any) -> None:
        self.fps_tracker.update()

    def close(self) -> None:
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
