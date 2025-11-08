#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Picamera2 Controller
--------------------

Thin wrapper around Picamera2 offering preview management, FPS tracking,
recording helpers, still capture utilities, and simple control inspection.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    from picamera2 import Picamera2, Preview
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import FfmpegOutput
except ImportError as exc:  # pragma: no cover - hardware dependency
    Picamera2 = None  # type: ignore[assignment]
    Preview = None  # type: ignore[assignment]
    H264Encoder = None  # type: ignore[assignment]
    FfmpegOutput = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

DEFAULT_RECORDING_BITRATE = 12_000_000  # 12 Mbps default bitrate for 1080p
DEFAULT_FRAME_RATE = 30
DEFAULT_RESOLUTION = (1920, 1080)


def _has_wayland_socket() -> bool:
    wayland = os.environ.get("WAYLAND_DISPLAY")
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if not wayland or not runtime_dir:
        return False
    return (Path(runtime_dir) / wayland).exists()


def _has_x11_socket() -> bool:
    display = os.environ.get("DISPLAY", "")
    if not display:
        return False
    display_name = display.split(".", 1)[0]
    if display_name.startswith(":"):
        display_name = display_name[1:]
    display_name = display_name or "0"
    return (Path("/tmp/.X11-unix") / f"X{display_name}").exists()


def _has_desktop_session() -> bool:
    """Heuristically detect if an interactive desktop session is active."""
    display = os.environ.get("DISPLAY")
    wayland = os.environ.get("WAYLAND_DISPLAY")
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    ssh_connection = os.environ.get("SSH_CONNECTION")
    ssh_client = os.environ.get("SSH_CLIENT")
    ssh_tty = os.environ.get("SSH_TTY")
    force_desktop = os.environ.get("PICAMERA2_FORCE_DESKTOP", "").lower()
    force_headless = os.environ.get("PICAMERA2_FORCE_HEADLESS", "").lower()
    x11_socket = _has_x11_socket()
    wayland_socket = _has_wayland_socket()
    print(
        "[DEBUG] Session env:",
        {
            "DISPLAY": display,
            "WAYLAND_DISPLAY": wayland,
            "XDG_SESSION_TYPE": session_type,
            "PICAMERA2_FORCE_DESKTOP": force_desktop,
            "PICAMERA2_FORCE_HEADLESS": force_headless,
            "SSH_CONNECTION": ssh_connection,
            "SSH_CLIENT": ssh_client,
            "SSH_TTY": ssh_tty,
            "HAS_X11_SOCKET": x11_socket,
            "HAS_WAYLAND_SOCKET": wayland_socket,
        },
    )
    if force_headless in {"1", "true", "yes"}:
        print("[DEBUG] Headless forced via PICAMERA2_FORCE_HEADLESS.")
        return False
    if force_desktop in {"1", "true", "yes"}:
        print("[DEBUG] Desktop forced via PICAMERA2_FORCE_DESKTOP.")
        return True
    if ssh_connection or ssh_client or ssh_tty:
        print("[DEBUG] SSH session detected; treating as headless.")
        return False
    if wayland_socket or x11_socket:
        return True
    return False


def sanitize_name(name: str) -> str:
    """Sanitize filename (and folder) stems."""
    clean = re.sub(r"[<>:\"/\\\\|?*]", "_", name.strip())
    clean = re.sub(r"\\s+", "_", clean)
    return clean or time.strftime("capture_%Y%m%d_%H%M%S")


def build_output_path(stem: str, ext: str, base_dir: Optional[Path] = None) -> Path:
    """
    Build `inputs_outputs/<stem>/<stem>.<ext>` path, ensuring directory exists.

    Args:
        stem: Desired filename stem. Will be sanitized.
        ext: File extension without dot (e.g. "mp4").
        base_dir: Optional base directory override (defaults to cwd/inputs_outputs).
    """
    safe_stem = sanitize_name(stem)
    root = Path(base_dir) if base_dir is not None else Path("inputs_outputs")
    folder = root / safe_stem
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{safe_stem}.{ext.lstrip('.')}"


def _start_best_preview(picam2: Picamera2, backend: str = "auto") -> str:
    """
    Start a preview using the requested backend, or pick a sensible default.

    - DRM works well on the console (no X/Wayland).
    - QTGL works under a desktop session with OpenGL.
    - NULL runs headless (no visible preview), useful for diagnostics.
    """
    if Preview is None:  # pragma: no cover - defensive
        raise RuntimeError("Picamera2 Preview is unavailable.")

    backends = {
        "drm": Preview.DRM,
        "qtgl": Preview.QTGL,
        "null": Preview.NULL,
    }

    if backend not in ("auto", *backends.keys()):
        raise ValueError(f"Unknown backend '{backend}'. Valid: auto, drm, qtgl, null")

    if backend == "auto":
        if _has_desktop_session():
            print("[DEBUG] Desktop session detected; trying qtgl then drm.")
            order = ["qtgl", "drm", "null"]
        else:
            print("[DEBUG] No desktop session; trying drm then qtgl.")
            order = ["drm", "qtgl", "null"]
    else:
        order = [backend]

    last_exc: Optional[Exception] = None

    for name in order:
        try:
            print(f"[DEBUG] Attempting preview backend: {name}")
            picam2.start_preview(backends[name])
            print(f"[DEBUG] Preview backend started: {name}")
            return name
        except Exception as exc:  # pragma: no cover - hardware dependency
            print(f"[WARN] Preview backend {name} failed: {exc}")
            last_exc = exc

    raise RuntimeError(f"Failed to start preview using {order}: {last_exc}")  # pragma: no cover


class FPSTracker:
    """Track FPS by monitoring frame timestamps."""

    def __init__(self, window_size: int = 30) -> None:
        self.timestamps: deque[float] = deque(maxlen=window_size)

    def update(self) -> None:
        """Record a new frame timestamp."""
        self.timestamps.append(time.time())

    def get_fps(self) -> float:
        """Calculate FPS based on recent frame timestamps."""
        if len(self.timestamps) < 2:
            return 0.0

        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span <= 0:
            return 0.0

        return (len(self.timestamps) - 1) / time_span


@dataclass
class RecordingState:
    """Track active recording resources."""

    filepath: Path
    encoder: H264Encoder
    output: FfmpegOutput


class Picamera2Controller:
    """High-level controller for Picamera2 preview, recording, and capture."""

    def __init__(
        self,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        framerate: int = DEFAULT_FRAME_RATE,
        backend: str = "auto",
    ) -> None:
        if Picamera2 is None:  # pragma: no cover - platform guard
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

    # --------------------------------------------------------------------- #
    # Lifecycle management
    # --------------------------------------------------------------------- #
    def start(self) -> None:
        """Start preview (if needed) and begin camera capture."""
        if self._preview_backend is None:
            self._preview_backend = _start_best_preview(self.picam2, self.backend)
            print(f"[INFO] Picamera2 preview running via backend: {self._preview_backend}")

        if not self._camera_started:
            controls = {"FrameRate": self.framerate} if self.framerate else {}
            if controls:
                try:
                    self.picam2.set_controls(controls)
                except Exception as exc:  # pragma: no cover - hardware dependency
                    print(f"[WARN] Could not set initial controls {controls}: {exc}")

            self.picam2.start()
            self._camera_started = True
            print("[INFO] Picamera2 camera stream started.")

    def stop(self) -> None:
        """Stop recording (if active), camera stream, and preview."""
        if self.is_recording:
            self.stop_recording()

        if self._camera_started:
            self.picam2.stop()
            self._camera_started = False
            print("[INFO] Picamera2 camera stream stopped.")

        if self._preview_backend is not None:
            self.picam2.stop_preview()
            print("[INFO] Picamera2 preview stopped.")
            self._preview_backend = None

    # --------------------------------------------------------------------- #
    # Recording
    # --------------------------------------------------------------------- #
    @property
    def is_recording(self) -> bool:
        return self._recording_state is not None

    def start_recording(self, filepath: Path, bitrate: int = DEFAULT_RECORDING_BITRATE) -> None:
        """Begin video recording to provided path."""
        if self.is_recording:
            raise RuntimeError("Recording already in progress.")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        encoder = H264Encoder(bitrate=bitrate)
        output = FfmpegOutput(str(filepath))
        self.picam2.start_recording(encoder, output)
        self._recording_state = RecordingState(filepath=filepath, encoder=encoder, output=output)
        print(f"[INFO] Recording started: {filepath}")

    def stop_recording(self) -> Path:
        """Stop video recording and return the recorded file path."""
        if not self.is_recording:
            raise RuntimeError("No active recording to stop.")

        assert self._recording_state is not None
        self.picam2.stop_recording()
        filepath = self._recording_state.filepath
        self._recording_state = None
        print(f"[INFO] Recording stopped: {filepath}")
        return filepath

    # --------------------------------------------------------------------- #
    # Still capture
    # --------------------------------------------------------------------- #
    def capture_image(self, filepath: Path) -> Path:
        """Capture a still image and save to the given path."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Use switch_mode_and_capture_file to get full-resolution stills.
        self.picam2.switch_mode_and_capture_file(self._still_config, str(filepath))
        # Restore preview configuration after still capture.
        self.picam2.configure(self._preview_config)
        self.picam2.start()
        print(f"[INFO] Captured still image: {filepath}")
        return filepath

    # --------------------------------------------------------------------- #
    # Control helpers
    # --------------------------------------------------------------------- #
    def list_controls(self) -> Dict[str, Dict[str, Any]]:
        """Return metadata for available camera controls."""
        info = {}
        for name, ctrl in self.picam2.camera_controls.items():
            info[name] = {
                "type": type(ctrl).__name__,
                "default": getattr(ctrl, "default", None),
                "min": getattr(ctrl, "min", None),
                "max": getattr(ctrl, "max", None),
            }
        return info

    def get_control(self, name: str) -> Any:
        """Fetch the current value of a camera control."""
        controls = self.picam2.controls
        return getattr(controls, name, None)

    def set_control(self, name: str, value: Any) -> None:
        """Set a camera control."""
        self.picam2.set_controls({name: value})
        print(f"[INFO] Control '{name}' set to {value}")

    # --------------------------------------------------------------------- #
    # Monitoring
    # --------------------------------------------------------------------- #
    def get_fps(self) -> float:
        """Return the current FPS estimate."""
        return self.fps_tracker.get_fps()

    def status(self) -> Dict[str, Any]:
        """Return a dict summarizing current controller state."""
        return {
            "preview_backend": self._preview_backend,
            "camera_started": self._camera_started,
            "recording": self.is_recording,
            "resolution": self.resolution,
            "framerate": self.framerate,
            "fps": self.get_fps(),
        }

    # --------------------------------------------------------------------- #
    # Internal callbacks
    # --------------------------------------------------------------------- #
    def _frame_callback(self, request: Any) -> None:
        """Callback fired for each camera frame."""
        self.fps_tracker.update()

    # --------------------------------------------------------------------- #
    # Utility context manager
    # --------------------------------------------------------------------- #
    def close(self) -> None:
        """Alias for stop()."""
        self.stop()

    def __enter__(self) -> "Picamera2Controller":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


