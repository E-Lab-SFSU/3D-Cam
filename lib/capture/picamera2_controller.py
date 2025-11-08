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
from dataclasses import dataclass, field
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
ALLOWED_CONTROL_NAMES = {
    "ExposureTime",
    "AnalogueGain",
    "AeEnable",
    "AwbEnable",
    "ColourGains",
    "AeFlickerMode",
    "Brightness",
    "Contrast",
    "Saturation",
    "Sharpness",
    "NoiseReductionMode",
    "ScalerCrop",
    "FrameDurationLimits",
}
DEFAULT_FRAME_RATE = 30
DEFAULT_RESOLUTION = (1920, 1080)
DEFAULT_OUTPUT_DIR = Path("inputs_outputs")
_QT_PLUGIN_HINTS = [
    Path("/usr/lib/arm-linux-gnueabihf/qt5/plugins"),
    Path("/usr/lib/aarch64-linux-gnu/qt5/plugins"),
    Path("/usr/lib/qt/plugins"),
]

def _ensure_qt_plugin_path() -> None:
    """Ensure Qt picks up a system plugin directory if possible."""
    current = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    if current:
        # Some environments set this to OpenCV's plugin dir. Prepend system hints if needed.
        paths = current.split(os.pathsep)
        for hint in _QT_PLUGIN_HINTS:
            if hint.exists() and str(hint) not in paths:
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.pathsep.join([str(hint), *paths])
                break
    else:
        for hint in _QT_PLUGIN_HINTS:
            if hint.exists():
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(hint)
                break

    # Ensure Qt runtime directories are owned by the user with 0700 perms
    uid = os.getuid()
    runtime_dir = Path(f"/run/user/{uid}")
    try:
        if runtime_dir.exists():
            runtime_dir.chmod(0o700)
    except PermissionError:
        pass


def _has_desktop_session() -> bool:
    """Return True if running under X11 or Wayland."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _running_over_ssh() -> bool:
    """Best-effort detection for SSH sessions."""
    return any(env in os.environ for env in ("SSH_CLIENT", "SSH_CONNECTION", "SSH_TTY"))


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
        has_desktop = _has_desktop_session()
        over_ssh = _running_over_ssh()
        if has_desktop:
            order = ["qtgl", "drm", "null"]
        elif over_ssh and not has_desktop:
            order = ["null", "drm", "qtgl"]
        else:
            order = ["drm", "null", "qtgl"]
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
    metadata_path: Path
    start_time: float
    frame_index: int = 0
    first_sensor_ts: Optional[int] = None
    metadata_file: Any = field(default=None, repr=False)


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

        _ensure_qt_plugin_path()

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
        try:
            output.set_timestamps("monotonic")  # Ensure packets carry increasing PTS
        except AttributeError:
            pass
        metadata_path = filepath.with_suffix(".csv")
        metadata_file = metadata_path.open("w", encoding="utf-8", newline="")
        metadata_file.write("frame_index,sensor_timestamp_ns,monotonic_seconds,delta_from_start_ns\n")
        self.picam2.start_recording(encoder, output)
        self._recording_state = RecordingState(
            filepath=filepath,
            encoder=encoder,
            output=output,
            metadata_path=metadata_path,
            start_time=time.perf_counter(),
            metadata_file=metadata_file,
        )
        print(f"[INFO] Recording -> {filepath}")

    def stop_recording(self) -> Path:
        if not self.is_recording:
            raise RuntimeError("No active recording to stop.")
        self.picam2.stop_recording()
        state = self._recording_state
        if state.metadata_file:
            state.metadata_file.flush()
            state.metadata_file.close()
        filepath = state.filepath
        self._recording_state = None
        print(f"[INFO] Recording saved: {filepath}")
        return filepath

    # Stills
    def capture_image(self, filepath: Path) -> Path:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        was_running = self._camera_started

        if was_running:
            try:
                self.picam2.stop()
            finally:
                self._camera_started = False

        try:
            self.picam2.switch_mode_and_capture_file(self._still_config, str(filepath))
        finally:
            try:
                self.picam2.configure(self._preview_config)
            except Exception as exc:
                print(f"[WARN] Failed to restore preview configuration: {exc}")
            else:
                if was_running:
                    if self.framerate:
                        try:
                            self.picam2.set_controls({"FrameRate": self.framerate})
                        except Exception:
                            pass
                    try:
                        self.picam2.start()
                    except Exception as exc:
                        print(f"[WARN] Failed to restart preview: {exc}")
                    else:
                        self._camera_started = True

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
        if self._recording_state and self._recording_state.metadata_file:
            metadata = request.get_metadata()
            sensor_ts = metadata.get("SensorTimestamp")
            state = self._recording_state
            if sensor_ts is not None and state.first_sensor_ts is None:
                state.first_sensor_ts = sensor_ts
            delta_ns = (
                sensor_ts - state.first_sensor_ts
                if (sensor_ts is not None and state.first_sensor_ts is not None)
                else ""
            )
            monotonic_seconds = time.perf_counter() - state.start_time
            line = (
                f"{state.frame_index},"
                f"{sensor_ts or ''},"
                f"{monotonic_seconds:.6f},"
                f"{delta_ns if delta_ns != '' else ''}\n"
            )
            try:
                state.metadata_file.write(line)
                state.metadata_file.flush()
            except Exception:
                pass
            state.frame_index += 1

    def list_controls(self) -> Dict[str, Dict[str, Any]]:
        """
        Return metadata about available camera controls.

        The Picamera2 API exposes control details via ControlInfo objects.
        We normalize those to simple dictionaries for CLI presentation.
        """
        controls: Dict[str, Dict[str, Any]] = {}

        if not hasattr(self.picam2, "camera_controls"):
            return controls

        def _coerce_value(value: Any) -> Any:
            """Best-effort conversion of libcamera control values to plain Python types."""
            if value is None:
                return None
            if isinstance(value, (bool, int, float, str)):
                return value
            if isinstance(value, (list, tuple, set)):
                coerced = [_coerce_value(v) for v in value]
                return type(value)(coerced)

            # libcamera ControlValue exposes a 'value' attribute in recent releases.
            for attr in ("value", "values", "numerator", "denominator"):
                if hasattr(value, attr):
                    try:
                        candidate = getattr(value, attr)
                        if callable(candidate):
                            candidate = candidate()
                    except Exception:
                        continue
                    coerced = _coerce_value(candidate)
                    if coerced is not None:
                        return coerced

            # Fall back to string representation (trim overly long dumps).
            text = str(value)
            return text if len(text) <= 200 else f"{text[:197]}..."

        def _extract(info: Any, *names: str) -> Any:
            """Try several attribute/key names to obtain a value."""
            for name in names:
                if hasattr(info, name):
                    try:
                        result = getattr(info, name)
                        if callable(result):
                            result = result()
                    except Exception:
                        pass
                    else:
                        coerced = _coerce_value(result)
                        if coerced is not None:
                            return coerced

            if isinstance(info, dict):
                for name in names:
                    if name in info:
                        coerced = _coerce_value(info[name])
                        if coerced is not None:
                            return coerced

            if hasattr(info, "__getitem__"):
                for name in names:
                    try:
                        result = info[name]
                    except Exception:
                        continue
                    coerced = _coerce_value(result)
                    if coerced is not None:
                        return coerced

            return None

        for name, info in self.picam2.camera_controls.items():  # type: ignore[attr-defined]
            if name not in ALLOWED_CONTROL_NAMES:
                continue
            ctrl_type = _extract(info, "type")
            type_name = getattr(ctrl_type, "__name__", None) or str(ctrl_type)
            controls[name] = {
                "default": _extract(info, "default", "def"),
                "min": _extract(info, "min"),
                "max": _extract(info, "max"),
                "step": _extract(info, "step"),
                "values": _extract(info, "values"),
                "type": type_name,
            }

            # If everything failed, include a raw representation for debugging.
            if all(value is None for key, value in controls[name].items() if key != "type"):
                controls[name]["raw"] = str(info)

        return controls

    def set_control(self, name: str, value: Any) -> None:
        """Thin wrapper around Picamera2.set_controls with simple error propagation."""
        self.picam2.set_controls({name: value})

    def close(self) -> None:
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


def sanitize_name(name: str) -> str:
    """
    Sanitize a user-provided name so it is safe for filesystem usage.

    - Converts whitespace to underscores
    - Removes characters outside of [A-Za-z0-9._-]
    - Strips leading/trailing separators
    - Guarantees a non-empty string by falling back to 'capture'
    """
    cleaned = re.sub(r"\s+", "_", name.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or "capture"


def build_output_path(stem: str, extension: str, base_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    """
    Construct an output path with a date-prefixed directory and filename:
        inputs_outputs/<YYYYMMDD_stem>/<YYYYMMDD_stem>.<extension>

    If the target file already exists, append a numeric suffix to both the
    directory and filename (e.g., <YYYYMMDD_stem>_01).
    """
    safe_stem = sanitize_name(stem)
    ext = extension.lstrip(".")
    root = Path(base_dir)

    from datetime import datetime

    timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp_prefix}_{safe_stem or 'capture'}"

    candidate_name = base_name
    counter = 1

    while True:
        directory = root / candidate_name
        filepath = directory / f"{candidate_name}.{ext}"
        if not filepath.exists():
            return filepath
        candidate_name = f"{base_name}_{counter:02d}"
        counter += 1
