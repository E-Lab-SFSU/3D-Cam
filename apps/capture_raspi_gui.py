#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi Picamera2 GUI
--------------------------

Simple Tkinter interface around Picamera2Controller that mirrors the CLI
features while exposing key camera controls via sliders, toggles, and dialogs.

Requires a Raspberry Pi with Picamera2 installed and access to a display (or
VC4-enabled virtual display). Designed to be lightweight for field usage.
"""

from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import Any

# Ensure project root is on sys.path when running as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lib.capture.picamera2_controller import (
    Picamera2Controller,
    build_output_path,
    sanitize_name,
)

TEMP_DIR = Path("inputs_outputs") / "_tmp"


def ensure_temp_dir() -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_DIR


class RaspiCaptureGUI:
    """Minimal Tkinter front-end for Picamera2."""

    SLIDER_CONTROLS = {
        "ExposureTime": {
            "scale_factor": 1.0,
            "value_type": "int",
            "display_format": "{value:.0f} μs",
            "fallback": {"min": 100, "max": 1_000_000, "step": 100, "default": 10_000},
        },
        "AnalogueGain": {
            "scale_factor": 0.01,
            "value_type": "float",
            "display_format": "{value:.2f}×",
            "fallback": {"min": 100, "max": 1_600, "step": 10, "default": 100},
        },
        "Brightness": {
            "scale_factor": 0.01,
            "value_type": "float",
            "display_format": "{value:+.2f}",
            "fallback": {"min": -100, "max": 100, "step": 5, "default": 0},
        },
        "Contrast": {
            "scale_factor": 0.01,
            "value_type": "float",
            "display_format": "{value:.2f}",
            "fallback": {"min": 0, "max": 200, "step": 5, "default": 100},
        },
        "Saturation": {
            "scale_factor": 0.01,
            "value_type": "float",
            "display_format": "{value:.2f}",
            "fallback": {"min": 0, "max": 200, "step": 5, "default": 100},
        },
        "Sharpness": {
            "scale_factor": 0.01,
            "value_type": "float",
            "display_format": "{value:.2f}",
            "fallback": {"min": 0, "max": 200, "step": 5, "default": 100},
        },
    }

    TOGGLE_CONTROLS = {
        "AeEnable": {"fallback": True},
        "AwbEnable": {"fallback": True},
    }

    OPTION_CONTROLS = {
        "AeFlickerMode": {
            "fallback_values": [0, 1, 2],
            "labels": {0: "Off", 1: "50 Hz", 2: "60 Hz"},
            "fallback_default": 0,
        },
        "NoiseReductionMode": {
            "fallback_values": [0, 1, 2, 3],
            "labels": {
                0: "Off",
                1: "Minimal",
                2: "Fast",
                3: "High Quality",
            },
            "fallback_default": 0,
        },
    }

    VECTOR_CONTROLS = {
        "ColourGains": {
            "length": 2,
            "value_type": "float",
            "labels": ["Red Gain", "Blue Gain"],
            "fallback": [1.5, 1.5],
            "display_format": "{values[0]:.2f}, {values[1]:.2f}",
        },
        "ScalerCrop": {
            "length": 4,
            "value_type": "int",
            "labels": ["X", "Y", "Width", "Height"],
            "fallback": [0, 0, 0, 0],
            "display_format": "{values[0]}, {values[1]}, {values[2]}, {values[3]}",
        },
        "FrameDurationLimits": {
            "length": 2,
            "value_type": "int",
            "labels": ["Min μs", "Max μs"],
            "fallback": [1_000, 33_333],
            "display_format": "{values[0]} μs, {values[1]} μs",
        },
    }

    def __init__(
        self,
        root: tk.Tk,
        controller: Picamera2Controller,
        debug: bool = False,
        initial_size: tuple[int, int] | None = None,
    ) -> None:
        self.root = root
        self.controller = controller
        self.root.title("3D-Cam | Raspberry Pi Capture")
        self.debug = debug
        self._requested_size = initial_size or (690, 570)
        self._geometry_applied = False

        self.status_var = tk.StringVar(value="Initializing camera…")
        self.preview_fps_var = tk.StringVar(value="FPS: —")
        self.record_timer_var = tk.StringVar(value="")

        self.control_vars: dict[str, tk.IntVar] = {}
        self.slider_widgets: dict[str, dict[str, tk.Widget]] = {}
        self.slider_defaults: dict[str, int] = {}
        self.toggle_vars: dict[str, tk.BooleanVar] = {}
        self.toggle_defaults: dict[str, bool] = {}
        self.option_vars: dict[str, tk.StringVar] = {}
        self.option_defaults: dict[str, str] = {}
        self.option_value_maps: dict[str, dict[str, Any]] = {}
        self.vector_labels: dict[str, tk.Widget] = {}
        self.vector_values: dict[str, list[Any]] = {}
        self.vector_defaults: dict[str, tuple[Any, ...]] = {}
        self._last_reported_size: tuple[int, int] | None = None

        self._record_indicator_job: str | None = None
        self._record_indicator_visible = False
        self._recording_started_at: float | None = None

        self.recording = False
        self._status_job: str | None = None
        self._fps_job: str | None = None

        self._build_ui()
        self._stop_recording_indicator()
        self.root.bind("<Configure>", self._on_root_configure)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        try:
            self.controller.start()
        except Exception as exc:  # pragma: no cover - hardware dependent
            messagebox.showerror("Camera Error", f"Unable to start preview:\n{exc}")
        else:
            self._populate_controls()
            self.update_status()
            self.update_fps()

    # UI Construction -----------------------------------------------------
    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding="6")
        main.pack(fill=tk.BOTH, expand=True)

        # Status row
        status_row = ttk.Frame(main)
        status_row.pack(fill=tk.X, pady=(0, 6))
        self.record_indicator_canvas = tk.Canvas(
            status_row, width=14, height=14, highlightthickness=0, bd=0
        )
        self.record_indicator_canvas.pack(side=tk.LEFT, padx=(0, 4))
        self.record_indicator_dot = self.record_indicator_canvas.create_oval(
            2, 2, 12, 12, fill="", outline=""
        )
        self._record_indicator_off_color = self.record_indicator_canvas["background"]
        self._record_indicator_on_color = "#ff3b30"

        ttk.Label(status_row, textvariable=self.record_timer_var, font=("TkDefaultFont", 9, "bold")).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Label(status_row, textvariable=self.status_var).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(status_row, textvariable=self.preview_fps_var).pack(side=tk.LEFT)

        # Buttons
        button_row = ttk.Frame(main)
        button_row.pack(fill=tk.X, pady=(0, 6))
        button_row.columnconfigure((0, 1, 2, 3), weight=1, uniform="buttons")

        self.preview_btn = ttk.Button(button_row, text="Start Preview", command=self.on_toggle_preview)
        self.preview_btn.grid(row=0, column=0, padx=4, sticky="ew")

        self.capture_btn = ttk.Button(button_row, text="Capture Still", command=self.on_capture)
        self.capture_btn.grid(row=0, column=1, padx=4, sticky="ew")

        self.record_btn = ttk.Button(button_row, text="Start Recording", command=self.on_toggle_recording)
        self.record_btn.grid(row=0, column=2, padx=4, sticky="ew")

        self.reset_btn = ttk.Button(button_row, text="Reset Controls", command=self.reset_controls)
        self.reset_btn.grid(row=0, column=3, padx=4, sticky="ew")

        # Control sliders container
        controls_frame = ttk.LabelFrame(main, text="Camera Controls", padding="6")
        controls_frame.pack(fill=tk.BOTH, expand=True)
        self.controls_panel = ttk.Frame(controls_frame)
        self.controls_panel.pack(fill=tk.BOTH, expand=True)

    # Control population --------------------------------------------------
    def _populate_controls(self) -> None:
        controls = self.controller.list_controls()
        for child in self.controls_panel.winfo_children():
            child.destroy()

        self.control_vars.clear()
        self.slider_widgets.clear()
        self.slider_defaults.clear()
        self.toggle_vars.clear()
        self.toggle_defaults.clear()
        self.option_vars.clear()
        self.option_defaults.clear()
        self.option_value_maps.clear()
        self.vector_labels.clear()
        self.vector_values.clear()
        self.vector_defaults.clear()

        row = 0
        created_any = False

        row, created = self._add_slider_controls(row, controls)
        created_any = created_any or created
        row, created = self._add_toggle_controls(row, controls)
        created_any = created_any or created
        row, created = self._add_option_controls(row, controls)
        created_any = created_any or created
        row, created = self._add_vector_controls(row, controls)
        created_any = created_any or created

        if not created_any:
            ttk.Label(
                self.controls_panel,
                text="Camera controls unavailable for this device.",
            ).grid(row=0, column=0, pady=10, sticky="w")

        self._adjust_window_size()

    def _add_section_header(self, row: int, text: str) -> int:
        header = ttk.Label(
            self.controls_panel,
            text=text,
            font=("TkDefaultFont", 9, "bold"),
        )
        header.grid(row=row, column=0, sticky="w", pady=(8, 2))
        return row + 1

    def _add_slider_controls(
        self, row: int, controls: dict[str, dict[str, Any]]
    ) -> tuple[int, bool]:
        created = False
        header_added = False

        for name, config in self.SLIDER_CONTROLS.items():
            meta = controls.get(name)
            if meta is None:
                continue
            meta = meta or {}

            scale_factor = float(config.get("scale_factor", 1.0)) or 1.0
            fallback = config.get("fallback", {})

            def to_slider(raw: Any) -> int | None:
                if isinstance(raw, (int, float)):
                    return int(round(raw / scale_factor))
                return None

            slider_min = int(fallback.get("min", 0))
            slider_max = int(fallback.get("max", slider_min + 1))
            slider_step = int(fallback.get("step", 1)) or 1
            default_slider = int(fallback.get("default", slider_min))

            meta_min = to_slider(meta.get("min"))
            if meta_min is not None:
                slider_min = meta_min
            meta_max = to_slider(meta.get("max"))
            if meta_max is not None:
                slider_max = meta_max
            meta_step = meta.get("step")
            if isinstance(meta_step, (int, float)) and meta_step > 0:
                slider_step = max(1, int(round(meta_step / scale_factor)))
            meta_default = to_slider(meta.get("default"))
            if meta_default is not None:
                default_slider = meta_default

            if slider_min > slider_max:
                slider_min, slider_max = slider_max, slider_min

            if slider_min == slider_max:
                slider_max = slider_min + slider_step

            resolution = max(1, slider_step)
            default_slider = min(max(default_slider, slider_min), slider_max)

            if not header_added:
                row = self._add_section_header(row, "Exposure & Image Quality")
                header_added = True

            frame = ttk.Frame(self.controls_panel, padding=(0, 2))
            frame.grid(row=row, column=0, sticky="ew")
            frame.columnconfigure(1, weight=1)

            ttk.Label(frame, text=name, width=16).grid(row=0, column=0, sticky="w")

            var = tk.IntVar(value=default_slider)
            self.control_vars[name] = var

            scale = tk.Scale(
                frame,
                from_=slider_min,
                to=slider_max,
                orient=tk.HORIZONTAL,
                resolution=resolution,
                showvalue=False,
                command=lambda value, control=name: self.on_slider_change(
                    control, value
                ),
            )
            scale.set(default_slider)
            scale.grid(row=0, column=1, sticky="ew", padx=(4, 4))

            entry = ttk.Entry(frame, width=6, textvariable=var, justify="center")
            entry.grid(row=0, column=2, padx=(0, 4))
            entry.bind(
                "<Return>",
                lambda event, control=name: self.on_entry_commit(control),
            )
            entry.bind(
                "<FocusOut>",
                lambda event, control=name: self.on_entry_commit(control),
            )

            value_label = ttk.Label(
                frame, text=self._format_slider_value(name, default_slider), width=12
            )
            value_label.grid(row=0, column=3, sticky="w")

            range_label = ttk.Label(
                frame,
                text=self._format_slider_range(name, slider_min, slider_max),
            )
            range_label.grid(row=0, column=4, sticky="w", padx=(4, 0))

            self.slider_widgets[name] = {
                "scale": scale,
                "entry": entry,
                "value_label": value_label,
                "range_label": range_label,
            }
            self.slider_defaults[name] = default_slider

            row += 1
            created = True

        return row, created

    def _add_toggle_controls(
        self, row: int, controls: dict[str, dict[str, Any]]
    ) -> tuple[int, bool]:
        created = False
        header_added = False

        for name, config in self.TOGGLE_CONTROLS.items():
            meta = controls.get(name)
            if meta is None:
                continue
            meta = meta or {}

            default = meta.get("default")
            if default is None:
                default = config.get("fallback", False)
            default_bool = bool(default)

            if not header_added:
                row = self._add_section_header(row, "Automatic Controls")
                header_added = True

            frame = ttk.Frame(self.controls_panel, padding=(0, 2))
            frame.grid(row=row, column=0, sticky="w")

            var = tk.BooleanVar(value=default_bool)
            check = ttk.Checkbutton(
                frame,
                text=name,
                variable=var,
                command=lambda control=name: self.on_toggle_change(control),
            )
            check.grid(row=0, column=0, sticky="w")

            self.toggle_vars[name] = var
            self.toggle_defaults[name] = default_bool
            row += 1
            created = True

        return row, created

    def _add_option_controls(
        self, row: int, controls: dict[str, dict[str, Any]]
    ) -> tuple[int, bool]:
        created = False
        header_added = False

        for name, config in self.OPTION_CONTROLS.items():
            meta = controls.get(name)
            if meta is None:
                continue
            meta = meta or {}

            raw_values = meta.get("values")
            if isinstance(raw_values, (list, tuple, set)) and raw_values:
                options = list(raw_values)
            else:
                options = list(config.get("fallback_values", []))

            if not options:
                continue

            labels_cfg = config.get("labels", {})
            label_map: dict[str, Any] = {}
            for value in options:
                label = labels_cfg.get(value)
                if label is None:
                    label = labels_cfg.get(str(value), None)
                if label is None:
                    label = str(value)
                original_label = label
                counter = 1
                while label in label_map:
                    label = f"{original_label} ({counter})"
                    counter += 1
                label_map[label] = value

            default_value = meta.get("default")
            if default_value is None:
                default_value = config.get("fallback_default")
            if default_value not in label_map.values():
                default_value = next(iter(label_map.values()))

            default_label = next(
                (label for label, value in label_map.items() if value == default_value),
                None,
            )
            if default_label is None:
                default_label = next(iter(label_map.keys()))

            if not header_added:
                row = self._add_section_header(row, "Mode Selection")
                header_added = True

            frame = ttk.Frame(self.controls_panel, padding=(0, 2))
            frame.grid(row=row, column=0, sticky="w")

            ttk.Label(frame, text=f"{name}:", width=16).grid(row=0, column=0, sticky="w")

            var = tk.StringVar(value=default_label)
            labels = list(label_map.keys())
            option = ttk.OptionMenu(
                frame,
                var,
                default_label,
                *labels,
                command=lambda selection, control=name: self.on_option_change(
                    control, selection
                ),
            )
            option.grid(row=0, column=1, sticky="w", padx=(4, 0))

            self.option_vars[name] = var
            self.option_value_maps[name] = label_map
            self.option_defaults[name] = default_label

            # Ensure current selection is applied once.
            self.on_option_change(name, default_label)

            row += 1
            created = True

        return row, created

    def _add_vector_controls(
        self, row: int, controls: dict[str, dict[str, Any]]
    ) -> tuple[int, bool]:
        created = False
        header_added = False

        for name, config in self.VECTOR_CONTROLS.items():
            meta = controls.get(name)
            if meta is None:
                continue
            meta = meta or {}

            length = int(config.get("length", 0))
            fallback_values = list(config.get("fallback", []))

            default_values = meta.get("default")
            if isinstance(default_values, (list, tuple)) and len(default_values) == length:
                current = list(default_values)
            elif isinstance(default_values, (list, tuple)) and len(default_values) != length:
                current = fallback_values[:length]
            else:
                current = fallback_values[:length]

            if len(current) < length:
                current.extend([0] * (length - len(current)))

            if not header_added:
                row = self._add_section_header(row, "Advanced Parameters")
                header_added = True

            frame = ttk.Frame(self.controls_panel, padding=(0, 2))
            frame.grid(row=row, column=0, sticky="ew")
            frame.columnconfigure(1, weight=1)

            ttk.Label(frame, text=name, width=16).grid(row=0, column=0, sticky="w")

            summary = ttk.Label(
                frame, text=self._format_vector_summary(name, current), width=30
            )
            summary.grid(row=0, column=1, sticky="w", padx=(4, 0))

            ttk.Button(
                frame,
                text="Set…",
                command=lambda control=name: self.on_vector_configure(control),
            ).grid(row=0, column=2, padx=(4, 0))

            self.vector_labels[name] = summary
            self.vector_values[name] = current
            self.vector_defaults[name] = tuple(current)

            row += 1
            created = True

        return row, created

    def _adjust_window_size(self) -> None:
        try:
            self.root.update_idletasks()
        except Exception:
            return

        required_width = self.root.winfo_reqwidth()
        required_height = self.root.winfo_reqheight()
        if required_width <= 0 or required_height <= 0:
            return

        target_width = required_width
        target_height = required_height

        if self._requested_size:
            target_width = max(target_width, self._requested_size[0])
            target_height = max(target_height, self._requested_size[1])
            if not self._geometry_applied:
                self._geometry_applied = True

        current_width = self.root.winfo_width()
        current_height = self.root.winfo_height()

        width = max(current_width, target_width)
        height = max(current_height, target_height)

        self.root.minsize(target_width, target_height)

        # Preserve existing position when possible.
        try:
            x = self.root.winfo_x()
            y = self.root.winfo_y()
            if x < 0 or y < 0:
                self.root.geometry(f"{width}x{height}")
            else:
                self.root.geometry(f"{width}x{height}+{x}+{y}")
        except Exception:
            self.root.geometry(f"{width}x{height}")

        self._report_window_size()

    def _report_window_size(self) -> None:
        if not self.debug:
            return
        try:
            actual_width = self.root.winfo_width()
            actual_height = self.root.winfo_height()
        except Exception:
            return
        if actual_width <= 0 or actual_height <= 0:
            return
        current_size = (actual_width, actual_height)
        if current_size != self._last_reported_size:
            print(f"[DEBUG] GUI size = {actual_width}x{actual_height}")
            self._last_reported_size = current_size

    def _on_root_configure(self, event: tk.Event) -> None:
        if event.widget is self.root:
            self._report_window_size()

    def _slider_to_actual(self, name: str, slider_value: int) -> Any:
        config = self.SLIDER_CONTROLS.get(name, {})
        scale_factor = float(config.get("scale_factor", 1.0)) or 1.0
        value_type = config.get("value_type", "float")
        actual = slider_value * scale_factor
        if value_type == "int":
            return int(round(actual))
        if value_type == "float":
            return float(actual)
        return actual

    def _format_slider_value(self, name: str, slider_value: int) -> str:
        config = self.SLIDER_CONTROLS.get(name, {})
        fmt = config.get("display_format")
        actual = self._slider_to_actual(name, slider_value)
        if fmt:
            try:
                return fmt.format(value=actual)
            except Exception:
                pass
        return str(actual)

    def _format_slider_range(
        self, name: str, slider_min: int, slider_max: int
    ) -> str:
        min_text = self._format_slider_value(name, slider_min)
        max_text = self._format_slider_value(name, slider_max)
        return f"Range: {min_text} → {max_text}"

    def _update_slider_value_label(self, name: str, slider_value: int) -> None:
        widgets = self.slider_widgets.get(name, {})
        label = widgets.get("value_label")
        if label and hasattr(label, "config"):
            label.config(text=self._format_slider_value(name, slider_value))

    def _format_vector_summary(self, name: str, values: list[Any]) -> str:
        config = self.VECTOR_CONTROLS.get(name, {})
        fmt = config.get("display_format")
        if fmt:
            try:
                return fmt.format(values=values)
            except Exception:
                pass
        return ", ".join(str(v) for v in values)

    def _update_vector_label(self, name: str, values: list[Any]) -> None:
        label = self.vector_labels.get(name)
        if label and hasattr(label, "config"):
            label.config(text=self._format_vector_summary(name, values))

    def _start_recording_indicator(self) -> None:
        if not hasattr(self, "record_indicator_canvas"):
            return
        self._stop_recording_indicator()
        self._recording_started_at = time.time()
        self.record_timer_var.set("00:00")
        self.record_indicator_canvas.itemconfig(
            self.record_indicator_dot, fill=self._record_indicator_on_color
        )
        self._record_indicator_visible = False
        self._record_indicator_tick()

    def _record_indicator_tick(self) -> None:
        if not getattr(self.controller, "is_recording", False):
            self._stop_recording_indicator()
            return

        fill = (
            self._record_indicator_on_color
            if self._record_indicator_visible
            else self._record_indicator_off_color
        )
        self.record_indicator_canvas.itemconfig(self.record_indicator_dot, fill=fill)
        self._record_indicator_visible = not self._record_indicator_visible

        if self._recording_started_at is not None:
            elapsed = max(0.0, time.time() - self._recording_started_at)
            minutes, seconds = divmod(int(elapsed), 60)
            self.record_timer_var.set(f"{minutes:02d}:{seconds:02d}")

        self._record_indicator_job = self.root.after(500, self._record_indicator_tick)

    def _stop_recording_indicator(self) -> None:
        if self._record_indicator_job:
            try:
                self.root.after_cancel(self._record_indicator_job)
            except Exception:
                pass
            self._record_indicator_job = None
        self._record_indicator_visible = False
        if hasattr(self, "record_indicator_canvas"):
            self.record_indicator_canvas.itemconfig(
                self.record_indicator_dot, fill=self._record_indicator_off_color
            )
        self.record_timer_var.set("")
        self._recording_started_at = None

    # Event Handlers ------------------------------------------------------
    def on_start_preview(self) -> None:
        try:
            self.controller.start()
        except Exception as exc:  # pragma: no cover - hardware dependent
            messagebox.showerror("Preview Error", f"Failed to start preview:\n{exc}")
        else:
            self._refresh_status_text()

    def on_stop_preview(self) -> None:
        try:
            self.controller.stop()
        except Exception as exc:  # pragma: no cover - hardware dependent
            messagebox.showerror("Preview Error", f"Failed to stop preview:\n{exc}")
        else:
            self._refresh_status_text()

    def on_capture(self) -> None:
        ensure_temp_dir()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_path = TEMP_DIR / f"temp_image_{timestamp}.jpg"
        try:
            self.controller.capture_image(temp_path)
        except Exception as exc:  # pragma: no cover - hardware dependent
            messagebox.showerror("Capture Error", f"Failed to capture still:\n{exc}")
            return

        default_stem = temp_path.stem.replace("temp_image_", "image_")
        raw = simpledialog.askstring(
            "Save Still Image",
            "Enter filename (stem only):",
            initialvalue=default_stem,
            parent=self.root,
        )
        if raw is None:
            try:
                temp_path.unlink()
            except Exception:
                pass
            return
        stem = sanitize_name(raw or default_stem)
        final_path = build_output_path(stem, "jpg")
        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.replace(final_path)
        except Exception as exc:  # pragma: no cover - hardware dependent
            messagebox.showerror("Save Error", f"Failed to save still:\n{exc}")
            return

        messagebox.showinfo("Capture Saved", f"Image saved to:\n{final_path.resolve()}")

    def on_start_recording(self) -> None:
        if self.controller.is_recording:
            messagebox.showinfo("Recording", "Recording already in progress.")
            return

        ensure_temp_dir()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_path = TEMP_DIR / f"temp_video_{timestamp}.mp4"
        try:
            self.controller.start_recording(temp_path)
        except Exception as exc:  # pragma: no cover - hardware dependent
            messagebox.showerror("Recording Error", f"Failed to start recording:\n{exc}")
            return

        self._start_recording_indicator()
        self._refresh_status_text()

    def on_stop_recording(self) -> None:
        if not self.controller.is_recording:
            messagebox.showinfo("Recording", "No active recording.")
            return

        try:
            temp_path = self.controller.stop_recording()
        except Exception as exc:  # pragma: no cover - hardware dependent
            self._stop_recording_indicator()
            messagebox.showerror("Recording Error", f"Failed to stop recording:\n{exc}")
            return
        self._stop_recording_indicator()

        default_stem = temp_path.stem.replace("temp_video_", "video_")
        raw = simpledialog.askstring(
            "Save Recording",
            "Enter filename (stem only):",
            initialvalue=default_stem,
            parent=self.root,
        )
        if raw is None:
            try:
                temp_path.unlink()
            except Exception:
                pass
            return
        stem = sanitize_name(raw or default_stem)
        final_path = build_output_path(stem, "mp4")
        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.replace(final_path)
        except Exception as exc:  # pragma: no cover - hardware dependent
            messagebox.showerror("Save Error", f"Failed to save recording:\n{exc}")
            return

        self._stop_recording_indicator()
        self._refresh_status_text()

    def on_toggle_preview(self) -> None:
        if getattr(self.controller, "_camera_started", False):
            self.on_stop_preview()
        else:
            self.on_start_preview()
        self._update_button_states()

    def on_toggle_recording(self) -> None:
        if self.controller.is_recording:
            self.on_stop_recording()
        else:
            self.on_start_recording()
        self._update_button_states()

    def on_slider_change(self, name: str, value: str) -> None:
        widgets = self.slider_widgets.get(name)
        var = self.control_vars.get(name)
        if widgets is None or var is None:
            return

        try:
            slider_value = int(round(float(value)))
        except ValueError:
            return

        scale_widget = widgets.get("scale")
        if isinstance(scale_widget, tk.Scale):
            min_val = int(scale_widget.cget("from"))
            max_val = int(scale_widget.cget("to"))
            slider_value = max(min(slider_value, max_val), min_val)

        if var.get() != slider_value:
            var.set(slider_value)

        self._update_slider_value_label(name, slider_value)
        actual_value = self._slider_to_actual(name, slider_value)
        self._apply_control(name, actual_value)

    def on_entry_commit(self, name: str) -> None:
        var = self.control_vars.get(name)
        widgets = self.slider_widgets.get(name)
        if var is None or widgets is None:
            return

        try:
            slider_value = int(var.get())
        except Exception:
            messagebox.showwarning("Control Input", "Please enter an integer value.")
            return

        scale = widgets.get("scale")
        if isinstance(scale, tk.Scale):
            min_val = int(scale.cget("from"))
            max_val = int(scale.cget("to"))
            slider_value = max(min(slider_value, max_val), min_val)
            if var.get() != slider_value:
                var.set(slider_value)
            scale.set(slider_value)

        self._update_slider_value_label(name, slider_value)
        actual_value = self._slider_to_actual(name, slider_value)
        self._apply_control(name, actual_value)

    def on_toggle_change(self, name: str) -> None:
        var = self.toggle_vars.get(name)
        if var is None:
            return
        self._apply_control(name, bool(var.get()))

    def on_option_change(self, name: str, selection: str) -> None:
        mapping = self.option_value_maps.get(name, {})
        value = mapping.get(selection, selection)
        self._apply_control(name, value)

    def on_vector_configure(self, name: str) -> None:
        config = self.VECTOR_CONTROLS.get(name)
        if not config:
            return

        current = list(self.vector_values.get(name, config.get("fallback", [])))
        labels = config.get("labels", [])
        length = int(config.get("length", len(labels)))
        value_type = config.get("value_type", "int")

        if len(current) < length:
            current.extend([0] * (length - len(current)))
        if len(labels) < length:
            labels = list(labels) + [f"Value {i+1}" for i in range(len(labels), length)]

        new_values: list[Any] = []
        for index in range(length):
            label = labels[index]
            initial = current[index]
            if value_type == "float":
                value = simpledialog.askfloat(
                    title=name,
                    prompt=f"{label}:",
                    initialvalue=float(initial),
                    parent=self.root,
                )
            else:
                value = simpledialog.askinteger(
                    title=name,
                    prompt=f"{label}:",
                    initialvalue=int(initial),
                    parent=self.root,
                )
            if value is None:
                return
            new_values.append(value)

        if name == "FrameDurationLimits" and new_values[0] > new_values[1]:
            messagebox.showwarning(
                "Invalid Limits", "Minimum frame duration must not exceed maximum."
            )
            return

        if value_type == "int":
            new_values = [int(round(v)) for v in new_values]
        else:
            new_values = [float(v) for v in new_values]

        self.vector_values[name] = new_values
        self._update_vector_label(name, new_values)
        self._apply_control(name, tuple(new_values))

    def reset_controls(self) -> None:
        for name, slider_default in self.slider_defaults.items():
            var = self.control_vars.get(name)
            widgets = self.slider_widgets.get(name)
            if var is None or widgets is None:
                continue
            current_slider = var.get()
            if current_slider == slider_default:
                continue
            scale = widgets.get("scale")
            if isinstance(scale, tk.Scale):
                scale.set(slider_default)

        for name, default_bool in self.toggle_defaults.items():
            var = self.toggle_vars.get(name)
            if var is None:
                continue
            if var.get() == default_bool:
                continue
            var.set(default_bool)
            self._apply_control(name, default_bool)

        for name, default_label in self.option_defaults.items():
            var = self.option_vars.get(name)
            if var is None:
                continue
            if var.get() == default_label:
                continue
            var.set(default_label)
            self.on_option_change(name, default_label)

        for name, default_values in self.vector_defaults.items():
            current_values = self.vector_values.get(name)
            if current_values == list(default_values):
                continue
            values = list(default_values)
            self.vector_values[name] = values
            self._update_vector_label(name, values)
            self._apply_control(name, tuple(values))

    def _apply_control(self, name: str, value: Any) -> None:
        if isinstance(value, str):
            stripped = value.strip()
            lower = stripped.lower()
            if lower in {"true", "false"}:
                value = lower == "true"
            else:
                try:
                    value = int(stripped, 0)
                except ValueError:
                    try:
                        value = float(stripped)
                    except ValueError:
                        pass
        elif isinstance(value, (list, tuple)):
            converted = []
            changed = False
            for item in value:
                if isinstance(item, str):
                    stripped = item.strip()
                    lower = stripped.lower()
                    if lower in {"true", "false"}:
                        converted.append(lower == "true")
                        changed = True
                        continue
                    try:
                        converted.append(int(stripped, 0))
                        changed = True
                        continue
                    except ValueError:
                        try:
                            converted.append(float(stripped))
                            changed = True
                            continue
                        except ValueError:
                            pass
                converted.append(item)
            if changed:
                value = type(value)(converted)

        try:
            self.controller.set_control(name, value)
        except Exception as exc:  # pragma: no cover - hardware dependent
            messagebox.showerror(
                "Control Error", f"Failed to set {name} to {value}:\n{exc}"
            )

    # Status updates ------------------------------------------------------
    def _refresh_status_text(self) -> None:
        status = self.controller.status()
        preview = status.get("preview_backend") or "stopped"
        recording = "Recording" if status.get("recording") else "Idle"
        self.status_var.set(f"Preview: {preview} | {recording}")
        self._update_button_states()

    def _update_button_states(self) -> None:
        if getattr(self.controller, "_camera_started", False):
            self.preview_btn.config(text="Stop Preview")
        else:
            self.preview_btn.config(text="Start Preview")

        if self.controller.is_recording:
            self.record_btn.config(text="Stop Recording")
        else:
            self.record_btn.config(text="Start Recording")

    def update_status(self) -> None:
        self._refresh_status_text()
        self._status_job = self.root.after(1000, self.update_status)

    def update_fps(self) -> None:
        preview_fps = self.controller.get_fps()
        self.preview_fps_var.set(f"FPS: {preview_fps:.1f}")

        self._fps_job = self.root.after(500, self.update_fps)

    # Shutdown ------------------------------------------------------------
    def on_close(self) -> None:
        if self._status_job:
            try:
                self.root.after_cancel(self._status_job)
            except Exception:
                pass
            self._status_job = None
        if self._fps_job:
            try:
                self.root.after_cancel(self._fps_job)
            except Exception:
                pass
            self._fps_job = None

        try:
            self.controller.stop()
        except Exception:
            pass
        if self.root.winfo_exists():
            self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Picamera2 GUI controller")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose GUI diagnostics (window size logs).",
    )
    parser.add_argument(
        "--window",
        type=str,
        help="Initial window size, e.g. 1100x800.",
    )
    args = parser.parse_args()

    try:
        controller = Picamera2Controller()
    except ImportError as exc:
        print(exc)
        sys.exit(1)

    root = tk.Tk()
    initial_size: tuple[int, int] | None = None
    if args.window:
        try:
            width_str, height_str = args.window.lower().split("x", 1)
            initial_size = (int(width_str), int(height_str))
        except Exception:
            print(f"[WARN] Invalid --window format: {args.window!r}. Expected WIDTHxHEIGHT.", file=sys.stderr)
            initial_size = None
    if initial_size is None:
        initial_size = (690, 570)
    root.geometry(f"{initial_size[0]}x{initial_size[1]}")
    app = RaspiCaptureGUI(root, controller, debug=args.debug, initial_size=initial_size)
    try:
        root.mainloop()
    finally:
        try:
            controller.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()

