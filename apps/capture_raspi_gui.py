#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi Picamera2 GUI
--------------------------

Simple Tkinter interface around Picamera2Controller that mirrors the CLI
features while exposing a handful of integer camera controls via sliders.

Requires a Raspberry Pi with Picamera2 installed and access to a display (or
VC4-enabled virtual display). Designed to be lightweight for field usage.
"""

from __future__ import annotations

import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

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

    NUMERIC_CONTROL_WHITELIST = [
        "AnalogueGain",
        "Brightness",
        "Contrast",
        "ExposureTime",
        "Saturation",
        "Sharpness",
    ]

    def __init__(self, root: tk.Tk, controller: Picamera2Controller) -> None:
        self.root = root
        self.controller = controller
        self.root.title("3D-Cam | Raspberry Pi Capture")

        self.status_var = tk.StringVar(value="Initializing camera…")
        self.fps_var = tk.StringVar(value="FPS: —")

        self.control_vars: dict[str, tk.IntVar] = {}
        self.slider_widgets: dict[str, dict[str, tk.Widget]] = {}

        self.recording = False
        self._status_job: str | None = None
        self._fps_job: str | None = None

        self._build_ui()
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
        ttk.Label(status_row, textvariable=self.status_var).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(status_row, textvariable=self.fps_var).pack(side=tk.LEFT)

        # Buttons
        button_row = ttk.Frame(main)
        button_row.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(button_row, text="Start Preview", command=self.on_start_preview).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(button_row, text="Stop Preview", command=self.on_stop_preview).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(button_row, text="Capture Still", command=self.on_capture).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(
            button_row,
            text="Start Recording",
            command=self.on_start_recording,
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            button_row,
            text="Stop Recording",
            command=self.on_stop_recording,
        ).pack(side=tk.LEFT)

        # Control sliders container
        controls_frame = ttk.LabelFrame(main, text="Camera Controls", padding="6")
        controls_frame.pack(fill=tk.BOTH, expand=True)
        self.controls_inner = ttk.Frame(controls_frame)
        self.controls_inner.pack(fill=tk.BOTH, expand=True)

        # Scrollable frame to handle many controls gracefully.
        self._add_scroll_region(self.controls_inner)

    def _add_scroll_region(self, container: ttk.Frame) -> None:
        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        frame = ttk.Frame(canvas)
        frame.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        window = canvas.create_window((0, 0), window=frame, anchor="nw")

        def _resize_canvas(event: tk.Event) -> None:
            canvas.itemconfig(window, width=event.width)

        canvas.bind("<Configure>", _resize_canvas)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.scroll_frame = frame

    # Control population --------------------------------------------------
    def _populate_controls(self) -> None:
        controls = self.controller.list_controls()
        for child in self.scroll_frame.winfo_children():
            child.destroy()

        row = 0
        for name in self.NUMERIC_CONTROL_WHITELIST:
            meta = controls.get(name)
            min_val = meta.get("min") if meta else None
            max_val = meta.get("max") if meta else None

            if not isinstance(min_val, (int, float)) or not isinstance(
                max_val, (int, float)
            ):
                continue

            # Only use integer-valued controls for consistency with the UI.
            min_int = int(round(min_val))
            max_int = int(round(max_val))
            if min_int == max_int:
                continue

            default = meta.get("default")
            try:
                default_int = int(round(default)) if default is not None else min_int
            except Exception:
                default_int = min_int

            step = meta.get("step")
            resolution = max(1, int(round(step))) if isinstance(step, (int, float)) else 1

            var = tk.IntVar(value=default_int)
            self.control_vars[name] = var

            frame = ttk.Frame(self.scroll_frame, padding=(0, 2))
            frame.grid(row=row, column=0, sticky="ew")
            frame.columnconfigure(1, weight=1)

            ttk.Label(frame, text=name, width=16).grid(row=0, column=0, sticky="w")

            scale = tk.Scale(
                frame,
                from_=min_int,
                to=max_int,
                orient=tk.HORIZONTAL,
                resolution=resolution,
                showvalue=False,
                command=lambda value, control=name: self.on_slider_change(
                    control, value
                ),
            )
            scale.set(default_int)
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

            ttk.Label(frame, text=f"[{min_int}, {max_int}]").grid(
                row=0, column=3, sticky="w"
            )

            self.slider_widgets[name] = {"scale": scale, "entry": entry}
            row += 1

        if row == 0:
            ttk.Label(
                self.scroll_frame,
                text="No integer-based controls detected. Update whitelist as needed.",
            ).grid(row=0, column=0, pady=10)

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

        messagebox.showinfo("Recording", "Recording started.")
        self._refresh_status_text()

    def on_stop_recording(self) -> None:
        if not self.controller.is_recording:
            messagebox.showinfo("Recording", "No active recording.")
            return

        try:
            temp_path = self.controller.stop_recording()
        except Exception as exc:  # pragma: no cover - hardware dependent
            messagebox.showerror("Recording Error", f"Failed to stop recording:\n{exc}")
            return

        default_stem = temp_path.stem.replace("temp_video_", "video_")
        raw = simpledialog.askstring(
            "Save Recording",
            "Enter filename (stem only):",
            initialvalue=default_stem,
            parent=self.root,
        )
        stem = sanitize_name(raw or default_stem)
        final_path = build_output_path(stem, "mp4")

        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.replace(final_path)
        except Exception as exc:  # pragma: no cover - hardware dependent
            messagebox.showerror("Save Error", f"Failed to save recording:\n{exc}")
            return

        messagebox.showinfo("Recording Saved", f"Video saved to:\n{final_path.resolve()}")
        self._refresh_status_text()

    def on_slider_change(self, name: str, value: str) -> None:
        try:
            int_val = int(round(float(value)))
        except ValueError:
            return

        var = self.control_vars.get(name)
        if var is not None and var.get() != int_val:
            var.set(int_val)
        entry = self.slider_widgets.get(name, {}).get("entry")
        if isinstance(entry, ttk.Entry):
            entry.icursor(tk.END)

        self._apply_control(name, int_val)

    def on_entry_commit(self, name: str) -> None:
        var = self.control_vars.get(name)
        widgets = self.slider_widgets.get(name)
        if var is None or widgets is None:
            return

        try:
            value = int(var.get())
        except Exception:
            messagebox.showwarning("Control Input", "Please enter an integer value.")
            return

        scale = widgets.get("scale")
        if isinstance(scale, tk.Scale):
            min_val = int(scale.cget("from"))
            max_val = int(scale.cget("to"))
            clamped = max(min(value, max_val), min_val)
            if clamped != value:
                var.set(clamped)
                value = clamped
            scale.set(value)

        self._apply_control(name, value)

    def _apply_control(self, name: str, value: int) -> None:
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

    def update_status(self) -> None:
        self._refresh_status_text()
        self._status_job = self.root.after(1000, self.update_status)

    def update_fps(self) -> None:
        fps = self.controller.get_fps()
        self.fps_var.set(f"FPS: {fps:.1f}")
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
    try:
        controller = Picamera2Controller()
    except ImportError as exc:
        print(exc)
        sys.exit(1)

    root = tk.Tk()
    app = RaspiCaptureGUI(root, controller)
    try:
        root.mainloop()
    finally:
        try:
            controller.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()

