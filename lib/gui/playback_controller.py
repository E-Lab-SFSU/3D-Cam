"""
Playback Controller Widget
---------------------------
Reusable playback control widget for video/trajectory playback.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable


class PlaybackController:
    """Internal playback control widget."""
    
    def __init__(self, parent_frame, on_frame_changed: Callable[[int], None],
                 max_frame: int = 0, initial_frame: int = 0, initial_speed: float = 1.0):
        self.on_frame_changed_callback = on_frame_changed
        self.max_frame = max_frame
        self.current_frame = initial_frame
        self.playing = False
        self.root = None
        
        widget = parent_frame
        while widget:
            if isinstance(widget, tk.Tk):
                self.root = widget
                break
            widget = widget.master
        
        if self.root is None:
            raise ValueError("Could not find root Tk window")
        
        self.control_frame = ttk.LabelFrame(parent_frame, text="Time Control", padding="10")
        self.control_frame.pack(fill="x", pady=5)
        
        self.frame_label = ttk.Label(self.control_frame, text=f"Frame: {initial_frame} / {max_frame}")
        self.frame_label.pack()
        
        self.frame_var = tk.IntVar(value=initial_frame)
        self.frame_slider = ttk.Scale(
            self.control_frame,
            from_=0,
            to=max_frame,
            orient="horizontal",
            variable=self.frame_var,
            command=self._on_slider_changed
        )
        self.frame_slider.pack(fill="x", pady=5)
        
        playback_frame = ttk.Frame(self.control_frame)
        playback_frame.pack(fill="x")
        
        self.play_button = ttk.Button(playback_frame, text="▶ Play", command=self.toggle_play)
        self.play_button.pack(side="left", padx=2)
        
        ttk.Button(playback_frame, text="⏮", command=lambda: self.set_frame(0)).pack(side="left", padx=2)
        ttk.Button(playback_frame, text="⏭", command=lambda: self.set_frame(self.max_frame)).pack(side="left", padx=2)
        
        speed_frame = ttk.Frame(self.control_frame)
        speed_frame.pack(fill="x", pady=5)
        ttk.Label(speed_frame, text="Speed:").pack(side="left")
        
        self.speed_var = tk.DoubleVar(value=initial_speed)
        speed_scale = ttk.Scale(
            speed_frame,
            from_=0.1,
            to=5.0,
            orient="horizontal",
            variable=self.speed_var,
            length=150,
            command=self._on_speed_changed
        )
        speed_scale.pack(side="left", fill="x", expand=True)
        
        self.speed_label = ttk.Label(speed_frame, text=f"{initial_speed:.1f}x")
        self.speed_label.pack(side="left", padx=5)
    
    def _on_slider_changed(self, value=None):
        frame = int(float(self.frame_var.get()))
        if frame != self.current_frame:
            self.current_frame = frame
            self.frame_label.config(text=f"Frame: {frame} / {self.max_frame}")
            self.on_frame_changed_callback(frame)
    
    def _on_speed_changed(self, value=None):
        speed = float(self.speed_var.get())
        self.speed_label.config(text=f"{speed:.1f}x")
    
    def set_frame(self, frame: int):
        frame = max(0, min(frame, self.max_frame))
        self.current_frame = frame
        self.frame_var.set(frame)
        self.frame_label.config(text=f"Frame: {frame} / {self.max_frame}")
        self.on_frame_changed_callback(frame)
    
    def toggle_play(self):
        if self.max_frame == 0:
            return
        self.playing = not self.playing
        self.play_button.config(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self.play_loop()
    
    def play_loop(self):
        if not self.playing:
            return
        speed = self.speed_var.get()
        next_frame = self.current_frame + int(speed)
        if next_frame > self.max_frame:
            next_frame = 0
        self.set_frame(next_frame)
        delay = max(1, int(1000 / (30 * speed)))
        self.root.after(delay, self.play_loop)
    
    def set_max_frame(self, max_frame: int):
        self.max_frame = max_frame
        self.frame_slider.config(to=max_frame)
        self.frame_label.config(text=f"Frame: {self.current_frame} / {self.max_frame}")
        if self.current_frame > max_frame:
            self.set_frame(max_frame)
    
    def stop(self):
        self.playing = False
        self.play_button.config(text="▶ Play")
    
    def is_playing(self) -> bool:
        return self.playing
    
    def get_current_frame(self) -> int:
        return self.current_frame
    
    def get_speed(self) -> float:
        return self.speed_var.get()

