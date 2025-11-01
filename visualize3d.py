#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Pair Visualization Tool
--------------------------
Interactive 3D visualization of X, Y, Z pair trajectories over time.

Features:
  • Load CSV files from pair_detect.py exports
  • Interactive 3D view with mouse drag to pan/rotate/zoom
  • Time slider to scrub through frames
  • Show trajectories with color-coded tracks
  • Toggle trails/history display
  • Select specific track IDs to visualize
"""

import tkinter as tk
from tkinter import ttk, messagebox

from lib.visualizing import Base3DVisualizer


class Pair3DVisualizer(Base3DVisualizer):
    """3D pair trajectory visualizer extending Base3DVisualizer."""
    
    def __init__(self, root):
        # Initialize base class
        super().__init__(root, "3D Pair Trajectory Visualizer", "1200x800")
    
    def get_plot_title(self):
        """Override to customize title."""
        return f'3D Pair Trajectories - Frame {self.current_frame}/{self.max_frame}'


def main():
    root = tk.Tk()
    app = Pair3DVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
