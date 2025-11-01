#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z Height Histogram Tool
------------------------
Visualization tool that shows Z height distribution as a histogram.

Features:
  • Load CSV files from pair_detect.py exports
  • Histogram showing Z height (mm) on Y-axis vs frequency on X-axis
  • Auto-load latest CSV from pair_detect_output
  • Statistics display (mean, median, std dev, min, max)
  • Adjustable bin count
"""

import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from typing import List, Optional
import os


class ZHistogramViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Z Height Histogram")
        self.root.geometry("1000x700")
        
        # Data storage
        self.csv_path = None
        self.z_values = []  # List of Z height values in mm
        
        # Histogram settings
        self.bin_count = 50
        
        # Setup UI
        self.setup_ui()
        
        # Load latest CSV if available
        self.auto_load_latest_csv()
    
    def setup_ui(self):
        """Create the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Left panel - controls
        left_panel = ttk.Frame(main_frame, width=250)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Load button
        ttk.Button(left_panel, text="Load CSV File", command=self.load_csv).pack(pady=5, fill="x")
        
        # File label
        self.file_label = ttk.Label(left_panel, text="No file loaded", wraplength=230)
        self.file_label.pack(pady=5)
        
        # Separator
        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", pady=10)
        
        # Histogram settings
        settings_frame = ttk.LabelFrame(left_panel, text="Histogram Settings", padding="10")
        settings_frame.pack(fill="x", pady=5)
        
        # Bin count control
        bin_frame = ttk.Frame(settings_frame)
        bin_frame.pack(fill="x", pady=5)
        ttk.Label(bin_frame, text="Bins:").pack(side="left")
        self.bin_var = tk.IntVar(value=50)
        bin_scale = ttk.Scale(
            bin_frame,
            from_=10,
            to=200,
            orient="horizontal",
            variable=self.bin_var,
            length=150,
            command=self.on_bin_count_changed
        )
        bin_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.bin_label = ttk.Label(bin_frame, text="50")
        self.bin_label.pack(side="left")
        
        # Statistics display
        stats_frame = ttk.LabelFrame(left_panel, text="Statistics", padding="10")
        stats_frame.pack(fill="x", pady=5)
        
        self.stats_label = ttk.Label(
            stats_frame,
            text="No data loaded",
            wraplength=230,
            justify="left"
        )
        self.stats_label.pack()
        
        # Info display
        info_frame = ttk.LabelFrame(left_panel, text="Info", padding="10")
        info_frame.pack(fill="x", pady=5)
        
        self.info_label = ttk.Label(
            info_frame,
            text="No data loaded",
            wraplength=230,
            justify="left"
        )
        self.info_label.pack()
        
        # Export button
        export_frame = ttk.LabelFrame(left_panel, text="Export", padding="10")
        export_frame.pack(fill="x", pady=5)
        ttk.Button(export_frame, text="Save Histogram Image", command=self.save_image).pack(pady=5, fill="x")
        
        # Right panel - histogram plot
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        toolbar.update()
        
        # Initial empty plot
        self.update_histogram()
    
    def on_bin_count_changed(self, value=None):
        """Handle bin count change."""
        self.bin_count = int(self.bin_var.get())
        self.bin_label.config(text=str(self.bin_count))
        self.update_histogram()
    
    def auto_load_latest_csv(self):
        """Automatically load the latest CSV file from pair_detect_output."""
        output_dir = Path("pair_detect_output")
        if not output_dir.exists():
            return
        
        # Find all CSV files
        csv_files = list(output_dir.rglob("pairs.csv"))
        if not csv_files:
            return
        
        # Sort by modification time (newest first)
        csv_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        latest_csv = csv_files[0]
        
        # Try to load it
        try:
            self.load_csv_file(str(latest_csv))
        except Exception as e:
            print(f"Failed to auto-load {latest_csv}: {e}")
    
    def load_csv(self):
        """Open file dialog to load CSV file."""
        file_path = filedialog.askopenfilename(
            title="Load Pair CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.load_csv_file(file_path)
    
    def load_csv_file(self, file_path: str):
        """Load and parse CSV file."""
        self.csv_path = file_path
        self.z_values = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Check if Z_mm column exists
                if 'Z_mm' not in reader.fieldnames:
                    messagebox.showwarning(
                        "Missing Z Data",
                        "CSV file doesn't contain Z_mm column.\n\n"
                        "Z height data is only available if calibration data was loaded during export.\n"
                        "Please re-export with calibration data to view Z height histogram."
                    )
                    return
                
                for row in reader:
                    try:
                        z_str = row.get('Z_mm', '').strip()
                        
                        # Skip rows with missing Z data
                        if not z_str:
                            continue
                        
                        z = float(z_str)
                        self.z_values.append(z)
                    
                    except (ValueError, KeyError) as e:
                        continue
            
            if not self.z_values:
                messagebox.showwarning(
                    "No Z Data",
                    "CSV file contains no valid Z height values.\n\n"
                    "Make sure calibration data was loaded during export."
                )
                return
            
            # Update UI
            self.file_label.config(text=f"Loaded: {Path(file_path).name}")
            
            # Update info
            self.info_label.config(
                text=f"Total Points: {len(self.z_values)}"
            )
            
            # Update statistics
            self.update_statistics()
            
            # Update histogram
            self.update_histogram()
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file:\n{e}")
    
    def update_statistics(self):
        """Calculate and display statistics."""
        if not self.z_values:
            self.stats_label.config(text="No data loaded")
            return
        
        z_array = np.array(self.z_values)
        
        mean_z = np.mean(z_array)
        median_z = np.median(z_array)
        std_z = np.std(z_array)
        min_z = np.min(z_array)
        max_z = np.max(z_array)
        
        stats_text = (
            f"Mean: {mean_z:.3f} mm\n"
            f"Median: {median_z:.3f} mm\n"
            f"Std Dev: {std_z:.3f} mm\n"
            f"Min: {min_z:.3f} mm\n"
            f"Max: {max_z:.3f} mm"
        )
        
        self.stats_label.config(text=stats_text)
    
    def update_histogram(self):
        """Update the histogram plot."""
        self.ax.clear()
        
        if not self.z_values:
            self.ax.text(0.5, 0.5, "No data loaded", transform=self.ax.transAxes, ha="center")
            self.canvas.draw()
            return
        
        # Create histogram
        # X-axis: Z height (mm), Y-axis: frequency (count)
        n, bins, patches = self.ax.hist(
            self.z_values,
            bins=self.bin_count,
            edgecolor='black',
            alpha=0.7,
            color='steelblue'
        )
        
        # Set labels
        self.ax.set_xlabel('Z Height (mm)', fontsize=12)
        self.ax.set_ylabel('Frequency (Count)', fontsize=12)
        self.ax.set_title(f'Z Height Distribution (n={len(self.z_values)})', fontsize=14, fontweight='bold')
        
        # Set Y-axis to logarithmic scale
        self.ax.set_yscale('log')
        
        # Add grid
        self.ax.grid(True, alpha=0.3, axis='y')
        self.ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Add vertical lines for mean and median
        z_array = np.array(self.z_values)
        mean_z = np.mean(z_array)
        median_z = np.median(z_array)
        
        max_freq = np.max(n)
        self.ax.axvline(mean_z, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_z:.3f} mm', alpha=0.7)
        self.ax.axvline(median_z, color='green', linestyle='--', linewidth=2, label=f'Median: {median_z:.3f} mm', alpha=0.7)
        
        # Add legend
        self.ax.legend(loc='upper right', fontsize=9)
        
        # Tight layout
        self.fig.tight_layout()
        
        self.canvas.draw()
    
    def save_image(self):
        """Save histogram as image file."""
        if not self.z_values:
            messagebox.showwarning("No Data", "Please load a CSV file first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Histogram Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Histogram saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{e}")


def main():
    root = tk.Tk()
    app = ZHistogramViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()

