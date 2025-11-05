"""
Dialog Widgets
---------------
Reusable dialog widgets for common user input tasks.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional
import cv2


class FPSDialog:
    """Custom dialog for FPS input with 'Match Input' button."""
    
    def __init__(self, parent, initial_fps: int = 30, csv_path: Optional[str] = None):
        """
        Initialize FPS dialog.
        
        Args:
            parent: Parent window
            initial_fps: Initial FPS value
            csv_path: Path to CSV file (used to find associated video file)
        """
        self.result = None
        self.csv_path = csv_path
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Video FPS")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(False, False)
        
        # Center the dialog
        self.dialog.update_idletasks()
        width = 350
        height = 150
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding="15")
        main_frame.pack(fill="both", expand=True)
        
        # Label
        ttk.Label(main_frame, text="Enter frames per second for the video:").pack(anchor="w", pady=(0, 10))
        
        # FPS input frame
        fps_frame = ttk.Frame(main_frame)
        fps_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(fps_frame, text="FPS:").pack(side="left", padx=(0, 5))
        
        self.fps_var = tk.StringVar(value=str(initial_fps))
        self.fps_entry = ttk.Entry(fps_frame, textvariable=self.fps_var, width=10)
        self.fps_entry.pack(side="left", padx=(0, 5))
        self.fps_entry.select_range(0, tk.END)
        self.fps_entry.focus_set()
        
        # Match Input button
        match_button = ttk.Button(fps_frame, text="Match Input", command=self.match_input_fps)
        match_button.pack(side="left", padx=(5, 0))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x")
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side="right", padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side="right")
        
        # Bind Enter key to OK
        self.dialog.bind('<Return>', lambda e: self.ok_clicked())
        self.dialog.bind('<Escape>', lambda e: self.cancel_clicked())
        
        # Wait for dialog to close
        self.dialog.wait_window()
    
    def match_input_fps(self):
        """Find the input video file and read its FPS."""
        if not self.csv_path:
            messagebox.showwarning("No CSV", "No CSV file loaded. Cannot find associated video.")
            return
        
        # Try to find the associated video file
        video_path = self._find_video_file(self.csv_path)
        
        if not video_path:
            messagebox.showwarning(
                "Video Not Found",
                "Could not find associated video file.\n\n"
                "Looking for video files in the same directory as the CSV file."
            )
            return
        
        # Read FPS from video
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                messagebox.showerror("Error", f"Could not open video file:\n{video_path}")
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if fps > 0:
                # Round to nearest integer and update the entry
                fps_int = int(round(fps))
                self.fps_var.set(str(fps_int))
                # Select all text in the entry for easy editing
                self.fps_entry.select_range(0, tk.END)
                self.fps_entry.focus_set()
            else:
                messagebox.showwarning("FPS Not Available", "Could not read FPS from video file.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error reading video FPS:\n{str(e)}")
    
    def _find_video_file(self, csv_path: str) -> Optional[str]:
        """
        Find the associated video file for a CSV file.
        
        Looks for video files in the same directory with matching base name.
        CSV files are typically named like: {video_name}-paired-tracked.csv
        Video files might be: {video_name}.mp4 or {video_name}.avi etc.
        """
        csv_path_obj = Path(csv_path)
        csv_dir = csv_path_obj.parent
        csv_name = csv_path_obj.stem  # filename without extension
        
        # Remove common CSV suffixes to get base video name
        # Examples: "video-paired-tracked" -> "video", "video-smoothed" -> "video"
        base_name = csv_name
        suffixes_to_remove = [
            '-paired-tracked',
            '-paired-tracked-smoothed',
            '-smoothed',
            '-tracked',
            '-paired'
        ]
        
        for suffix in suffixes_to_remove:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        # Look for video files with this base name
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        for ext in video_extensions:
            video_path = csv_dir / f"{base_name}{ext}"
            if video_path.exists():
                return str(video_path)
        
        # If exact match not found, look for any video file in the directory
        # that starts with the base name
        for ext in video_extensions:
            for video_file in csv_dir.glob(f"{base_name}*{ext}"):
                if video_file.exists():
                    return str(video_file)
        
        return None
    
    def ok_clicked(self):
        """Handle OK button click."""
        try:
            fps_value = int(self.fps_var.get())
            if fps_value < 1 or fps_value > 120:
                messagebox.showerror("Invalid FPS", "FPS must be between 1 and 120.")
                return
            self.result = fps_value
            self.dialog.destroy()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer for FPS.")
    
    def cancel_clicked(self):
        """Handle Cancel button click."""
        self.result = None
        self.dialog.destroy()

