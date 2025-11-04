#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Calibration Tool - Raspberry Pi
Simple, efficient calibration tool for Raspberry Pi
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from datetime import datetime

from lib.gui import apply_standard_theme, format_window_title, get_standard_size

# Global state
image_original = None
points = []
scale_x = 1.0
scale_y = 1.0
window_name = "Calibration - Click Two Points"
display_update_scheduled = False

# Defaults
DEFAULT_FOCAL_LENGTH_MM = 16.0
DEFAULT_PIXEL_SIZE_MICRONS = 3.0
DEFAULT_SENSOR_X_MM = 5.76
DEFAULT_SENSOR_Y_MM = 3.24


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks."""
    global points, scale_x, scale_y
    
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        # Convert to original image coordinates
        orig_x = int(x / scale_x) if scale_x > 0 else x
        orig_y = int(y / scale_y) if scale_y > 0 else y
        
        # Clamp to image bounds
        if image_original is not None:
            h, w = image_original.shape[:2]
            orig_x = max(0, min(orig_x, w - 1))
            orig_y = max(0, min(orig_y, h - 1))
        
        points.append((orig_x, orig_y))
        print(f"Point {len(points)}: ({orig_x}, {orig_y})")
        schedule_update()


def schedule_update():
    """Schedule display update via tkinter to avoid blocking."""
    global display_update_scheduled
    if not display_update_scheduled:
        display_update_scheduled = True
        root.after(10, do_update_display)


def do_update_display():
    """Perform the actual display update."""
    global display_update_scheduled
    display_update_scheduled = False
    update_display()


def update_display():
    """Update the OpenCV window."""
    global image_original, points, scale_x, scale_y
    
    if image_original is None:
        return
    
    # Create display image
    img = image_original.copy()
    
    # Draw points
    if len(points) >= 1:
        cv2.circle(img, points[0], 8, (0, 255, 0), -1)
        cv2.putText(img, "1", (points[0][0] + 12, points[0][1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if len(points) >= 2:
        cv2.circle(img, points[1], 8, (0, 255, 0), -1)
        cv2.putText(img, "2", (points[1][0] + 12, points[1][1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.line(img, points[0], points[1], (0, 255, 0), 2)
        
        # Calculate distance
        dist = np.sqrt((points[1][0] - points[0][0])**2 + 
                      (points[1][1] - points[0][1])**2)
        mid_x = (points[0][0] + points[1][0]) // 2
        mid_y = (points[0][1] + points[1][1]) // 2
        cv2.putText(img, f"{dist:.1f} px", (mid_x, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Resize for display (max 80% of screen, but ensure minimum size for visibility)
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    
    # Ensure reasonable screen size (RPi sometimes reports incorrectly)
    if screen_w < 640:
        screen_w = 1920
    if screen_h < 480:
        screen_h = 1080
    
    max_w = max(int(screen_w * 0.8), 1280)
    max_h = max(int(screen_h * 0.8), 720)
    
    img_h, img_w = img.shape[:2]
    aspect = img_w / img_h
    
    if img_w > max_w or img_h > max_h:
        if aspect > 1:
            display_w = max_w
            display_h = int(max_w / aspect)
        else:
            display_h = max_h
            display_w = int(max_h * aspect)
    else:
        display_w = img_w
        display_h = img_h
    
    # Resize and store scale factors
    display_img = cv2.resize(img, (display_w, display_h))
    scale_x = display_w / img_w
    scale_y = display_h / img_h
    
    # Show image - minimal waitKey to avoid blocking
    cv2.imshow(window_name, display_img)
    cv2.waitKey(1)  # Single minimal wait


def load_image():
    """Load image file."""
    global image_original, points
    
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("All", "*.*")]
    )
    
    if not file_path:
        return
    
    image_original = cv2.imread(file_path)
    if image_original is None:
        messagebox.showerror("Error", f"Could not load image: {file_path}")
        return
    
    points = []
    
    # Create window if needed
    try:
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
    except:
        pass
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Initial display
    update_display()
    cv2.waitKey(1)
    
    # Schedule window focus setup (non-blocking)
    def setup_focus():
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
            cv2.waitKey(1)
        except:
            pass
    
    root.after(100, setup_focus)
    
    image_path_var.set(file_path)
    result_label.config(text="Click two points on the image")


def calculate():
    """Calculate calibration."""
    global points
    
    if len(points) != 2:
        messagebox.showwarning("Warning", "Please click two points first")
        return
    
    try:
        mm_val = float(mm_entry.get())
        if mm_val <= 0:
            raise ValueError("Measurement must be > 0")
        
        pixel_dist = np.sqrt((points[1][0] - points[0][0])**2 + 
                            (points[1][1] - points[0][1])**2)
        pixels_per_mm = pixel_dist / mm_val
        
        # Get camera params
        focal_mm = float(focal_entry.get())
        pixel_microns = float(pixel_entry.get())
        sensor_x_mm = float(sensor_x_entry.get())
        sensor_y_mm = float(sensor_y_entry.get())
        
        # Calculate working distance
        pixel_size_mm = pixel_microns / 1000.0
        image_size_sensor_mm = pixel_dist * pixel_size_mm
        working_dist_mm = (focal_mm * mm_val) / image_size_sensor_mm if image_size_sensor_mm > 0 else 0
        working_dist_m = working_dist_mm / 1000.0
        
        result_label.config(
            text=f"Calibration: {pixels_per_mm:.4f} px/mm\n"
                 f"Distance: {pixel_dist:.2f} px = {mm_val:.2f} mm\n"
                 f"Working Distance: {working_dist_mm:.2f} mm ({working_dist_m:.4f} m)"
        )
        
        # Store for saving
        global calibration_data
        calibration_data = {
            "pixels_per_mm": pixels_per_mm,
            "pixel_distance": pixel_dist,
            "mm_measurement": mm_val,
            "working_distance_mm": working_dist_mm,
            "working_distance_m": working_dist_m,
            "focal_length_mm": focal_mm,
            "pixel_size_microns": pixel_microns,
            "sensor_x_mm": sensor_x_mm,
            "sensor_y_mm": sensor_y_mm
        }
        
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid input: {e}")


def save_calibration():
    """Save calibration to file."""
    global calibration_data, points, image_original
    
    if 'calibration_data' not in globals():
        messagebox.showwarning("Warning", "Please calculate calibration first")
        return
    
    try:
        # Get image path
        img_path = image_path_var.get()
        
        # Create calibrations directory
        os.makedirs("calibrations", exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = ""
        if img_path:
            prefix = os.path.splitext(os.path.basename(img_path))[0] + "_"
        filename = f"{prefix}image_calibration_{timestamp}.json"
        file_path = os.path.join("calibrations", filename)
        
        # Build full calibration data
        pixel_dist = calibration_data["pixel_distance"]
        full_data = {
            "image_path": img_path,
            "point1": {"x": int(points[0][0]), "y": int(points[0][1])},
            "point2": {"x": int(points[1][0]), "y": int(points[1][1])},
            "pixel_distance": float(pixel_dist),
            "mm_measurement": float(calibration_data["mm_measurement"]),
            "pixels_per_mm": float(calibration_data["pixels_per_mm"]),
            "working_distance_mm": float(calibration_data["working_distance_mm"]),
            "working_distance_m": float(calibration_data["working_distance_m"]),
            "camera_parameters": {
                "focal_length_mm": float(calibration_data["focal_length_mm"]),
                "pixel_size_microns": float(calibration_data["pixel_size_microns"]),
                "pixel_size_m": float(calibration_data["pixel_size_microns"] * 1e-6),
                "sensor_x_mm": float(calibration_data["sensor_x_mm"]),
                "sensor_y_mm": float(calibration_data["sensor_y_mm"]),
                "sensor_x_m": float(calibration_data["sensor_x_mm"] * 1e-3),
                "sensor_y_m": float(calibration_data["sensor_y_mm"] * 1e-3)
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2)
        
        messagebox.showinfo("Success", f"Saved to:\n{file_path}")
        print(f"Calibration saved: {file_path}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save: {e}")


def reset_points():
    """Reset points."""
    global points
    points = []
    if image_original is not None:
        schedule_update()
    result_label.config(text="Click two points on the image")


def periodic_update():
    """Periodic update to keep OpenCV window responsive."""
    if image_original is not None:
        cv2.waitKey(1)
    root.after(100, periodic_update)


def on_closing():
    """Handle window close."""
    cv2.destroyAllWindows()
    root.quit()


# Create GUI
root = tk.Tk()
width, height = get_standard_size("small")
root.geometry(f"{width}x{height}")
root.minsize(width, height)
root.title(format_window_title("Image Calibration Tool", platform="Raspberry Pi"))
apply_standard_theme(root)

main = ttk.Frame(root, padding="15")
main.pack(fill="both", expand=True)

# Instructions
ttk.Label(main, text="1. Load image\n2. Click two points\n3. Enter mm\n4. Calculate & Save",
          justify="left").pack(pady=(0, 10))

# Load button
ttk.Button(main, text="📂 Load Image", command=load_image).pack(pady=5, fill="x")

image_path_var = tk.StringVar()
ttk.Label(main, textvariable=image_path_var, wraplength=350, 
          foreground="gray").pack(pady=2)

# MM entry
mm_frame = ttk.Frame(main)
mm_frame.pack(pady=10, fill="x")
ttk.Label(mm_frame, text="Measurement (mm):").pack(side="left", padx=(0, 10))
mm_entry = ttk.Entry(mm_frame, width=15)
mm_entry.pack(side="left")
mm_entry.bind("<Return>", lambda e: calculate())

# Camera parameters
cam_frame = ttk.LabelFrame(main, text="Camera Parameters", padding="10")
cam_frame.pack(pady=10, fill="x")

focal_frame = ttk.Frame(cam_frame)
focal_frame.pack(fill="x", pady=3)
ttk.Label(focal_frame, text="Focal Length (mm):").pack(side="left", padx=(0, 10))
focal_entry = ttk.Entry(focal_frame, width=15)
focal_entry.pack(side="left")
focal_entry.insert(0, str(DEFAULT_FOCAL_LENGTH_MM))

pixel_frame = ttk.Frame(cam_frame)
pixel_frame.pack(fill="x", pady=3)
ttk.Label(pixel_frame, text="Pixel Size (microns):").pack(side="left", padx=(0, 10))
pixel_entry = ttk.Entry(pixel_frame, width=15)
pixel_entry.pack(side="left")
pixel_entry.insert(0, str(DEFAULT_PIXEL_SIZE_MICRONS))

sensor_x_frame = ttk.Frame(cam_frame)
sensor_x_frame.pack(fill="x", pady=3)
ttk.Label(sensor_x_frame, text="Sensor X (mm):").pack(side="left", padx=(0, 10))
sensor_x_entry = ttk.Entry(sensor_x_frame, width=15)
sensor_x_entry.pack(side="left")
sensor_x_entry.insert(0, str(DEFAULT_SENSOR_X_MM))

sensor_y_frame = ttk.Frame(cam_frame)
sensor_y_frame.pack(fill="x", pady=3)
ttk.Label(sensor_y_frame, text="Sensor Y (mm):").pack(side="left", padx=(0, 10))
sensor_y_entry = ttk.Entry(sensor_y_frame, width=15)
sensor_y_entry.pack(side="left")
sensor_y_entry.insert(0, str(DEFAULT_SENSOR_Y_MM))

# Calculate button
ttk.Button(main, text="Calculate", command=calculate).pack(pady=5, fill="x")

# Result
result_label = ttk.Label(main, text="Calibration: Not calculated", justify="left")
result_label.pack(pady=10)

# Save button
ttk.Button(main, text="💾 Save Calibration", command=save_calibration).pack(pady=5, fill="x")

# Reset button
ttk.Button(main, text="🔄 Reset Points", command=reset_points).pack(pady=5, fill="x")

# Exit
ttk.Button(main, text="Exit", command=on_closing).pack(pady=(10, 0), fill="x")

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start periodic update for OpenCV window responsiveness
root.after(100, periodic_update)

if __name__ == "__main__":
    print("Image Calibration Tool - Raspberry Pi")
    root.mainloop()

