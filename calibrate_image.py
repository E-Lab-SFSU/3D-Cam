#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Calibration Tool

This tool allows you to:
  • Load an image screenshot
  • Click two points to mark a known distance
  • Enter the measurement in millimeters
  • Calculate pixels/mm ratio
  • Save the calibration data to a file
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from datetime import datetime
from typing import Optional, Tuple

# Global state
image_path: Optional[str] = None
image: Optional[np.ndarray] = None
points: list = []  # Store clicked points
window_name = "Image Calibration - Click Two Points"
mm_measurement: Optional[float] = None
pixels_per_mm: Optional[float] = None

# Camera parameters with defaults
DEFAULT_FOCAL_LENGTH_MM = 16.0
DEFAULT_PIXEL_SIZE_MICRONS = 3.0  # 3.00E-06 m = 3.0 microns
DEFAULT_SENSOR_X_MM = 5.76  # 5.76E-03 m = 5.76 mm
DEFAULT_SENSOR_Y_MM = 3.24  # 3.24E-03 m = 3.24 mm


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to select two points."""
    global points, image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            print(f"[INFO] Point {len(points)} selected: ({x}, {y})")
            
            # Draw point on image
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            if len(points) == 1:
                cv2.putText(image, "Point 1", (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif len(points) == 2:
                cv2.putText(image, "Point 2", (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Draw line between points
                cv2.line(image, points[0], points[1], (0, 255, 0), 2)
                # Calculate distance in pixels
                pixel_distance = np.sqrt((points[1][0] - points[0][0])**2 + 
                                        (points[1][1] - points[0][1])**2)
                cv2.putText(image, f"{pixel_distance:.1f} px", 
                           ((points[0][0] + points[1][0]) // 2, 
                            (points[0][1] + points[1][1]) // 2 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(window_name, image)


def load_image():
    """Load an image file."""
    global image_path, image, points
    
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        return
    
    image_path = file_path
    image = cv2.imread(file_path)
    
    if image is None:
        messagebox.showerror("Error", f"Could not load image: {file_path}")
        return
    
    # Reset points
    points = []
    
    # Create window and set mouse callback
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, image)
    
    print(f"[INFO] Image loaded: {file_path}")
    print("[INFO] Click two points on the image to mark a known distance.")


def calculate_calibration():
    """Calculate pixels/mm and working distance, then update GUI."""
    global pixels_per_mm, mm_measurement, points
    
    if len(points) != 2:
        messagebox.showwarning("Warning", "Please click two points on the image first.")
        return
    
    try:
        mm_val = float(mm_entry.get())
        if mm_val <= 0:
            messagebox.showerror("Error", "Measurement must be greater than zero.")
            return
        
        # Get camera parameters for working distance calculation
        try:
            focal_length_mm = float(focal_length_entry.get())
            pixel_size_microns = float(pixel_size_entry.get())
            if focal_length_mm <= 0 or pixel_size_microns <= 0:
                raise ValueError("Camera parameters must be greater than zero")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid camera parameters for working distance calculation.")
            return
        
        mm_measurement = mm_val
        pixel_distance = np.sqrt((points[1][0] - points[0][0])**2 + 
                                (points[1][1] - points[0][1])**2)
        pixels_per_mm = pixel_distance / mm_measurement
        
        # Calculate working distance
        # Formula: working_distance = (focal_length * object_size) / (image_size_on_sensor)
        # image_size_on_sensor = pixel_distance * pixel_size_mm
        pixel_size_mm = pixel_size_microns / 1000.0  # Convert microns to mm
        image_size_on_sensor_mm = pixel_distance * pixel_size_mm
        working_distance_mm = (focal_length_mm * mm_measurement) / image_size_on_sensor_mm if image_size_on_sensor_mm > 0 else 0
        working_distance_m = working_distance_mm / 1000.0  # Convert to meters
        
        # Update result label
        result_label.config(
            text=f"Calibration: {pixels_per_mm:.4f} pixels/mm\n"
                 f"Distance: {pixel_distance:.2f} px = {mm_measurement:.2f} mm\n"
                 f"Working Distance: {working_distance_mm:.2f} mm ({working_distance_m:.4f} m)"
        )
        
        print(f"[INFO] Calibration calculated: {pixels_per_mm:.4f} pixels/mm")
        print(f"[INFO] Distance: {pixel_distance:.2f} px = {mm_measurement:.2f} mm")
        print(f"[INFO] Working Distance: {working_distance_mm:.2f} mm ({working_distance_m:.4f} m)")
        
    except ValueError as e:
        messagebox.showerror("Error", f"Please enter valid numbers: {e}")


def save_calibration():
    """Save calibration data to file automatically with timestamp."""
    global pixels_per_mm, mm_measurement, points, image_path
    
    if pixels_per_mm is None:
        messagebox.showwarning("Warning", "Please calculate the calibration first.")
        return
    
    # Get camera parameters
    try:
        focal_length_mm = float(focal_length_entry.get())
        pixel_size_microns = float(pixel_size_entry.get())
        sensor_x_mm = float(sensor_x_entry.get())
        sensor_y_mm = float(sensor_y_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers for all camera parameters.")
        return
    
    # Create calibrations folder if it doesn't exist
    calibrations_dir = "calibrations"
    os.makedirs(calibrations_dir, exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_calibration_{timestamp}.json"
    file_path = os.path.join(calibrations_dir, filename)
    
    # Calculate working distance for saving
    pixel_distance = np.sqrt((points[1][0] - points[0][0])**2 + 
                            (points[1][1] - points[0][1])**2)
    pixel_size_mm = pixel_size_microns / 1000.0  # Convert microns to mm
    image_size_on_sensor_mm = pixel_distance * pixel_size_mm
    working_distance_mm = (focal_length_mm * mm_measurement) / image_size_on_sensor_mm if image_size_on_sensor_mm > 0 else 0
    working_distance_m = working_distance_mm / 1000.0  # Convert to meters
    
    calibration_data = {
        "image_path": image_path,
        "point1": {"x": int(points[0][0]), "y": int(points[0][1])},
        "point2": {"x": int(points[1][0]), "y": int(points[1][1])},
        "pixel_distance": float(pixel_distance),
        "mm_measurement": float(mm_measurement),
        "pixels_per_mm": float(pixels_per_mm),
        "working_distance_mm": float(working_distance_mm),
        "working_distance_m": float(working_distance_m),
        "camera_parameters": {
            "focal_length_mm": float(focal_length_mm),
            "pixel_size_microns": float(pixel_size_microns),
            "pixel_size_m": float(pixel_size_microns * 1e-6),  # Convert microns to meters
            "sensor_x_mm": float(sensor_x_mm),
            "sensor_y_mm": float(sensor_y_mm),
            "sensor_x_m": float(sensor_x_mm * 1e-3),  # Convert mm to meters
            "sensor_y_m": float(sensor_y_mm * 1e-3)   # Convert mm to meters
        }
    }
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2)
        messagebox.showinfo("Success", f"Calibration saved to:\n{file_path}")
        print(f"[INFO] Calibration saved to: {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save calibration: {e}")


def reset_points():
    """Reset points and reload image."""
    global points, image, image_path
    
    if image_path and os.path.exists(image_path):
        points = []
        image = cv2.imread(image_path)
        cv2.imshow(window_name, image)
        result_label.config(text="Calibration: Not calculated")
        print("[INFO] Points reset.")


def on_closing():
    """Handle window closing."""
    cv2.destroyAllWindows()
    root.quit()


# Create GUI
root = tk.Tk()
root.title("Image Calibration Tool")
root.geometry("400x550")
root.resizable(True, True)
root.minsize(400, 550)

# Style
style = ttk.Style(root)
try:
    style.theme_use("clam")
except:
    pass

# Main frame
main_frame = ttk.Frame(root, padding="15")
main_frame.pack(fill="both", expand=True)

# Instructions
instructions = ttk.Label(
    main_frame,
    text="1. Load an image\n2. Click two points\n3. Enter mm measurement\n4. Enter camera parameters\n5. Calculate & Save",
    justify="left"
)
instructions.pack(pady=(0, 15))

# Load image button
load_btn = ttk.Button(main_frame, text="📂 Load Image", command=load_image)
load_btn.pack(pady=5, fill="x")

# MM measurement entry
mm_frame = ttk.Frame(main_frame)
mm_frame.pack(pady=10, fill="x")

ttk.Label(mm_frame, text="Measurement (mm):").pack(side="left", padx=(0, 10))
mm_entry = ttk.Entry(mm_frame, width=15)
mm_entry.pack(side="left")
mm_entry.bind("<Return>", lambda e: calculate_calibration())

# Camera parameters frame
camera_frame = ttk.LabelFrame(main_frame, text="Camera Parameters", padding="10")
camera_frame.pack(pady=10, fill="x")

# Focal length
focal_frame = ttk.Frame(camera_frame)
focal_frame.pack(fill="x", pady=3)
ttk.Label(focal_frame, text="Focal Length (mm):").pack(side="left", padx=(0, 10))
focal_length_entry = ttk.Entry(focal_frame, width=15)
focal_length_entry.pack(side="left")
focal_length_entry.insert(0, str(DEFAULT_FOCAL_LENGTH_MM))

# Pixel size
pixel_frame = ttk.Frame(camera_frame)
pixel_frame.pack(fill="x", pady=3)
ttk.Label(pixel_frame, text="Pixel Size (microns):").pack(side="left", padx=(0, 10))
pixel_size_entry = ttk.Entry(pixel_frame, width=15)
pixel_size_entry.pack(side="left")
pixel_size_entry.insert(0, str(DEFAULT_PIXEL_SIZE_MICRONS))

# Sensor X size
sensor_x_frame = ttk.Frame(camera_frame)
sensor_x_frame.pack(fill="x", pady=3)
ttk.Label(sensor_x_frame, text="Sensor X Size (mm):").pack(side="left", padx=(0, 10))
sensor_x_entry = ttk.Entry(sensor_x_frame, width=15)
sensor_x_entry.pack(side="left")
sensor_x_entry.insert(0, str(DEFAULT_SENSOR_X_MM))

# Sensor Y size
sensor_y_frame = ttk.Frame(camera_frame)
sensor_y_frame.pack(fill="x", pady=3)
ttk.Label(sensor_y_frame, text="Sensor Y Size (mm):").pack(side="left", padx=(0, 10))
sensor_y_entry = ttk.Entry(sensor_y_frame, width=15)
sensor_y_entry.pack(side="left")
sensor_y_entry.insert(0, str(DEFAULT_SENSOR_Y_MM))

# Calculate button
calc_btn = ttk.Button(main_frame, text="Calculate", command=calculate_calibration)
calc_btn.pack(pady=5, fill="x")

# Result label
result_label = ttk.Label(main_frame, text="Calibration: Not calculated", justify="left")
result_label.pack(pady=10)

# Save button
save_btn = ttk.Button(main_frame, text="💾 Save Calibration", command=save_calibration)
save_btn.pack(pady=5, fill="x")

# Reset button
reset_btn = ttk.Button(main_frame, text="🔄 Reset Points", command=reset_points)
reset_btn.pack(pady=5, fill="x")

# Exit button
exit_btn = ttk.Button(main_frame, text="Exit", command=on_closing)
exit_btn.pack(pady=(10, 0), fill="x")

# Handle window closing
root.protocol("WM_DELETE_WINDOW", on_closing)

if __name__ == "__main__":
    print("[INFO] Image Calibration Tool started.")
    print("[INFO] Use the GUI to load an image and calibrate.")
    root.mainloop()

