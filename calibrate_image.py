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
import sys
from datetime import datetime
from typing import Optional, Tuple

# Global state
image_path: Optional[str] = None
image: Optional[np.ndarray] = None
image_original: Optional[np.ndarray] = None  # Store original unmodified image  
points: list = []  # Store clicked points
window_name = "Image Calibration - Click Two Points"
mm_measurement: Optional[float] = None
pixels_per_mm: Optional[float] = None
current_display_size = None  # (width, height) of current display window
window_initialized = False  # Track if window has been initialized
current_scale_x = 1.0  # Current scale factor for X (display to original)
current_scale_y = 1.0  # Current scale factor for Y (display to original)

# Camera parameters with defaults
DEFAULT_FOCAL_LENGTH_MM = 16.0
DEFAULT_PIXEL_SIZE_MICRONS = 3.0  # 3.00E-06 m = 3.0 microns
DEFAULT_SENSOR_X_MM = 5.76  # 5.76E-03 m = 5.76 mm
DEFAULT_SENSOR_Y_MM = 3.24  # 3.24E-03 m = 3.24 mm

# Platform detection
IS_LINUX = sys.platform.startswith('linux')
IS_RASPBERRY_PI = os.path.exists('/proc/device-tree/model') and 'raspberry' in open('/proc/device-tree/model').read().lower() if os.path.exists('/proc/device-tree/model') else False


def get_screen_size():
    """Get screen resolution using tkinter root window."""
    try:
        # Use the existing root window if available, otherwise create temporary
        if 'root' in globals() and root.winfo_exists():
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
        else:
            # Fallback: create temporary window
            root_tmp = tk.Tk()
            root_tmp.withdraw()  # Hide the temporary window
            screen_width = root_tmp.winfo_screenwidth()
            screen_height = root_tmp.winfo_screenheight()
            root_tmp.destroy()
        # On Raspberry Pi, sometimes screen size is reported incorrectly
        # Ensure minimum reasonable values
        if screen_width < 640:
            screen_width = 1920
        if screen_height < 480:
            screen_height = 1080
        return screen_width, screen_height
    except:
        # Fallback to common resolutions if tkinter fails
        return 1920, 1080


def get_display_size(image_shape, max_width=None, max_height=None):
    """
    Calculate display size maintaining aspect ratio.
    Full image will be visible, resized to max 2/3 of screen if needed.
    
    Args:
        image_shape: (height, width) or (height, width, channels) of the image  
        max_width: Maximum display width (if None, uses 2/3 of screen width)
        max_height: Maximum display height (if None, uses 2/3 of screen height)
    
    Returns:
        (display_width, display_height) tuple
    """
    img_height, img_width = image_shape[:2]
    
    # Get screen size and calculate max as 2/3 of screen
    if max_width is None or max_height is None:
        screen_width, screen_height = get_screen_size()
        if max_width is None:
            max_width = int(screen_width * 2 / 3)
        if max_height is None:
            max_height = int(screen_height * 2 / 3)
        # On Raspberry Pi, use larger size - at least 1280x720 for better visibility
        # but still respect 2/3 max
        if IS_RASPBERRY_PI or IS_LINUX:
            # On Linux/RPi, ensure minimum size for better visibility
            max_width = max(max_width, min(1280, screen_width))
            max_height = max(max_height, min(720, screen_height))
    
    aspect_ratio = img_width / img_height
    
    # Calculate display size maintaining aspect ratio
    # Ensure full image is visible, scale down if larger than max
    if img_width > max_width or img_height > max_height:
        # Image is larger than max - scale down while maintaining aspect ratio
        if aspect_ratio > 1:
            # Landscape - fit to width
            display_width = max_width
            display_height = int(max_width / aspect_ratio)
            # Check if height exceeds max (shouldn't happen, but double-check)
            if display_height > max_height:
                display_height = max_height
                display_width = int(max_height * aspect_ratio)
        else:
            # Portrait - fit to height
            display_height = max_height
            display_width = int(max_height * aspect_ratio)
            # Check if width exceeds max (shouldn't happen, but double-check)
            if display_width > max_width:
                display_width = max_width
                display_height = int(max_width / aspect_ratio)
    else:
        # Image fits within max - display at original size
        display_width = img_width
        display_height = img_height
    
    return (display_width, display_height)


def resize_image_to_fit(image, target_width, target_height):
    """
    Resize image to fit within target dimensions while maintaining aspect ratio.
    
    Args:
        image: Input image (numpy array)
        target_width: Target width
        target_height: Target height
    
    Returns:
        Resized image and scale factor
    """
    img_height, img_width = image.shape[:2]
    img_aspect = img_width / img_height
    target_aspect = target_width / target_height
    
    if img_aspect > target_aspect:
        # Image is wider - fit to width
        new_width = target_width
        new_height = int(target_width / img_aspect)
    else:
        # Image is taller - fit to height
        new_height = target_height
        new_width = int(target_height * img_aspect)
    
    if new_width > 0 and new_height > 0:
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        scale_x = new_width / img_width
        scale_y = new_height / img_height
        return resized, scale_x, scale_y
    else:
        return image, 1.0, 1.0


def update_display():
    """Update the display with current image and points."""
    global image, image_original, window_name, current_display_size, window_initialized
    global current_scale_x, current_scale_y
    
    if image_original is None:
        return
    
    # Start with a copy of the original
    image = image_original.copy()
    
    # Draw existing points and line if applicable (in original image coordinates)
    if len(points) >= 1:
        cv2.circle(image, points[0], 5, (0, 255, 0), -1)
        cv2.putText(image, "Point 1", (points[0][0] + 10, points[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if len(points) >= 2:
        cv2.circle(image, points[1], 5, (0, 255, 0), -1)
        cv2.putText(image, "Point 2", (points[1][0] + 10, points[1][1] - 10),
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
    
    # Get current window size if window is initialized, otherwise use calculated size
    was_initialized = window_initialized
    if window_initialized and current_display_size is not None:
        # Window is initialized - use current window size (allows resizing)
        try:
            window_prop = cv2.getWindowImageRect(window_name)
            if len(window_prop) >= 4 and window_prop[2] > 0 and window_prop[3] > 0:
                # Use actual window size (user may have resized it)
                display_width = window_prop[2]
                display_height = window_prop[3]
                current_display_size = (display_width, display_height)
            else:
                # Fallback to calculated size if window size can't be determined
                display_width, display_height = get_display_size(image.shape)
                current_display_size = (display_width, display_height)
        except (cv2.error, AttributeError, IndexError):
            # Fallback to calculated size if window query fails
            display_width, display_height = get_display_size(image.shape)
            current_display_size = (display_width, display_height)
    else:
        # First initialization - calculate display size
        display_width, display_height = get_display_size(image.shape)
        current_display_size = (display_width, display_height)
        window_initialized = True
    
    # Only resize window programmatically on first initialization
    if not was_initialized:
        try:
            cv2.resizeWindow(window_name, display_width, display_height)
        except cv2.error:
            # Window might not exist yet, will be created by imshow
            pass
    
    # Resize image to the current display size while maintaining aspect ratio
    display_image, scale_x, scale_y = resize_image_to_fit(image, display_width, display_height)
    
    # Store scale factors for mouse coordinate conversion
    current_scale_x = scale_x
    current_scale_y = scale_y
    
    # Show image and process window events
    cv2.imshow(window_name, display_image)
    
    # Process window events - use minimal waitKey to avoid blocking tkinter event loop
    # This is especially important on Raspberry Pi to prevent GUI freezing
    cv2.waitKey(1)  # Minimal wait - just enough to process OpenCV window events
    
    # On Linux/RPi, ensure window can receive mouse events (but don't block)
    try:
        if IS_RASPBERRY_PI or IS_LINUX:
            # Only set window properties, don't use multiple waitKey calls that block
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
    except:
        pass


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to select two points."""
    global points, current_scale_x, current_scale_y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            # Convert coordinates from scaled display image to original image coordinates
            # Mouse coordinates are in the displayed/scaled image coordinate system
            # Safety check to avoid division by zero
            if current_scale_x > 0 and current_scale_y > 0:
                original_x = int(x / current_scale_x)
                original_y = int(y / current_scale_y)
            else:
                # Fallback if scale factors are invalid
                original_x = x
                original_y = y
            
            # Clamp to image bounds
            if image_original is not None:
                img_height, img_width = image_original.shape[:2]
                original_x = max(0, min(original_x, img_width - 1))
                original_y = max(0, min(original_y, img_height - 1))
            
            points.append((original_x, original_y))
            print(f"[INFO] Point {len(points)} selected: ({original_x}, {original_y}) (display: {x}, {y})")
            
            # Schedule display update asynchronously to avoid blocking tkinter event loop
            # This prevents GUI freezing on Raspberry Pi
            root.after(10, lambda: update_display())


def load_image():
    """Load an image file."""
    global image_path, image, image_original, points, window_initialized

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
    image_original = cv2.imread(file_path)

    if image_original is None:
        messagebox.showerror("Error", f"Could not load image: {file_path}")     
        return

    # Reset points and window initialization flag
    points = []
    window_initialized = False

    # Destroy existing window if it exists (for clean restart)
    try:
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)  # Process window events
    except:
        pass

    # Create window and set mouse callback
    # Use WINDOW_NORMAL but we'll set fixed size and prevent resizing
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        # On Linux/Raspberry Pi, ensure window can receive focus and events
        try:
            # Try to bring window to front and ensure it's focusable
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
            # Give window a moment to initialize
            cv2.waitKey(10)
        except:
            pass
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create preview window: {e}\n\nMake sure you have a display available.")
        return

    # Initial display update (this will set window size and show image)
    update_display()
    
    # Minimal waitKey call to ensure window is created (avoid blocking tkinter event loop)
    # On Raspberry Pi, we schedule additional window setup asynchronously
    cv2.waitKey(1)
    
    # Schedule window focus setup asynchronously to avoid blocking
    def setup_window_focus():
        try:
            if IS_RASPBERRY_PI or IS_LINUX:
                # On Linux/RPi, ensure window receives focus for mouse events
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
                cv2.waitKey(1)  # Minimal wait
        except:
            pass
    
    root.after(50, setup_window_focus)

    print(f"[INFO] Image loaded: {file_path}")
    print(f"[INFO] Image size: {image_original.shape[1]}x{image_original.shape[0]} pixels")
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
    
    # Generate timestamped filename with prefix from image filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = ""
    if image_path:
        # Extract filename without extension
        prefix = os.path.splitext(os.path.basename(image_path))[0] + "_"
    filename = f"{prefix}image_calibration_{timestamp}.json"
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
    global points, image_original, image_path

    if image_path and os.path.exists(image_path):
        points = []
        image_original = cv2.imread(image_path)
        if image_original is not None:
            # Schedule display update asynchronously to avoid blocking
            root.after(10, lambda: update_display())
            result_label.config(text="Calibration: Not calculated")
            print("[INFO] Points reset.")
        else:
            messagebox.showerror("Error", f"Could not reload image: {image_path}")


def periodic_display_update():
    """Periodically check window size and update display if resized (allow resizing)."""
    global window_initialized, image_original, current_display_size
    
    # Only check if window is initialized and image is loaded
    if window_initialized and image_original is not None and current_display_size is not None:
        try:
            # Check if window still exists and get its size
            window_prop = cv2.getWindowImageRect(window_name)
            if len(window_prop) >= 4 and window_prop[2] > 0 and window_prop[3] > 0:
                current_size = (window_prop[2], window_prop[3])
                # Allow small tolerance for window manager differences
                size_tolerance = 5  # pixels
                size_diff = abs(current_size[0] - current_display_size[0]) + abs(current_size[1] - current_display_size[1])
                
                # If window size changed significantly (user resized it), update display to match
                if size_diff > size_tolerance:
                    # Update current_display_size to match new window size
                    current_display_size = current_size
                    # Update display to fill the new window size properly
                    # Use root.after to avoid blocking the event loop
                    root.after(10, lambda: update_display())
        except (cv2.error, AttributeError, IndexError):
            # Window doesn't exist or error, ignore
            pass
    
    # Schedule next update (every 300ms to check for resize)
    root.after(300, periodic_display_update)


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
    
    # Start periodic check to enforce fixed window size (prevent resizing)
    root.after(300, periodic_display_update)
    
    root.mainloop()

