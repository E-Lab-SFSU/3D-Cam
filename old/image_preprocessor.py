import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.filters import difference_of_gaussians
import json
import os

class PreprocessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Preprocessor GUI")
        self.root.geometry("1400x900")
        self.config_file = "config.json"
        
        self.original_image = None
        self.processed_images = {}
        self.tk_images = {}

        # Main frame layout
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Control Panel
        self.control_panel = ttk.Frame(self.main_frame, width=300)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y)

        # Right: Single Image Display Area
        self.image_display_frame = ttk.Frame(self.main_frame)
        self.image_display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        # Main Image Canvas
        self.main_canvas = tk.Canvas(self.image_display_frame, bg="gray")
        self.main_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.main_canvas.bind("<Configure>", self.resize_images)

        self.create_controls()
        self.load_config()

    def create_controls(self):
        # File & Action Buttons
        button_frame = ttk.Frame(self.control_panel)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Current View", command=self.save_image).pack(side=tk.LEFT, padx=5)

        # Status Label
        self.status_label = ttk.Label(self.control_panel, text="Ready.")
        self.status_label.pack(fill=tk.X, pady=5)
        
        # Reset Button
        ttk.Button(self.control_panel, text="Reset to Defaults", command=self.reset_to_defaults).pack(fill=tk.X, pady=5)
        
        # Profile Buttons
        profile_frame = ttk.Frame(self.control_panel)
        profile_frame.pack(fill=tk.X, pady=5)
        ttk.Button(profile_frame, text="Save Profile", command=self.save_profile).pack(side=tk.LEFT, padx=5)
        ttk.Button(profile_frame, text="Load Profile", command=self.load_profile).pack(side=tk.LEFT, padx=5)

        self.notebook = ttk.Notebook(self.control_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # Preprocessing Tab
        pre_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(pre_tab, text="Preprocessing")
        self.create_pre_controls(pre_tab)

        # View Options Tab
        view_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(view_tab, text="View Options")
        self.create_view_controls(view_tab)

    def create_labeled_entry_scale(self, parent, text, from_, to, initial_value, decimals=1, step=1):
        """Helper to create a label, scale, and numeric entry."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=text).pack(side=tk.LEFT)

        value_var = tk.DoubleVar(value=initial_value)
        entry = ttk.Entry(frame, textvariable=value_var, width=7)
        entry.pack(side=tk.RIGHT)
        
        scale = ttk.Scale(parent, from_=from_, to=to, orient=tk.HORIZONTAL, variable=value_var)
        scale.pack(fill=tk.X)

        def on_entry_change(*args):
            try:
                val = value_var.get()
                scale.set(val)
                self.update_visualization()
            except tk.TclError:
                pass
        
        value_var.trace_add("write", on_entry_change)
        scale.config(command=lambda val: self.update_visualization())

        return value_var, scale

    def create_pre_controls(self, parent):
        self.use_clahe = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Use CLAHE", variable=self.use_clahe, command=self.update_visualization).pack(fill=tk.X, pady=2)
        
        self.low_sigma_var, _ = self.create_labeled_entry_scale(parent, "DoG Low Sigma", 0.1, 10.0, 1.5, decimals=2, step=0.1)
        self.high_sigma_var, _ = self.create_labeled_entry_scale(parent, "DoG High Sigma", 0.1, 20.0, 4.0, decimals=2, step=0.1)
        self.binary_threshold_var, _ = self.create_labeled_entry_scale(parent, "Binary Threshold", 0, 255, 127, decimals=0)

    def create_view_controls(self, parent):
        self.view_var = tk.StringVar(value="Original")
        
        ttk.Radiobutton(parent, text="Original", variable=self.view_var, value="Original", command=self.draw_visualization).pack(anchor='w', pady=2)
        ttk.Radiobutton(parent, text="Grayscale", variable=self.view_var, value="Grayscale", command=self.draw_visualization).pack(anchor='w', pady=2)
        ttk.Radiobutton(parent, text="CLAHE", variable=self.view_var, value="CLAHE", command=self.draw_visualization).pack(anchor='w', pady=2)
        ttk.Radiobutton(parent, text="DoG", variable=self.view_var, value="DoG", command=self.draw_visualization).pack(anchor='w', pady=2)
        ttk.Radiobutton(parent, text="Binary", variable=self.view_var, value="Binary", command=self.draw_visualization).pack(anchor='w', pady=2)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Error", "Could not load image.")
                return
            self.update_visualization()
            self.status_label.config(text=f"Image loaded: {os.path.basename(file_path)}")

    def save_image(self):
        current_view = self.view_var.get()
        image_to_save = self.processed_images.get(current_view)

        if image_to_save is None:
            messagebox.showinfo("Info", "No image to save. Please load an image first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            initialfile=f"processed_{current_view}.png",
            title="Save Current View"
        )
        if file_path:
            try:
                cv2.imwrite(file_path, image_to_save)
                self.status_label.config(text=f"Image saved at original dimensions to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")

    def update_visualization(self, event=None):
        if self.original_image is None:
            return
        
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.processed_images['Original'] = self.original_image
        self.processed_images['Grayscale'] = gray

        # CLAHE
        if self.use_clahe.get():
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(gray)
            self.processed_images['CLAHE'] = clahe_img
        else:
            self.processed_images['CLAHE'] = gray

        # DoG
        low_sigma = self.low_sigma_var.get()
        high_sigma = self.high_sigma_var.get()
        dog_img = difference_of_gaussians(self.processed_images['CLAHE'], low_sigma, high_sigma)
        dog_img = cv2.normalize(dog_img, None, 0, 255, cv2.NORM_MINMAX)
        dog_img = dog_img.astype(np.uint8)
        self.processed_images['DoG'] = dog_img

        # Binary
        threshold = int(self.binary_threshold_var.get())
        _, binary_img = cv2.threshold(dog_img, threshold, 255, cv2.THRESH_BINARY)
        self.processed_images['Binary'] = binary_img

        self.draw_visualization()
        self.save_config()

    def draw_visualization(self):
        current_view = self.view_var.get()
        img_to_display = self.processed_images.get(current_view)
        
        if img_to_display is None:
            self.main_canvas.delete("all")
            return
        
        if len(img_to_display.shape) == 2:
            img_to_display = cv2.cvtColor(img_to_display, cv2.COLOR_GRAY2BGR)

        self.display_image(self.main_canvas, img_to_display)

    def display_image(self, canvas, img_cv):
        h, w = img_cv.shape[:2]
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        
        if w > canvas_w or h > canvas_h:
            ratio = min(canvas_w / w, canvas_h / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)

        tk_image = ImageTk.PhotoImage(image=img_pil)
        canvas.delete("all")
        canvas.create_image(canvas_w / 2, canvas_h / 2, anchor="center", image=tk_image)
        self.tk_images[canvas] = tk_image

    def resize_images(self, event):
        if self.original_image is not None:
            self.draw_visualization()

    def reset_to_defaults(self):
        self.use_clahe.set(False)
        self.low_sigma_var.set(1.5)
        self.high_sigma_var.set(4.0)
        self.binary_threshold_var.set(127)
        self.view_var.set("Original")
        self.update_visualization()

    def save_config(self):
        config = {
            "use_clahe": self.use_clahe.get(),
            "low_sigma": self.low_sigma_var.get(),
            "high_sigma": self.high_sigma_var.get(),
            "binary_threshold": self.binary_threshold_var.get(),
            "view_option": self.view_var.get()
        }
        with open(self.config_file, "w") as f:
            json.dump(config, f)

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                config = json.load(f)
            
            self.use_clahe.set(config.get("use_clahe", False))
            self.low_sigma_var.set(config.get("low_sigma", 1.5))
            self.high_sigma_var.set(config.get("high_sigma", 4.0))
            self.binary_threshold_var.set(config.get("binary_threshold", 127))
            self.view_var.set(config.get("view_option", "Original"))

    def save_profile(self):
        profile_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Profile"
        )
        if profile_path:
            config = {
                "use_clahe": self.use_clahe.get(),
                "low_sigma": self.low_sigma_var.get(),
                "high_sigma": self.high_sigma_var.get(),
                "binary_threshold": self.binary_threshold_var.get()
            }
            try:
                with open(profile_path, "w") as f:
                    json.dump(config, f, indent=4)
                messagebox.showinfo("Success", "Profile saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save profile: {e}")

    def load_profile(self):
        profile_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Load Profile"
        )
        if profile_path:
            try:
                with open(profile_path, "r") as f:
                    config = json.load(f)
                
                self.use_clahe.set(config.get("use_clahe", False))
                self.low_sigma_var.set(config.get("low_sigma", 1.5))
                self.high_sigma_var.set(config.get("high_sigma", 4.0))
                self.binary_threshold_var.set(config.get("binary_threshold", 127))
                
                self.update_visualization()
                messagebox.showinfo("Success", "Profile loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load profile: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PreprocessorApp(root)
    root.mainloop()