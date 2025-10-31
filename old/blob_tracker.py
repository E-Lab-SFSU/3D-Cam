import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import difference_of_gaussians
from math import sqrt, pi
import csv
import os
from scipy.spatial import distance

class BlobTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GUI Blob Tracker")
        self.root.geometry("1400x900")

        self.original_image = None
        self.processed_images = {}
        self.blobs = []
        self.pairs = []
        self.args = None

        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.control_panel = ttk.Frame(self.main_frame, width=300)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y)

        self.image_display_frame = ttk.Frame(self.main_frame)
        self.image_display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        self.main_canvas = tk.Canvas(self.image_display_frame, bg="gray")
        self.main_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.main_canvas.bind("<Configure>", self.resize_images)

        self.create_controls()

    def create_controls(self):
        button_frame = ttk.Frame(self.control_panel)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Data", command=self.save_data).pack(side=tk.LEFT, padx=5)

        self.debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_panel, text="Debug Info", variable=self.debug_var).pack(anchor='w', pady=5)
        
        self.status_label = ttk.Label(self.control_panel, text="Ready.")
        self.status_label.pack(fill=tk.X, pady=5)

        self.notebook = ttk.Notebook(self.control_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        pre_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(pre_tab, text="Preprocessing")
        self.create_pre_controls(pre_tab)

        pair_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(pair_tab, text="Pair Filters")
        self.create_pair_controls(pair_tab)
        
        pre_view_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(pre_view_tab, text="View Options")
        self.create_pre_view_controls(pre_view_tab)

    def create_labeled_scale(self, parent, text, from_, to, initial_value, decimals=1):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=text).pack(side=tk.LEFT)
        
        value_var = tk.StringVar()
        value_label = ttk.Label(frame, textvariable=value_var)
        value_label.pack(side=tk.RIGHT)
        
        scale = ttk.Scale(parent, from_=from_, to=to, orient=tk.HORIZONTAL, 
                          command=lambda val: self.update_value_label(val, value_var, decimals))
        scale.set(initial_value)
        scale.pack(fill=tk.X)
        self.update_value_label(initial_value, value_var, decimals)
        return scale

    def update_value_label(self, val, var, decimals):
        if decimals == 0:
            var.set(f"{float(val):.0f}")
        else:
            var.set(f"{float(val):.{decimals}f}")
        self.update_visualization_and_data()

    def create_pre_controls(self, parent):
        self.use_clahe = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Use CLAHE", variable=self.use_clahe, command=self.update_visualization_and_data).pack(fill=tk.X, pady=2)
        
        self.low_sigma_scale = self.create_labeled_scale(parent, "DoG Low Sigma", 0.1, 10.0, 1.5)
        self.high_sigma_scale = self.create_labeled_scale(parent, "DoG High Sigma", 0.1, 20.0, 4.0)

    def create_pair_controls(self, parent):
        self.max_pair_distance = self.create_labeled_scale(parent, "Max Pair Distance", 1, 150, 50, decimals=0)
        self.min_blob_diameter = self.create_labeled_scale(parent, "Min Blob Diameter", 1, 50, 5, decimals=0)
        self.max_blob_diameter = self.create_labeled_scale(parent, "Max Blob Diameter", 1, 100, 50, decimals=0)
        self.peak_min_distance = self.create_labeled_scale(parent, "Peak Min Distance", 1, 50, 15, decimals=0)
        
    def create_pre_view_controls(self, parent):
        self.pre_view_var = tk.StringVar(value="DoG")
        
        ttk.Radiobutton(parent, text="Original", variable=self.pre_view_var, value="Original", command=self.draw_visualization).pack(anchor='w', pady=2)
        ttk.Radiobutton(parent, text="Grayscale", variable=self.pre_view_var, value="Grayscale", command=self.draw_visualization).pack(anchor='w', pady=2)
        ttk.Radiobutton(parent, text="CLAHE", variable=self.pre_view_var, value="CLAHE", command=self.draw_visualization).pack(anchor='w', pady=2)
        ttk.Radiobutton(parent, text="DoG", variable=self.pre_view_var, value="DoG", command=self.draw_visualization).pack(anchor='w', pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=10)

        self.overlay_blobs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Overlay Pairs", variable=self.overlay_blobs_var, command=self.draw_visualization).pack(anchor='w', pady=2)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Error", "Could not load image.")
                return
            self.blobs = []
            self.pairs = []
            self.update_visualization_and_data()
            self.status_label.config(text=f"Image loaded: {file_path.split('/')[-1]}")

    def save_data(self):
        if not self.pairs:
            self.status_label.config(text="No pairs to save. Please run detection first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Pair Data"
        )
        if not file_path:
            self.status_label.config(text="Save operation cancelled.")
            return

        try:
            with open(file_path, 'w', newline='') as csvfile:
                fieldnames = ['pair_id', 'blob1_x', 'blob1_y', 'blob2_x', 'blob2_y', 'distance']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for i, pair in enumerate(self.pairs):
                    writer.writerow({
                        'pair_id': i,
                        'blob1_x': pair['blobs'][0]['centroid'][0],
                        'blob1_y': pair['blobs'][0]['centroid'][1],
                        'blob2_x': pair['blobs'][1]['centroid'][0],
                        'blob2_y': pair['blobs'][1]['centroid'][1],
                        'distance': pair['distance']
                    })
            self.status_label.config(text=f"Data saved to {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving the file: {e}")
            self.status_label.config(text="Failed to save data.")

    def update_visualization_and_data(self, event=None):
        if self.original_image is None:
            return

        class Args:
            pass
        self.args = Args()
        self.args.use_clahe = self.use_clahe.get()
        self.args.low_sigma = self.low_sigma_scale.get()
        self.args.high_sigma = self.high_sigma_scale.get()
        self.args.min_blob_diameter = self.min_blob_diameter.get()
        self.args.max_blob_diameter = self.max_blob_diameter.get()
        self.args.max_pair_distance = self.max_pair_distance.get()
        self.args.peak_min_distance = int(self.peak_min_distance.get())
        
        if self.debug_var.get():
            print("--- Updating All ---")
            print(f"Parameters: {self.args.__dict__}")
        
        pre_img = self.preprocess_image(self.original_image, self.args)
        
        self.blobs = self.find_local_maxima_blobs(pre_img, self.args)
        
        self.pairs = self.group_blobs_into_pairs(self.blobs, self.args)
        
        self.status_label.config(text=f"Pairs found: {len(self.pairs)}")
        self.draw_visualization()

    def draw_visualization(self, event=None):
        if self.original_image is None or self.args is None or not self.processed_images:
            return

        view_type = self.pre_view_var.get()
        base_img = self.processed_images.get(view_type)
        if base_img is None:
            return

        if len(base_img.shape) == 2:
            img_to_display = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            img_to_display = base_img.copy()

        if self.overlay_blobs_var.get():
            self.draw_pairs(img_to_display, self.pairs)
        
        pair_count = len(self.pairs)
        pairs_text = f"Pairs: {pair_count}"
        
        cv2.putText(img_to_display, pairs_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        if img_to_display is not None:
            self.display_image(self.main_canvas, img_to_display)
        else:
            self.main_canvas.delete("all")

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
        canvas.image = tk_image

    def resize_images(self, event):
        if self.original_image is not None:
            self.draw_visualization()

    def preprocess_image(self, img, args):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.processed_images['Original'] = img
        self.processed_images['Grayscale'] = gray

        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        if args.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
        self.processed_images['CLAHE'] = gray
        
        gray = cv2.medianBlur(gray, 3)

        bp = difference_of_gaussians(gray, low_sigma=args.low_sigma, high_sigma=args.high_sigma)
        bp = (bp - bp.min()) / (np.ptp(bp) + 1e-9)
        self.processed_images['DoG'] = cv2.convertScaleAbs(bp * 255)
        
        return bp

    def find_local_maxima_blobs(self, image_DoG, args):
        blobs = []
        
        # Find all local peaks in the DoG image
        coordinates = peak_local_max(image_DoG, min_distance=args.peak_min_distance)

        # Iterate over each peak and get its properties
        for (cy, cx) in coordinates:
            # We don't have a contour, but we can estimate blob properties
            # by finding the size of the bright spot around the centroid.
            # A simple way is to find the area of pixels above a certain intensity.
            mask = np.zeros(image_DoG.shape, dtype="uint8")
            cv2.circle(mask, (cx, cy), 10, 255, -1) # Use a small circle around the peak
            
            masked_region = image_DoG * (mask / 255)
            # Use Otsu's threshold on the local region to get an area estimate
            local_thresh_val = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            _, local_binary = cv2.threshold(cv2.convertScaleAbs(masked_region * 255), 0, 255, local_thresh_val)
            
            area = cv2.countNonZero(local_binary)
            if area > 0:
                diameter = 2 * sqrt(area / pi)
                if args.min_blob_diameter <= diameter <= args.max_blob_diameter:
                    blobs.append({
                        'centroid': (cx, cy),
                        'diameter': diameter
                    })
        return blobs

    def group_blobs_into_pairs(self, blobs, args):
        pairs = []
        blobs_to_process = list(blobs)
        
        while blobs_to_process:
            blob1 = blobs_to_process.pop(0)
            closest_blob = None
            min_dist = float('inf')

            for blob2 in blobs_to_process:
                dist = distance.euclidean(blob1['centroid'], blob2['centroid'])
                if dist < min_dist:
                    min_dist = dist
                    closest_blob = blob2

            if closest_blob and min_dist <= args.max_pair_distance:
                blobs_to_process.remove(closest_blob)
                pairs.append({
                    'blobs': [blob1, closest_blob],
                    'distance': min_dist
                })
        return pairs

    def draw_pairs(self, vis, pairs):
        """Draws pair components and connecting lines."""
        for i, pair in enumerate(pairs):
            c1 = pair['blobs'][0]['centroid']
            c2 = pair['blobs'][1]['centroid']
            
            cv2.line(vis, c1, c2, (255, 0, 0), 2)
            cv2.circle(vis, c1, 5, (0, 255, 255), -1)
            cv2.circle(vis, c2, 5, (0, 255, 255), -1)
            
            mid_point = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
            cv2.putText(vis, str(i), (mid_point[0] + 10, mid_point[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

if __name__ == "__main__":
    root = tk.Tk()
    app = BlobTrackerApp(root)
    root.mainloop()