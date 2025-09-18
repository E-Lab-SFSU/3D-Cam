import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.filters import difference_of_gaussians
from skimage import img_as_ubyte

class BlobTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GUI Blob Tracker")
        self.root.geometry("1400x900")

        self.original_image = None
        self.processed_images = {}
        self.visualization_image = None
        self.filtered_contours = []
        self.contour_pairs = []
        self.args = None

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

        # Control panel widgets
        self.create_controls()

    def create_controls(self):
        # File & Action Buttons
        button_frame = ttk.Frame(self.control_panel)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)

        # Debug Flag
        self.debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_panel, text="Debug Info", variable=self.debug_var).pack(anchor='w', pady=5)
        
        # Status Label
        self.status_label = ttk.Label(self.control_panel, text="Ready.")
        self.status_label.pack(fill=tk.X, pady=5)

        self.notebook = ttk.Notebook(self.control_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # Preprocessing Tab
        pre_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(pre_tab, text="Preprocessing")
        self.create_pre_controls(pre_tab)

        # Blob Filters Tab (new)
        blob_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(blob_tab, text="Contour Filters")
        self.create_contour_controls(blob_tab)
        
        # Preprocessed Image View Tab
        pre_view_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(pre_view_tab, text="View Options")
        self.create_pre_view_controls(pre_view_tab)

    def create_pre_controls(self, parent):
        self.use_clahe = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Use CLAHE", variable=self.use_clahe, command=self.update_preview).pack(fill=tk.X, pady=2)
        
        ttk.Label(parent, text="DoG Low Sigma").pack(fill=tk.X)
        self.low_sigma_scale = ttk.Scale(parent, from_=0.1, to=10.0, orient=tk.HORIZONTAL, command=self.update_preview)
        self.low_sigma_scale.set(1.5)
        self.low_sigma_scale.pack(fill=tk.X)

        ttk.Label(parent, text="DoG High Sigma").pack(fill=tk.X)
        self.high_sigma_scale = ttk.Scale(parent, from_=0.1, to=20.0, orient=tk.HORIZONTAL, command=self.update_preview)
        self.high_sigma_scale.set(4.0)
        self.high_sigma_scale.pack(fill=tk.X)
        
        # Adaptive Thresholding Controls
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=5)
        self.use_adaptive_thresh = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Use Adaptive Threshold", variable=self.use_adaptive_thresh, command=self.update_preview).pack(fill=tk.X, pady=2)

        ttk.Label(parent, text="Adaptive Block Size (Odd)").pack(fill=tk.X)
        self.adaptive_block_size = ttk.Scale(parent, from_=3, to=101, orient=tk.HORIZONTAL, command=self.update_preview)
        self.adaptive_block_size.set(11)
        self.adaptive_block_size.pack(fill=tk.X)
        
        ttk.Label(parent, text="Threshold (0-1)").pack(fill=tk.X)
        self.thresh_scale = ttk.Scale(parent, from_=0.0, to=1.0, orient=tk.HORIZONTAL, command=self.update_preview)
        self.thresh_scale.set(0.2)
        self.thresh_scale.pack(fill=tk.X)

    def create_contour_controls(self, parent):
        ttk.Label(parent, text="Min Area").pack(fill=tk.X)
        self.min_area_scale = ttk.Scale(parent, from_=1, to=500, orient=tk.HORIZONTAL, command=self.update_preview)
        self.min_area_scale.set(10)
        self.min_area_scale.pack(fill=tk.X)

        ttk.Label(parent, text="Max Area").pack(fill=tk.X)
        self.max_area_scale = ttk.Scale(parent, from_=100, to=5000, orient=tk.HORIZONTAL, command=self.update_preview)
        self.max_area_scale.set(1000)
        self.max_area_scale.pack(fill=tk.X)
        
        ttk.Label(parent, text="Max Shape Diff (0-1)").pack(fill=tk.X)
        self.max_shape_diff_scale = ttk.Scale(parent, from_=0.0, to=1.0, orient=tk.HORIZONTAL, command=self.update_preview)
        self.max_shape_diff_scale.set(0.2)
        self.max_shape_diff_scale.pack(fill=tk.X)

        ttk.Label(parent, text="Max Scale Diff").pack(fill=tk.X)
        self.max_scale_diff_scale = ttk.Scale(parent, from_=1.0, to=5.0, orient=tk.HORIZONTAL, command=self.update_preview)
        self.max_scale_diff_scale.set(1.5)
        self.max_scale_diff_scale.pack(fill=tk.X)

        ttk.Label(parent, text="Max Angle from Center").pack(fill=tk.X)
        self.max_angle_scale = ttk.Scale(parent, from_=0, to=180, orient=tk.HORIZONTAL, command=self.update_preview)
        self.max_angle_scale.set(90)
        self.max_angle_scale.pack(fill=tk.X)

        ttk.Label(parent, text="Max Pair Distance").pack(fill=tk.X)
        self.max_distance_scale = ttk.Scale(parent, from_=1, to=1000, orient=tk.HORIZONTAL, command=self.update_preview)
        self.max_distance_scale.set(200)
        self.max_distance_scale.pack(fill=tk.X)
        
    def create_pre_view_controls(self, parent):
        self.pre_view_var = tk.StringVar(value="Final")
        
        ttk.Radiobutton(parent, text="Original", variable=self.pre_view_var, value="Original", command=self.update_processed_view).pack(anchor='w', pady=2)
        ttk.Radiobutton(parent, text="Grayscale", variable=self.pre_view_var, value="Grayscale", command=self.update_processed_view).pack(anchor='w', pady=2)
        ttk.Radiobutton(parent, text="CLAHE", variable=self.pre_view_var, value="CLAHE", command=self.update_processed_view).pack(anchor='w', pady=2)
        ttk.Radiobutton(parent, text="DoG", variable=self.pre_view_var, value="DoG", command=self.update_processed_view).pack(anchor='w', pady=2)
        ttk.Radiobutton(parent, text="Final (Thresholded)", variable=self.pre_view_var, value="Final", command=self.update_processed_view).pack(anchor='w', pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=10)

        self.overlay_pairs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Overlay Pairs", variable=self.overlay_pairs_var, command=self.update_processed_view).pack(anchor='w', pady=2)
        self.overlay_contours_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Overlay Contours", variable=self.overlay_contours_var, command=self.update_processed_view).pack(anchor='w', pady=2)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Error", "Could not load image.")
                return
            self.filtered_contours = []
            self.contour_pairs = []
            self.update_preview()
            self.status_label.config(text=f"Image loaded: {file_path.split('/')[-1]}")

    def update_preview(self, event=None):
        if self.original_image is None:
            return

        # Get parameters from GUI
        class Args:
            pass
        self.args = Args()
        self.args.use_clahe = self.use_clahe.get()
        self.args.thresh = self.thresh_scale.get()
        self.args.low_sigma = self.low_sigma_scale.get()
        self.args.high_sigma = self.high_sigma_scale.get()
        self.args.min_area = self.min_area_scale.get()
        self.args.max_area = self.max_area_scale.get()
        self.args.max_shape_diff = self.max_shape_diff_scale.get()
        self.args.max_scale_diff = self.max_scale_diff_scale.get()
        self.args.max_angle_from_center = self.max_angle_scale.get()
        self.args.max_pair_distance = self.max_distance_scale.get()
        self.args.use_adaptive_thresh = self.use_adaptive_thresh.get()
        self.args.adaptive_block_size = self.adaptive_block_size.get()

        if self.debug_var.get():
            print("--- Updating Preview ---")
            print(f"Parameters: {self.args.__dict__}")

        # Preprocess the image, but don't find contours or pairs yet
        pre_img = self.preprocess_image(self.original_image, self.args)
        
        # Trigger the view update which will decide if it needs to find contours/pairs
        self.update_processed_view()

    def update_processed_view(self, event=None):
        if self.original_image is None or self.args is None:
            return

        # Check if we need to recalculate contours and pairs
        if self.overlay_pairs_var.get() or self.overlay_contours_var.get():
            pre_img = self.processed_images.get('Final')
            self.filtered_contours = self.find_and_filter_contours(pre_img, self.args)
            self.contour_pairs = self.find_contour_pairs(self.filtered_contours, self.args)
            self.status_label.config(text=f"Preview updated. Filtered Contours: {len(self.filtered_contours)}, Pairs: {len(self.contour_pairs)}")
            if self.debug_var.get():
                print(f"Filtered contours: {len(self.filtered_contours)}")
                print(f"Contour pairs: {len(self.contour_pairs)}")
        else:
            self.status_label.config(text="Preview updated. Overlays disabled.")
            self.filtered_contours = []
            self.contour_pairs = []

        view_type = self.pre_view_var.get()
        
        base_img = self.processed_images.get(view_type)
        if base_img is None:
            return

        if len(base_img.shape) == 2:
            img_to_display = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            img_to_display = base_img.copy()

        # Overlay blobs if the checkbox is checked
        if self.overlay_contours_var.get():
            self.draw_contours(img_to_display, self.filtered_contours)
        
        # Overlay pairs if the checkbox is checked
        if self.overlay_pairs_var.get():
            self.draw_pairs(img_to_display, self.contour_pairs)

        # Add text overlays
        pair_count = len(self.contour_pairs)
        contour_count = len(self.filtered_contours)
        pairs_text = f"Pairs: {pair_count}"
        contours_text = f"Contours: {contour_count}"
        
        cv2.putText(img_to_display, pairs_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_to_display, contours_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        if img_to_display is not None:
            self.display_image(self.main_canvas, img_to_display)
        else:
            self.main_canvas.delete("all")

    def display_image(self, canvas, img_cv):
        h, w = img_cv.shape[:2]
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize image to fit the canvas
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
        canvas.image = tk_image  # Keep a reference to prevent garbage collection

    def resize_images(self, event):
        if self.original_image is not None:
            self.update_processed_view()

    # --- Core Processing & Segmentation Functions ---

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
        bp_u8 = img_as_ubyte(bp)
        self.processed_images['DoG'] = bp_u8

        pre_u8 = bp_u8.copy()
        
        # Adaptive or Global Thresholding based on user selection
        if args.use_adaptive_thresh:
            block_size = int(args.adaptive_block_size)
            if block_size % 2 == 0:
                block_size += 1
            thresh_type = cv2.THRESH_BINARY
            pre_u8 = cv2.adaptiveThreshold(pre_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, block_size, 2)
        else:
            thresh_type = cv2.THRESH_BINARY
            _, pre_u8 = cv2.threshold(pre_u8, int(args.thresh * 255), 255, thresh_type)
            
        # The 'Final' image stored here is the binary image for contour detection.
        # It's always black background with white blobs.
        self.processed_images['Final'] = pre_u8
        
        return pre_u8

    def find_and_filter_contours(self, final_image, args):
        # We use cv2.RETR_CCOMP to retrieve all contours and organize them into a two-level hierarchy.
        # This allows us to detect individual contours that might be nested inside a larger, merged one.
        contours, _ = cv2.findContours(final_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if args.min_area <= area <= args.max_area:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    hu_moments = cv2.HuMoments(M).flatten()
                    
                    filtered_contours.append({
                        'contour': contour,
                        'center': (cx, cy),
                        'area': area,
                        'hu_moments': hu_moments
                    })
        return filtered_contours

    def find_contour_pairs(self, contours, args):
        pairs = []
        contours_used = [False] * len(contours)
        
        # Get the center of the image to check the ray direction
        img_h, img_w, _ = self.original_image.shape
        center_x, center_y = img_w // 2, img_h // 2
        
        for i in range(len(contours)):
            if contours_used[i]:
                continue
            
            contour1 = contours[i]
            best_match_idx = -1
            min_shape_diff = float('inf')

            for j in range(len(contours)):
                if i == j or contours_used[j]:
                    continue
                
                contour2 = contours[j]
                
                # Filter 1: Max Pair Distance
                dist = np.linalg.norm(np.array(contour1['center']) - np.array(contour2['center']))
                if dist > args.max_pair_distance:
                    continue

                # Filter 2: Max Scale Difference
                area_ratio = max(contour1['area'], contour2['area']) / min(contour1['area'], contour2['area'])
                if area_ratio > args.max_scale_diff:
                    continue

                # Filter 3: Ray Direction Constraint
                v_center = np.array([center_x - contour1['center'][0], center_y - contour1['center'][1]])
                v_pair = np.array([contour2['center'][0] - contour1['center'][0], contour2['center'][1] - contour1['center'][1]])
                
                v_center_norm = np.linalg.norm(v_center)
                v_pair_norm = np.linalg.norm(v_pair)
                
                if v_center_norm == 0 or v_pair_norm == 0:
                    continue
                
                dot_product = np.dot(v_center, v_pair)
                cos_angle = dot_product / (v_center_norm * v_pair_norm)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                
                if angle > args.max_angle_from_center:
                    continue

                # Filter 4: Min Shape Difference
                shape_diff = cv2.matchShapes(contour1['contour'], contour2['contour'], cv2.CONTOURS_MATCH_I1, 0.0)

                if shape_diff < min_shape_diff:
                    min_shape_diff = shape_diff
                    best_match_idx = j

            if best_match_idx != -1 and min_shape_diff <= args.max_shape_diff:
                pairs.append((contour1, contours[best_match_idx]))
                contours_used[i] = True
                contours_used[best_match_idx] = True
                
        return pairs

    def draw_contours(self, vis, contours):
        for contour in contours:
            cv2.drawContours(vis, [contour['contour']], -1, (0, 255, 255), 2)
            cv2.circle(vis, contour['center'], 3, (0, 255, 0), -1)

    def draw_pairs(self, vis, pairs):
        for contour1, contour2 in pairs:
            cv2.line(vis, contour1['center'], contour2['center'], (255, 0, 255), 2)
            
if __name__ == "__main__":
    root = tk.Tk()
    app = BlobTrackerApp(root)
    root.mainloop()
