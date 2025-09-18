# blob_tracker.py
# pip install trackpy pandas scikit-image opencv-python

import cv2, numpy as np, pandas as pd, trackpy as tp, argparse
from skimage import img_as_ubyte
from skimage.filters import difference_of_gaussians

def preprocess_image(img, args, fnum=0):
    # 1. grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if args.debug:
        print(f"[DEBUG] frame {fnum}: grayscale shape {gray.shape} min={gray.min()} max={gray.max()}")
        cv2.imshow("1. grayscale", gray)
        cv2.waitKey(0)

    # 2. contrast stretch
    gray_eq = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    if args.debug:
        print(f"[DEBUG] contrast-stretched min={gray_eq.min()} max={gray_eq.max()}")
        cv2.imshow("2. contrast stretched", gray_eq)
        cv2.waitKey(0)

    # 3. optional CLAHE
    if args.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_eq = clahe.apply(gray_eq)
        if args.debug:
            print("[DEBUG] applied CLAHE")
            cv2.imshow("3. CLAHE", gray_eq)
            cv2.waitKey(0)

    # 4. DoG filter
    bp = difference_of_gaussians(gray_eq, low_sigma=1, high_sigma=3)
    bp = (bp - bp.min()) / (np.ptp(bp) + 1e-9)
    if args.debug:
        print(f"[DEBUG] DoG filtered: min={bp.min():.3f}, max={bp.max():.3f}")
        cv2.imshow("4. DoG filtered", img_as_ubyte(bp))
        cv2.waitKey(0)

    # 5. threshold weak pixels
    if args.thresh > 0:
        mask = bp >= args.thresh
        bp = bp * mask
        if args.debug:
            print(f"[DEBUG] thresholded at {args.thresh}, kept {mask.sum()} pixels")
            cv2.imshow("5. thresholded", img_as_ubyte(bp))
            cv2.waitKey(0)

    return img_as_ubyte(bp)

def detect_blobs(img, fnum, args):
    bp = preprocess_image(img, args, fnum)
    feats = tp.locate(bp, diameter=args.diameter, minmass=args.minmass,
                      separation=args.separation, invert=False, preprocess=False)
    if feats is not None and len(feats) > 0:
        feats["frame"] = fnum
    return feats

def process_image(path, args):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    feats = detect_blobs(img, 0, args)

    vis = img.copy()
    if feats is not None:
        for _, r in feats.iterrows():
            cv2.circle(vis, (int(r.x), int(r.y)), args.diameter//2, (0,255,0), 1)

    if args.save:
        cv2.imwrite(args.save, vis)
        print(f"Saved annotated image to {args.save}")

    if not args.no_display:
        cv2.imshow("detections", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_video(path, args):
    cap = cv2.VideoCapture(path)
    fnum, rows = 0, []

    out = None
    if args.save and not args.save.endswith(".csv") and not path == 0:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(args.save, fourcc, fps, (w,h))

    while True:
        ok, frame = cap.read()
        if not ok: break
        feats = detect_blobs(frame, fnum, args)
        if feats is not None:
            rows.append(feats[["x","y","mass","size","ecc","frame"]])
            for _, r in feats.iterrows():
                cv2.circle(frame, (int(r.x), int(r.y)), args.diameter//2, (0,255,0), 1)

        if out: out.write(frame)
        if not args.no_display:
            cv2.imshow("video detections", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
        fnum += 1

    cap.release()
    if out: out.release()
    if not args.no_display: cv2.destroyAllWindows()

    if rows:
        df = pd.concat(rows, ignore_index=True)
        linked = tp.link_df(df, search_range=args.search_range, memory=args.memory)
        csv_name = args.save if args.save and args.save.endswith(".csv") else "tracks.csv"
        linked.to_csv(csv_name, index=False)
        print(f"Saved tracks to {csv_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blob tracker with preprocessing and debug mode")
    parser.add_argument("source", help="Image, video file, or 0 for webcam")
    parser.add_argument("--diameter", type=int, default=7, help="Approximate blob diameter (px)")
    parser.add_argument("--minmass", type=int, default=200, help="Reject dim blobs")
    parser.add_argument("--separation", type=int, default=6, help="Minimum spacing between blobs")
    parser.add_argument("--search-range", type=int, default=10, help="Max movement per frame (px)")
    parser.add_argument("--memory", type=int, default=5, help="Frames a particle can vanish and keep ID")
    parser.add_argument("--save", help="Save annotated image/video/CSV (output path)")
    parser.add_argument("--no-display", action="store_true", help="Disable cv2.imshow (headless mode)")
    parser.add_argument("--debug", action="store_true", help="Show debug messages + windows for each preprocessing step")
    parser.add_argument("--use-clahe", action="store_true", help="Apply CLAHE contrast enhancement")
    parser.add_argument("--thresh", type=float, default=0.0, help="Threshold (0-1) after DoG to kill weak noise")
    args = parser.parse_args()

    src = args.source
    if src.isdigit():
        process_video(int(src), args)
    elif src.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
        process_image(src, args)
    else:
        process_video(src, args)
