# Deprecated Code

This folder contains deprecated or experimental code that is no longer recommended for general use.

## Deprecated Components

### `detect_pairs_yolo.py` - YOLO-based Detection (Experimental/Deprecated)

**Status:** Experimental / Not Recommended  
**Location:** `deprecated/detect_pairs_yolo.py` (moved from `apps/`)

**Why it's deprecated:**
- AI-based object detection (YOLO) is overkill for simple particle blob detection
- Traditional computer vision (binary thresholding + contours) is more appropriate for this use case:
  - **No training required** - Works immediately with any particle video
  - **Faster and more predictable** - No model inference overhead
  - **Better suited for simple blob shapes** - Particles are typically small, simple shapes
  - **Geometric pairing doesn't need AI** - Pairing is based on physical constraints (angle, radius, colinearity), not learned patterns

**Technical issues:**
- YOLO detects bounding boxes, but still requires thresholding inside each box to calculate area
- Adds unnecessary complexity (model loading, inference) without clear benefits
- Requires a trained YOLO model file (.pt), which needs to be created separately
- Pairing algorithm is identical to the blob-based detector, so YOLO doesn't improve pairing quality

**When to use (if ever):**
- Only consider if you have a specific case where YOLO detects particles that standard blob detection completely misses
- Requires custom training data and model training workflow
- Not recommended for general use

**Recommended alternative:**
- Use `apps/detect_pairs.py` (standard blob detection) for most cases
- Use `apps/detect_pairs_watershed.py` for cases with overlapping/touching particles

**Files (all in `deprecated/` folder):**
- `deprecated/detect_pairs_yolo.py` - Main application (still functional, but not recommended)
- `deprecated/detect_pairs_yolo.bat` / `deprecated/detect_pairs_yolo.ps1` / `deprecated/detect_pairs_yolo.sh` - Launch scripts
- `deprecated/run_detect_pairs_yolo.sh` - Alternative launch script

---

## Migration Guide

If you're currently using `deprecated/detect_pairs_yolo.py`, consider migrating to:

1. **Standard blob detection** (`detect_pairs.py`):
   - Works for most particle tracking scenarios
   - No dependencies on ML frameworks
   - Faster and more reliable

2. **Watershed detection** (`detect_pairs_watershed.py`):
   - Better for overlapping or touching particles
   - Uses watershed segmentation to separate connected blobs
   - Same interface and pairing algorithms as standard detector

Both alternatives use the same geometric pairing algorithms and tracking system, so your workflow and results will be very similar.

