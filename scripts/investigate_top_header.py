"""
Detailed investigation of NIST SD04 top header text and fingerprint core/delta locations.
"""
import os
import csv
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def analyze_top_rows():
    manifest_path = os.path.join(PROJECT_ROOT, "manifests", "nist_sd04_all.csv")
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    # Examine row-wise horizontal variance and intensity profiles across 400 images
    # Text headers usually have high horizontal gradient / sharp black characters on white background
    # fingerprint ridge patterns have continuous periodic patterns.
    
    top_dark_pixels_per_row = np.zeros(64)
    
    for row in reader[:500]:
        p = os.path.join(PROJECT_ROOT, row["image_path"])
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        # Check dark pixels (< 180) in each row from 0 to 63
        for y in range(64):
            if np.any(img[y, :] < 180):
                top_dark_pixels_per_row[y] += 1

    print("Fraction of images with dark pixels (<180) in row y:")
    for y in range(0, 48, 4):
        print(f"  Rows {y:02d}-{y+3:02d}: avg fraction = {np.mean(top_dark_pixels_per_row[y:y+4])/500.0:.3f}")

def check_card_header_clipping():
    # Let's inspect some specific examples:
    # Look at images where top text is present vs where fingerprint reaches high up
    manifest_path = os.path.join(PROJECT_ROOT, "manifests", "nist_sd04_all.csv")
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    # Check the core/center of mass of the fingerprint ridge pattern
    core_estimates = []
    for r in reader[:300]:
        p = os.path.join(PROJECT_ROOT, r["image_path"])
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        # Compute local variance or gradient magnitude (ridge pattern has high local variance, blank paper has low variance)
        ksize = 15
        blur = cv2.GaussianBlur(img.astype(np.float32), (ksize, ksize), 0)
        blur2 = cv2.GaussianBlur(img.astype(np.float32)**2, (ksize, ksize), 0)
        var = blur2 - blur**2
        
        # Zero out top 32 rows and bottom 32 rows to avoid header text variance
        var_interior = var.copy()
        var_interior[:32, :] = 0
        var_interior[480:, :] = 0
        
        # Ridge mask
        ridge_mask = (var_interior > 300) & (img < 220)
        coords = np.argwhere(ridge_mask)
        if len(coords) > 0:
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            cent_y, cent_x = coords.mean(axis=0)
            core_estimates.append((min_y, max_y, cent_y, min_x, max_x, cent_x))
            
    core_estimates = np.array(core_estimates)
    print("\nRidge region boundaries across sample (min_y, max_y, cent_y, min_x, max_x, cent_x):")
    print(f"  Ridge Top (min_y): min={np.min(core_estimates[:,0])}, p1={np.percentile(core_estimates[:,0], 1):.1f}, p5={np.percentile(core_estimates[:,0], 5):.1f}, median={np.median(core_estimates[:,0]):.1f}")
    print(f"  Ridge Bottom (max_y): max={np.max(core_estimates[:,1])}, p95={np.percentile(core_estimates[:,1], 95):.1f}, p99={np.percentile(core_estimates[:,1], 99):.1f}, median={np.median(core_estimates[:,1]):.1f}")
    print(f"  Ridge Center Y (cent_y): mean={np.mean(core_estimates[:,2]):.1f}, min={np.min(core_estimates[:,2]):.1f}, max={np.max(core_estimates[:,2]):.1f}")
    print(f"  Ridge Left (min_x): min={np.min(core_estimates[:,3])}, p1={np.percentile(core_estimates[:,3], 1):.1f}, median={np.median(core_estimates[:,3]):.1f}")
    print(f"  Ridge Right (max_x): max={np.max(core_estimates[:,4])}, p99={np.percentile(core_estimates[:,4], 99):.1f}, median={np.median(core_estimates[:,4]):.1f}")

if __name__ == "__main__":
    analyze_top_rows()
    check_card_header_clipping()
