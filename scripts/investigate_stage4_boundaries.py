"""
Investigate boundary artifacts and cropping in NIST SD04 and SOCOFing datasets.
"""
import os
import csv
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def check_nist_boundaries():
    manifest_path = os.path.join(PROJECT_ROOT, "manifests", "nist_sd04_all.csv")
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    
    print(f"Total NIST records: {len(reader)}")
    
    # Sample 500 images across classes and splits for deep boundary stats
    bottom_32_means = []
    bottom_32_mins = []
    top_16_means = []
    top_32_means = []
    left_16_means = []
    right_16_means = []
    
    # We will also compute the active content bounding box using Otsu threshold / threshold on background
    # NIST paper background is high (white/light gray ~200-255), ink ridges are dark (<150).
    crop_boxes = []
    
    for idx, row in enumerate(reader):
        img_path = os.path.join(PROJECT_ROOT, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Check bottom 32 rows (480:512)
        bottom_patch = img[480:512, :]
        bottom_32_means.append(np.mean(bottom_patch))
        bottom_32_mins.append(np.min(bottom_patch))
        
        top_16_means.append(np.mean(img[0:16, :]))
        top_32_means.append(np.mean(img[0:32, :]))
        left_16_means.append(np.mean(img[:, 0:16]))
        right_16_means.append(np.mean(img[:, 496:512]))
        
        # Fingerprint foreground mask: dark pixels (ink) < 200
        # Ignore top header (row < 32) and bottom border (row >= 480)
        fg = (img < 200)
        fg_rows = np.where(fg.any(axis=1))[0]
        fg_cols = np.where(fg.any(axis=0))[0]
        if len(fg_rows) > 0 and len(fg_cols) > 0:
            crop_boxes.append((fg_rows[0], fg_rows[-1], fg_cols[0], fg_cols[-1]))

    print(f"--- NIST SD04 Boundary Statistics ({len(bottom_32_means)} images) ---")
    print(f"Bottom 32 rows (480-511): mean of means = {np.mean(bottom_32_means):.2f}, min overall = {np.min(bottom_32_mins)}")
    print(f"Bottom 32 rows is 100% white (mean == 255.0): {np.mean(np.array(bottom_32_means) == 255.0)*100:.2f}%")
    print(f"Bottom 32 rows mean >= 254.0: {np.mean(np.array(bottom_32_means) >= 254.0)*100:.2f}%")
    print(f"Top 16 rows mean: {np.mean(top_16_means):.2f} (std {np.std(top_16_means):.2f})")
    print(f"Top 32 rows mean: {np.mean(top_32_means):.2f} (std {np.std(top_32_means):.2f})")
    print(f"Left 16 cols mean: {np.mean(left_16_means):.2f}")
    print(f"Right 16 cols mean: {np.mean(right_16_means):.2f}")
    
    crop_boxes = np.array(crop_boxes)
    print("Content bounding box percentiles across all images:")
    print(f"  Row min (min y): min={np.min(crop_boxes[:, 0])}, p1={np.percentile(crop_boxes[:, 0], 1):.1f}, p5={np.percentile(crop_boxes[:, 0], 5):.1f}, median={np.median(crop_boxes[:, 0]):.1f}")
    print(f"  Row max (max y): max={np.max(crop_boxes[:, 1])}, p95={np.percentile(crop_boxes[:, 1], 95):.1f}, p99={np.percentile(crop_boxes[:, 1], 99):.1f}, median={np.median(crop_boxes[:, 1]):.1f}")
    print(f"  Col min (min x): min={np.min(crop_boxes[:, 2])}, p1={np.percentile(crop_boxes[:, 2], 1):.1f}, p5={np.percentile(crop_boxes[:, 2], 5):.1f}, median={np.median(crop_boxes[:, 2]):.1f}")
    print(f"  Col max (max x): max={np.max(crop_boxes[:, 3])}, p95={np.percentile(crop_boxes[:, 3], 95):.1f}, p99={np.percentile(crop_boxes[:, 3], 99):.1f}, median={np.median(crop_boxes[:, 3]):.1f}")

def check_socofing_boundaries():
    manifest_path = os.path.join(PROJECT_ROOT, "manifests", "socofing_eval.csv")
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    
    print(f"\nTotal SOCOFing records: {len(reader)}")
    std_imgs = [r for r in reader if int(r["width"]) == 96 and int(r["height"]) == 103]
    hi_imgs = [r for r in reader if int(r["width"]) == 241 and int(r["height"]) == 298]
    print(f"Standard (96x103): {len(std_imgs)}, High-res (241x298): {len(hi_imgs)}")
    
    # Inspect sample images of each
    for label, sample_list in [("Standard 96x103", std_imgs[:5]), ("High-res 241x298", hi_imgs[:5])]:
        print(f"\nExamining {label}:")
        for r in sample_list:
            p = os.path.join(PROJECT_ROOT, r["image_path"])
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            print(f"  {r['image_path']}: shape={img.shape}, corner values=[{img[0,0]}, {img[0,-1]}, {img[-1,0]}, {img[-1,-1]}], min={np.min(img)}, max={np.max(img)}, mean={np.mean(img):.1f}")

if __name__ == "__main__":
    check_nist_boundaries()
    check_socofing_boundaries()
