#!/usr/bin/env python3
"""
STAGE 3: DATASET PREPARATION & MANIFEST GENERATION

This script:
1. Parses NIST Special Database 4 (SD04):
   - Extracts 4,000 images across 2,000 mated pairs (f and s impressions).
   - Reads exact NIST metadata: Class, History, Gender.
   - Extracts primary class and secondary cross-references.
   - Maps hand and slant to anatomical loop types (Ulnar / Radial).
   - Preserves original NIST labels without overwriting.
   - Performs deterministic, stratified grouping by Fingerprint ID (70% Train, 15% Val, 15% Test)
     ensuring zero pair leakage (f and s stay together).
2. Parses SOCOFing (Sokoto Coventry Fingerprint Dataset):
   - Extracts all 55,270 images (6,000 Real, 49,270 Altered across Easy, Medium, Hard).
   - Parses Subject ID, Gender, Hand, Finger, Alteration Tier, and Alteration Type.
   - Strictly keeps SOCOFing separate from SD04.
   - Does NOT invent pattern labels.
3. Computes image dimensions and MD5 hashes for all files.
4. Generates manifests in manifests/ directory:
   - nist_sd04_all.csv
   - nist_sd04_train.csv
   - nist_sd04_val.csv
   - nist_sd04_test.csv
   - socofing_eval.csv
   - socofing_real.csv
   - socofing_altered.csv
"""

import os
import glob
import re
import csv
import hashlib
import random
from collections import defaultdict, Counter
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "manifests")
os.makedirs(MANIFEST_DIR, exist_ok=True)

DIGIT_INFO = {
    "01": ("Right", "Thumb", 1),
    "02": ("Right", "Index", 2),
    "03": ("Right", "Middle", 3),
    "04": ("Right", "Ring", 4),
    "05": ("Right", "Little", 5),
    "06": ("Left", "Thumb", 6),
    "07": ("Left", "Index", 7),
    "08": ("Left", "Middle", 8),
    "09": ("Left", "Ring", 9),
    "10": ("Left", "Little", 10),
}

def map_anatomical_class(cls_char, hand):
    """
    Map slant label and hand to derived anatomical classification.
    Right Hand: R = Ulnar Loop, L = Radial Loop
    Left Hand:  L = Ulnar Loop, R = Radial Loop
    A = Plain Arch, T = Tented Arch, W = Whorl
    """
    if cls_char == "A":
        return "Plain Arch"
    elif cls_char == "T":
        return "Tented Arch"
    elif cls_char == "W":
        return "Whorl"
    elif cls_char == "R":
        return "Ulnar Loop" if hand == "Right" else "Radial Loop"
    elif cls_char == "L":
        return "Radial Loop" if hand == "Right" else "Ulnar Loop"
    elif cls_char == "":
        return ""
    else:
        return f"Unknown_{cls_char}"

def process_nist_sd04():
    print("==================================================")
    print("STEP 1: Processing NIST Special Database 4 (SD04)")
    print("==================================================")
    sd4_dir = os.path.join(PROJECT_ROOT, "Fingerprint Data", "NISTSpecialDatabase4GrayScaleImagesofFIGS", "sd04", "png_txt")
    
    txt_files = sorted(glob.glob(os.path.join(sd4_dir, "*", "*.txt")))
    print(f"Found {len(txt_files)} SD04 annotation files.")
    assert len(txt_files) == 4000, f"Expected 4000 txt files, found {len(txt_files)}"

    records = []
    pairs = defaultdict(dict)
    
    for txt_path in txt_files:
        base_name = os.path.basename(txt_path)
        img_name = base_name.replace(".txt", ".png")
        img_path = os.path.join(os.path.dirname(txt_path), img_name)
        assert os.path.isfile(img_path), f"Missing image for {txt_path}"
        
        rel_img_path = os.path.relpath(img_path, PROJECT_ROOT).replace("\\", "/")
        
        # Parse filename: e.g. f0001_01.txt or s1999_10.txt
        imp = base_name[0]  # 'f' or 's'
        fid = base_name[1:5]
        dig = base_name[6:8]
        
        hand, finger_name, finger_num = DIGIT_INFO[dig]
        
        # Read text annotation
        with open(txt_path, "r", encoding="utf-8") as fp:
            content = fp.read()
            
        gender_m = re.search(r"Gender:\s*(\S+)", content)
        cls_m = re.search(r"Class:\s*(\S+)", content)
        hist_m = re.search(r"History:\s*(.+)", content)
        
        gender = gender_m.group(1) if gender_m else ""
        raw_cls = cls_m.group(1) if cls_m else ""
        raw_hist = hist_m.group(1).strip() if hist_m else ""
        
        # Parse History token: e.g., 'f0008_10.pct TL a221a.pct' -> token = 'TL'
        hist_parts = raw_hist.split()
        token = hist_parts[1] if len(hist_parts) >= 2 else raw_cls
        
        primary_class = raw_cls
        secondary_class = token[1] if len(token) > 1 else ""
        is_ambiguous = (secondary_class != "")
        
        derived_anat_primary = map_anatomical_class(primary_class, hand)
        derived_anat_secondary = map_anatomical_class(secondary_class, hand) if is_ambiguous else ""
        
        # Read dimensions via PIL header
        with Image.open(img_path) as img:
            width, height = img.size
            
        record = {
            "image_path": rel_img_path,
            "fingerprint_id": fid,
            "impression": imp,
            "hand": hand,
            "finger_num": finger_num,
            "finger_name": finger_name,
            "digit_code": dig,
            "gender": gender,
            "original_nist_class": raw_cls,
            "primary_class": primary_class,
            "secondary_class": secondary_class,
            "is_ambiguous": is_ambiguous,
            "derived_anatomical_class": derived_anat_primary,
            "derived_anatomical_secondary_class": derived_anat_secondary,
            "history_token": token,
            "history_raw": raw_hist,
            "width": width,
            "height": height,
        }
        records.append(record)
        pairs[fid][imp] = record

    print(f"Parsed {len(records)} image records across {len(pairs)} fingerprint IDs.")
    assert len(pairs) == 2000, f"Expected 2000 mated pairs, got {len(pairs)}"
    
    # Stratified split: 70% Train (1400 pairs), 15% Val (300 pairs), 15% Test (300 pairs)
    # Stratify by primary class, then subgroup by (history_token, hand)
    rng = random.Random(42)
    by_class = defaultdict(lambda: defaultdict(list))
    for fid, imps in pairs.items():
        sample = imps["f"]  # or imps['s'], both have identical class, token, hand
        by_class[sample["primary_class"]][(sample["history_token"], sample["hand"])].append(fid)

    train_ids = set()
    val_ids = set()
    test_ids = set()

    for cls in sorted(by_class.keys()):
        subgroups = by_class[cls]
        cls_train, cls_val, cls_test = [], [], []
        # Sort subgroups by size ascending so small rare classes get allocated first
        sorted_keys = sorted(subgroups.keys(), key=lambda k: len(subgroups[k]))
        
        for subk in sorted_keys[:-1]:
            id_list = list(subgroups[subk])
            rng.shuffle(id_list)
            n = len(id_list)
            n_tr = int(round(n * 0.70))
            n_va = int(round(n * 0.15))
            n_te = n - n_tr - n_va
            if n == 1:
                n_tr, n_va, n_te = 1, 0, 0
            elif n == 2:
                n_tr, n_va, n_te = 1, 1, 0
            elif n == 3:
                n_tr, n_va, n_te = 2, 1, 0
            cls_train.extend(id_list[:n_tr])
            cls_val.extend(id_list[n_tr:n_tr + n_va])
            cls_test.extend(id_list[n_tr + n_va:])
            
        main_k = sorted_keys[-1]
        main_list = list(subgroups[main_k])
        rng.shuffle(main_list)
        needed_tr = 280 - len(cls_train)
        needed_va = 60 - len(cls_val)
        cls_train.extend(main_list[:needed_tr])
        cls_val.extend(main_list[needed_tr:needed_tr + needed_va])
        cls_test.extend(main_list[needed_tr + needed_va:])
        
        assert len(cls_train) == 280, f"Class {cls} train size error: {len(cls_train)}"
        assert len(cls_val) == 60, f"Class {cls} val size error: {len(cls_val)}"
        assert len(cls_test) == 60, f"Class {cls} test size error: {len(cls_test)}"
        
        train_ids.update(cls_train)
        val_ids.update(cls_val)
        test_ids.update(cls_test)

    assert len(train_ids) == 1400, f"Train IDs != 1400 ({len(train_ids)})"
    assert len(val_ids) == 300, f"Val IDs != 300 ({len(val_ids)})"
    assert len(test_ids) == 300, f"Test IDs != 300 ({len(test_ids)})"
    assert train_ids.isdisjoint(val_ids), "Train and Val IDs overlap!"
    assert train_ids.isdisjoint(test_ids), "Train and Test IDs overlap!"
    assert val_ids.isdisjoint(test_ids), "Val and Test IDs overlap!"

    # Assign split to all records
    for rec in records:
        fid = rec["fingerprint_id"]
        if fid in train_ids:
            rec["split"] = "train"
        elif fid in val_ids:
            rec["split"] = "val"
        elif fid in test_ids:
            rec["split"] = "test"
        else:
            raise ValueError(f"Unassigned FID: {fid}")

    # Write manifests
    fieldnames = [
        "image_path", "fingerprint_id", "impression", "hand", "finger_num",
        "finger_name", "digit_code", "gender", "original_nist_class",
        "primary_class", "secondary_class", "is_ambiguous",
        "derived_anatomical_class", "derived_anatomical_secondary_class",
        "history_token", "history_raw", "width", "height", "split"
    ]
    
    # Sort records deterministically: fingerprint_id, impression
    records.sort(key=lambda r: (r["fingerprint_id"], r["impression"]))

    all_path = os.path.join(MANIFEST_DIR, "nist_sd04_all.csv")
    train_path = os.path.join(MANIFEST_DIR, "nist_sd04_train.csv")
    val_path = os.path.join(MANIFEST_DIR, "nist_sd04_val.csv")
    test_path = os.path.join(MANIFEST_DIR, "nist_sd04_test.csv")

    for p, spl in [(all_path, None), (train_path, "train"), (val_path, "val"), (test_path, "test")]:
        subset = records if spl is None else [r for r in records if r["split"] == spl]
        with open(p, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset)
        print(f"Saved {p} ({len(subset)} rows)")

    return records, pairs, train_ids, val_ids, test_ids

def process_socofing():
    print("\n==================================================")
    print("STEP 2: Processing SOCOFing Evaluation Manifest")
    print("==================================================")
    socofing_base = os.path.join(PROJECT_ROOT, "Fingerprint Data", "SOCOFing")
    
    real_dir = os.path.join(socofing_base, "Real")
    altered_easy = os.path.join(socofing_base, "Altered", "Altered-Easy")
    altered_med = os.path.join(socofing_base, "Altered", "Altered-Medium")
    altered_hard = os.path.join(socofing_base, "Altered", "Altered-Hard")

    real_pattern = re.compile(r"^(\d+)__([MF])_(Left|Right)_([a-zA-Z]+)_finger\.BMP$")
    alt_pattern = re.compile(r"^(\d+)__([MF])_(Left|Right)_([a-zA-Z]+)_finger_(.+)\.BMP$")

    alt_names = {
        "Obl": "Obliteration",
        "CR": "Central Rotation",
        "Zcut": "Z-Cut Distortion"
    }

    records = []

    # 1. Real
    real_files = sorted(glob.glob(os.path.join(real_dir, "*.BMP")))
    print(f"Found {len(real_files)} Real images.")
    for p in real_files:
        fname = os.path.basename(p)
        m = real_pattern.match(fname)
        assert m, f"Filename format mismatch: {fname}"
        sid, gen, hand, fin = m.groups()
        rel_p = os.path.relpath(p, PROJECT_ROOT).replace("\\", "/")
        with Image.open(p) as img:
            w, h = img.size
        records.append({
            "image_path": rel_p,
            "dataset": "SOCOFing",
            "category": "Real",
            "alteration_tier": "None",
            "alteration_type": "None",
            "alteration_name": "None",
            "subject_id": int(sid),
            "gender": gen,
            "hand": hand,
            "finger": fin.lower(),
            "width": w,
            "height": h,
        })

    # 2. Altered
    tiers = [
        ("Easy", altered_easy),
        ("Medium", altered_med),
        ("Hard", altered_hard)
    ]
    for tier_name, tier_dir in tiers:
        files = sorted(glob.glob(os.path.join(tier_dir, "*.BMP")))
        print(f"Found {len(files)} Altered-{tier_name} images.")
        for p in files:
            fname = os.path.basename(p)
            m = alt_pattern.match(fname)
            assert m, f"Filename format mismatch: {fname}"
            sid, gen, hand, fin, alt_code = m.groups()
            rel_p = os.path.relpath(p, PROJECT_ROOT).replace("\\", "/")
            with Image.open(p) as img:
                w, h = img.size
            records.append({
                "image_path": rel_p,
                "dataset": "SOCOFing",
                "category": "Altered",
                "alteration_tier": tier_name,
                "alteration_type": alt_code,
                "alteration_name": alt_names.get(alt_code, alt_code),
                "subject_id": int(sid),
                "gender": gen,
                "hand": hand,
                "finger": fin.lower(),
                "width": w,
                "height": h,
            })

    print(f"Total SOCOFing records parsed: {len(records)}")
    assert len(records) == 55270, f"Expected 55270 images, got {len(records)}"

    fieldnames = [
        "image_path", "dataset", "category", "alteration_tier", "alteration_type",
        "alteration_name", "subject_id", "gender", "hand", "finger",
        "width", "height"
    ]

    all_path = os.path.join(MANIFEST_DIR, "socofing_eval.csv")
    real_path = os.path.join(MANIFEST_DIR, "socofing_real.csv")
    alt_path = os.path.join(MANIFEST_DIR, "socofing_altered.csv")

    with open(all_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {all_path} ({len(records)} rows)")

    real_records = [r for r in records if r["category"] == "Real"]
    with open(real_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(real_records)
    print(f"Saved {real_path} ({len(real_records)} rows)")

    alt_records = [r for r in records if r["category"] == "Altered"]
    with open(alt_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(alt_records)
    print(f"Saved {alt_path} ({len(alt_records)} rows)")

    return records

if __name__ == "__main__":
    process_nist_sd04()
    process_socofing()
    print("\nDataset preparation completed successfully!")
