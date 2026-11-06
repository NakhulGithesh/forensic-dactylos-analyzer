#!/usr/bin/env python3
"""
STAGE 3 VALIDATION & INTEGRITY SUITE

Thoroughly verifies:
1. Exact row counts and image existence.
2. Complete absence of split leakage (ID overlap, mated pair separation, hash collisions).
3. Exact class balance in Train, Val, Test.
4. Correctness of anatomical loop mapping (Slant + Hand -> Ulnar / Radial).
5. Preservation of NIST original labels (never overwritten).
6. Cross-reference parsing from History field.
7. SOCOFing integrity (metadata consistency, no invented pattern labels).
8. Missing annotation audit.
9. Duplicate image audit across all datasets.
"""

import os
import csv
import glob
import hashlib
from collections import defaultdict, Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "manifests")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_csv(rel_path):
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    assert os.path.isfile(full_path), f"File not found: {full_path}"
    with open(full_path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        return list(reader)

def run_validations():
    print("==================================================")
    print("RUNNING STAGE 3 VALIDATION CHECKS")
    print("==================================================")

    # 1. Load Manifests
    all_sd4 = load_csv("manifests/nist_sd04_all.csv")
    train_sd4 = load_csv("manifests/nist_sd04_train.csv")
    val_sd4 = load_csv("manifests/nist_sd04_val.csv")
    test_sd4 = load_csv("manifests/nist_sd04_test.csv")

    socofing_all = load_csv("manifests/socofing_eval.csv")
    socofing_real = load_csv("manifests/socofing_real.csv")
    socofing_alt = load_csv("manifests/socofing_altered.csv")

    print(f"Loaded NIST SD04 manifests: Total={len(all_sd4)}, Train={len(train_sd4)}, Val={len(val_sd4)}, Test={len(test_sd4)}")
    print(f"Loaded SOCOFing manifests: Total={len(socofing_all)}, Real={len(socofing_real)}, Altered={len(socofing_alt)}")

    assert len(all_sd4) == 4000, f"Expected 4000 in SD04 all, got {len(all_sd4)}"
    assert len(train_sd4) == 2800, f"Expected 2800 in SD04 train, got {len(train_sd4)}"
    assert len(val_sd4) == 600, f"Expected 600 in SD04 val, got {len(val_sd4)}"
    assert len(test_sd4) == 600, f"Expected 600 in SD04 test, got {len(test_sd4)}"
    assert len(train_sd4) + len(val_sd4) + len(test_sd4) == len(all_sd4), "Split counts do not sum to total"

    assert len(socofing_all) == 55270, f"Expected 55270 in SOCOFing all, got {len(socofing_all)}"
    assert len(socofing_real) == 6000, f"Expected 6000 in SOCOFing Real, got {len(socofing_real)}"
    assert len(socofing_alt) == 49270, f"Expected 49270 in SOCOFing Altered, got {len(socofing_alt)}"
    assert len(socofing_real) + len(socofing_alt) == len(socofing_all), "SOCOFing subset counts do not sum to total"

    # 2. File Existence & Image Dimensions Check
    print("\n[Check 1/10] Verifying 100% of image files exist on disk with valid dimensions...")
    for idx, row in enumerate(all_sd4):
        p = os.path.join(PROJECT_ROOT, row["image_path"])
        assert os.path.isfile(p), f"Missing SD04 image: {p}"
        assert int(row["width"]) == 512 and int(row["height"]) == 512, f"Invalid dimensions in SD04: {row}"
    print("  -> All 4,000 NIST SD04 images exist on disk with dimensions 512x512.")

    # Check 100% of SOCOFing images for existence and dimensions
    real_96_103 = 0
    real_241_298 = 0
    alt_96_103 = 0
    alt_241_298 = 0
    odd_real_fingers = set()
    odd_alt_fingers = set()

    for row in socofing_all:
        p = os.path.join(PROJECT_ROOT, row["image_path"])
        assert os.path.isfile(p), f"Missing SOCOFing image: {p}"
        w, h = int(row["width"]), int(row["height"])
        cat = row["category"]
        if (w, h) == (96, 103):
            if cat == "Real":
                real_96_103 += 1
            else:
                alt_96_103 += 1
        elif (w, h) == (241, 298):
            finger_key = (row["subject_id"], row["hand"], row["finger"])
            if cat == "Real":
                real_241_298 += 1
                odd_real_fingers.add(finger_key)
            else:
                alt_241_298 += 1
                odd_alt_fingers.add(finger_key)
        else:
            raise AssertionError(f"Unexpected SOCOFing dimension ({w}x{h}) in {row['image_path']}")

    assert real_96_103 == 5956 and real_241_298 == 44, f"Real dims mismatch: 96x103={real_96_103}, 241x298={real_241_298}"
    assert alt_96_103 == 48881 and alt_241_298 == 389, f"Altered dims mismatch: 96x103={alt_96_103}, 241x298={alt_241_298}"
    assert odd_alt_fingers.issubset(odd_real_fingers), "Altered 241x298 images do not map back to Real 241x298 fingers!"
    print(f"  -> All 55,270 SOCOFing images exist on disk (54,837 standard 96x103, 433 high-res 241x298: 44 Real, 389 Altered).")

    # 3. Leakage Checks: Fingerprint ID Overlap & Pair Integrity
    print("\n[Check 2/10] Verifying zero leakage between splits...")
    train_ids = set(r["fingerprint_id"] for r in train_sd4)
    val_ids = set(r["fingerprint_id"] for r in val_sd4)
    test_ids = set(r["fingerprint_id"] for r in test_sd4)

    assert len(train_ids) == 1400, f"Train IDs != 1400 ({len(train_ids)})"
    assert len(val_ids) == 300, f"Val IDs != 300 ({len(val_ids)})"
    assert len(test_ids) == 300, f"Test IDs != 300 ({len(test_ids)})"

    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    assert len(train_val_overlap) == 0, f"Train-Val ID overlap: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Train-Test ID overlap: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Val-Test ID overlap: {val_test_overlap}"
    print("  -> Zero Fingerprint ID overlap across splits: Train & Val = 0, Train & Test = 0, Val & Test = 0.")

    # Check mated pair leakage: both 'f' and 's' must be in the same split
    pairs = defaultdict(dict)
    for r in all_sd4:
        pairs[r["fingerprint_id"]][r["impression"]] = r

    pair_leakage = []
    for fid, imps in pairs.items():
        assert "f" in imps and "s" in imps, f"Incomplete pair for FID {fid}"
        if imps["f"]["split"] != imps["s"]["split"]:
            pair_leakage.append((fid, imps["f"]["split"], imps["s"]["split"]))

    assert len(pair_leakage) == 0, f"Found pair leakage: {pair_leakage}"
    print("  -> Zero mated-pair leakage: all 2,000 (f, s) pairs strictly reside in the same split.")

    # 4. Class Distribution & Stratification Verification
    print("\n[Check 3/10] Verifying class balance and stratification...")
    split_classes = defaultdict(Counter)
    for r in all_sd4:
        split_classes[r["split"]][r["primary_class"]] += 1

    expected_per_class = {"train": 560, "val": 120, "test": 120}
    for spl, target in expected_per_class.items():
        for cls in ["A", "T", "W", "L", "R"]:
            actual = split_classes[spl][cls]
            assert actual == target, f"Class {cls} in split {spl}: expected {target}, got {actual}"
    print("  -> Perfect NIST class balance verified: exactly 560 Train, 120 Val, 120 Test for all 5 classes (A, T, W, L, R).")

    # 5. Anatomical Classification & Hand Consistency
    print("\n[Check 4/10] Verifying anatomical loop orientation mapping...")
    for r in all_sd4:
        h = r["hand"]
        c = r["original_nist_class"]
        anat = r["derived_anatomical_class"]
        if c == "A":
            assert anat == "Plain Arch"
        elif c == "T":
            assert anat == "Tented Arch"
        elif c == "W":
            assert anat == "Whorl"
        elif c == "R":
            assert anat == ("Ulnar Loop" if h == "Right" else "Radial Loop")
        elif c == "L":
            assert anat == ("Radial Loop" if h == "Right" else "Ulnar Loop")
        else:
            raise ValueError(f"Unknown class {c}")
    print("  -> 100% adherence to anatomical slant mapping verified (Right+R = Ulnar, Right+L = Radial, Left+L = Ulnar, Left+R = Radial).")

    # 6. Anatomical Class Representation Across Splits
    print("\n[Check 5/10] Verifying anatomical class representation in all splits...")
    split_anat = defaultdict(Counter)
    for r in all_sd4:
        split_anat[r["split"]][r["derived_anatomical_class"]] += 1

    expected_anat = {
        "train": {"Plain Arch": 560, "Tented Arch": 560, "Whorl": 560, "Ulnar Loop": 1076, "Radial Loop": 44},
        "val": {"Plain Arch": 120, "Tented Arch": 120, "Whorl": 120, "Ulnar Loop": 230, "Radial Loop": 10},
        "test": {"Plain Arch": 120, "Tented Arch": 120, "Whorl": 120, "Ulnar Loop": 232, "Radial Loop": 8},
    }
    for spl, expected_counts in expected_anat.items():
        for anat_cls, exp_c in expected_counts.items():
            act_c = split_anat[spl][anat_cls]
            assert act_c == exp_c, f"Anatomical class {anat_cls} in {spl}: expected {exp_c}, got {act_c}"
    print("  -> All 5 derived anatomical classes represented across splits (including rare Radial Loops: 44 Train, 10 Val, 8 Test).")

    # 7. NIST Label Preservation Check & Unambiguous Formatting
    print("\n[Check 6/10] Verifying NIST label preservation (no overwrite)...")
    for r in all_sd4:
        assert r["original_nist_class"] in {"A", "T", "W", "L", "R"}
        assert r["primary_class"] == r["original_nist_class"]
        assert r["derived_anatomical_class"] in {"Plain Arch", "Tented Arch", "Whorl", "Ulnar Loop", "Radial Loop"}
        if r["is_ambiguous"] == "False":
            assert r["secondary_class"] == ""
            assert r["derived_anatomical_secondary_class"] == ""
            assert r["history_token"] == r["primary_class"]
    print("  -> Original NIST classes strictly preserved and decoupled from derived anatomical labels.")

    # 8. Secondary Cross-References Parsing Check
    print("\n[Check 7/10] Verifying secondary cross-reference parsing...")
    ambiguous_count = sum(1 for r in all_sd4 if r["is_ambiguous"] == "True")
    unambiguous_count = sum(1 for r in all_sd4 if r["is_ambiguous"] == "False")
    assert ambiguous_count == 700, f"Expected 700 ambiguous images, got {ambiguous_count}"
    assert unambiguous_count == 3300, f"Expected 3300 unambiguous images, got {unambiguous_count}"
    
    # Check secondary class values
    for r in all_sd4:
        if r["is_ambiguous"] == "True":
            assert r["secondary_class"] in {"A", "T", "W", "L", "R"}
            assert len(r["history_token"]) == 2
            assert r["history_token"][0] == r["primary_class"]
            assert r["history_token"][1] == r["secondary_class"]
            assert r["derived_anatomical_secondary_class"] in {
                "Plain Arch", "Tented Arch", "Whorl", "Ulnar Loop", "Radial Loop"
            }
    print(f"  -> Secondary cross-references parsed accurately: 700 ambiguous images (350 pairs), 3,300 single-class images.")

    # 9. Missing Annotation Audit
    print("\n[Check 8/10] Auditing for missing annotations...")
    missing_sd4 = []
    for r in all_sd4:
        for field in ["image_path", "fingerprint_id", "impression", "hand", "finger_num",
                      "original_nist_class", "primary_class", "derived_anatomical_class",
                      "history_token", "gender", "width", "height", "split"]:
            if not r.get(field):
                missing_sd4.append((r["image_path"], field))
    assert len(missing_sd4) == 0, f"Missing fields in SD04: {missing_sd4}"

    missing_socofing = []
    for r in socofing_all:
        for field in ["image_path", "dataset", "category", "alteration_tier", "alteration_type",
                      "subject_id", "gender", "hand", "finger", "width", "height"]:
            if r.get(field) is None or r.get(field) == "":
                missing_socofing.append((r["image_path"], field))
    assert len(missing_socofing) == 0, f"Missing fields in SOCOFing: {missing_socofing}"
    print("  -> Zero missing fields in any required columns across NIST SD04 and SOCOFing.")

    # 10. Duplicate Image Audit
    print("\n[Check 9/10] Auditing duplicate images across datasets...")
    sd4_hashes = {}
    sd4_dups = []
    for r in all_sd4:
        p = os.path.join(PROJECT_ROOT, r["image_path"])
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        if h in sd4_hashes:
            sd4_dups.append((r["image_path"], sd4_hashes[h]))
        else:
            sd4_hashes[h] = r["image_path"]
    assert len(sd4_dups) == 0, f"Unexpected duplicates in SD04: {sd4_dups}"
    print("  -> NIST SD04: 0 duplicate images across all 4,000 files.")

    # SOCOFing duplicates audit
    socofing_real_hashes = defaultdict(list)
    for r in socofing_real:
        p = os.path.join(PROJECT_ROOT, r["image_path"])
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        socofing_real_hashes[h].append(r["image_path"])
    
    real_dup_hashes = {h: paths for h, paths in socofing_real_hashes.items() if len(paths) > 1}
    redundant_real_dups = sum(len(paths) - 1 for paths in real_dup_hashes.values())
    total_real_dup_files = sum(len(paths) for paths in real_dup_hashes.values())
    assert len(real_dup_hashes) == 11, f"Expected 11 duplicate hashes in Real, got {len(real_dup_hashes)}"
    assert redundant_real_dups == 15, f"Expected 15 redundant duplicate files in Real, got {redundant_real_dups}"
    assert total_real_dup_files == 26, f"Expected 26 files involved in duplicates in Real, got {total_real_dup_files}"
    print(f"  -> SOCOFing Real: Found 11 duplicate hashes across 26 files (15 redundant duplicate files, documented).")

    # SOCOFing altered duplicates audit
    socofing_alt_hashes = set()
    alt_dup_count = 0
    for r in socofing_alt:
        p = os.path.join(PROJECT_ROOT, r["image_path"])
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        if h in socofing_alt_hashes:
            alt_dup_count += 1
        socofing_alt_hashes.add(h)
    assert alt_dup_count == 0, f"Found {alt_dup_count} duplicate hashes in Altered!"
    print(f"  -> SOCOFing Altered: 0 duplicate hashes across all 49,270 files.")

    # Cross contamination between Real and Altered
    cross_real_alt = set(socofing_real_hashes.keys()).intersection(socofing_alt_hashes)
    assert len(cross_real_alt) == 0, f"Cross-contamination between Real and Altered: {len(cross_real_alt)}"
    print(f"  -> Zero cross-contamination between SOCOFing Real and Altered images.")

    # Cross contamination between SD04 and SOCOFing
    cross_sd4_soc = set(sd4_hashes.keys()).intersection(set(socofing_real_hashes.keys()).union(socofing_alt_hashes))
    assert len(cross_sd4_soc) == 0, f"Cross-contamination between SD04 and SOCOFing: {len(cross_sd4_soc)}"
    print(f"  -> Zero cross-contamination between NIST SD04 and SOCOFing images.")

    # 11. Subset Partition Integrity Check
    print("\n[Check 10/10] Auditing manifest partition integrity...")
    assert len(train_sd4) + len(val_sd4) + len(test_sd4) == len(all_sd4)
    sd4_train_val_test = set(r["image_path"] for r in train_sd4) | set(r["image_path"] for r in val_sd4) | set(r["image_path"] for r in test_sd4)
    assert sd4_train_val_test == set(r["image_path"] for r in all_sd4), "SD04 subset paths do not match all.csv"

    assert len(socofing_real) + len(socofing_alt) == len(socofing_all)
    soc_real_alt = set(r["image_path"] for r in socofing_real) | set(r["image_path"] for r in socofing_alt)
    assert soc_real_alt == set(r["image_path"] for r in socofing_all), "SOCOFing subset paths do not match eval.csv"
    print("  -> Manifest partitions are exact and comprehensive.")

    print("\n==================================================")
    print("ALL 10 VALIDATION CHECKS PASSED WITH ZERO CRITICAL DEFECTS")
    print("==================================================")
    return True

if __name__ == "__main__":
    run_validations()
