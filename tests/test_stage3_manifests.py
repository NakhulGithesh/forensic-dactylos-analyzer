import os
import csv
import hashlib
import pytest
from collections import defaultdict, Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_manifest(name):
    path = os.path.join(PROJECT_ROOT, "manifests", name)
    assert os.path.isfile(path), f"Manifest missing: {path}"
    with open(path, "r", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))

@pytest.fixture(scope="module")
def sd4_all():
    return load_manifest("nist_sd04_all.csv")

@pytest.fixture(scope="module")
def sd4_train():
    return load_manifest("nist_sd04_train.csv")

@pytest.fixture(scope="module")
def sd4_val():
    return load_manifest("nist_sd04_val.csv")

@pytest.fixture(scope="module")
def sd4_test():
    return load_manifest("nist_sd04_test.csv")

@pytest.fixture(scope="module")
def socofing_eval():
    return load_manifest("socofing_eval.csv")

@pytest.fixture(scope="module")
def socofing_real():
    return load_manifest("socofing_real.csv")

@pytest.fixture(scope="module")
def socofing_alt():
    return load_manifest("socofing_altered.csv")

def test_nist_sd04_manifest_counts_and_splits(sd4_all, sd4_train, sd4_val, sd4_test):
    assert len(sd4_all) == 4000
    assert len(sd4_train) == 2800  # 70% of 4000
    assert len(sd4_val) == 600    # 15% of 4000
    assert len(sd4_test) == 600   # 15% of 4000
    assert len(sd4_train) + len(sd4_val) + len(sd4_test) == len(sd4_all)
    assert {r["image_path"] for r in sd4_train} | {r["image_path"] for r in sd4_val} | {r["image_path"] for r in sd4_test} == {r["image_path"] for r in sd4_all}

def test_zero_pair_leakage_and_id_disjointness(sd4_all, sd4_train, sd4_val, sd4_test):
    train_ids = {r["fingerprint_id"] for r in sd4_train}
    val_ids = {r["fingerprint_id"] for r in sd4_val}
    test_ids = {r["fingerprint_id"] for r in sd4_test}

    assert len(train_ids) == 1400
    assert len(val_ids) == 300
    assert len(test_ids) == 300

    # Disjoint ID sets
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # Mated pair integrity
    pairs = defaultdict(dict)
    for r in sd4_all:
        pairs[r["fingerprint_id"]][r["impression"]] = r

    assert len(pairs) == 2000
    for fid, imps in pairs.items():
        assert "f" in imps and "s" in imps
        assert imps["f"]["split"] == imps["s"]["split"], f"Pair leakage in FID {fid}"
        assert imps["f"]["primary_class"] == imps["s"]["primary_class"]
        assert imps["f"]["secondary_class"] == imps["s"]["secondary_class"]
        assert imps["f"]["finger_num"] == imps["s"]["finger_num"]
        assert imps["f"]["hand"] == imps["s"]["hand"]

def test_nist_class_balance_across_splits(sd4_train, sd4_val, sd4_test):
    for split_name, dataset, expected_count in [
        ("train", sd4_train, 560),
        ("val", sd4_val, 120),
        ("test", sd4_test, 120),
    ]:
        counts = Counter(r["primary_class"] for r in dataset)
        for cls in ["A", "T", "W", "L", "R"]:
            assert counts[cls] == expected_count, (
                f"Split {split_name} class {cls} count mismatch: got {counts[cls]}, expected {expected_count}"
            )

def test_anatomical_loop_mapping_rules(sd4_all):
    for r in sd4_all:
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

def test_anatomical_class_split_representation(sd4_train, sd4_val, sd4_test):
    expected_anat = {
        "train": {"Plain Arch": 560, "Tented Arch": 560, "Whorl": 560, "Ulnar Loop": 1076, "Radial Loop": 44},
        "val": {"Plain Arch": 120, "Tented Arch": 120, "Whorl": 120, "Ulnar Loop": 230, "Radial Loop": 10},
        "test": {"Plain Arch": 120, "Tented Arch": 120, "Whorl": 120, "Ulnar Loop": 232, "Radial Loop": 8},
    }
    for spl_name, dataset in [("train", sd4_train), ("val", sd4_val), ("test", sd4_test)]:
        counts = Counter(r["derived_anatomical_class"] for r in dataset)
        assert dict(counts) == expected_anat[spl_name]

def test_original_labels_preserved_not_overwritten(sd4_all):
    for r in sd4_all:
        # Original NIST class remains pure 1-letter code
        assert r["original_nist_class"] in {"A", "T", "W", "L", "R"}
        # Never overwritten by derived labels
        assert r["original_nist_class"] != r["derived_anatomical_class"]

def test_secondary_history_parsing(sd4_all):
    ambiguous = [r for r in sd4_all if r["is_ambiguous"] == "True"]
    assert len(ambiguous) == 700  # 350 pairs
    for r in ambiguous:
        assert r["secondary_class"] in {"A", "T", "W", "L", "R"}
        assert r["derived_anatomical_secondary_class"] in {
            "Plain Arch", "Tented Arch", "Whorl", "Ulnar Loop", "Radial Loop"
        }
        assert r["history_token"] == r["primary_class"] + r["secondary_class"]

def test_unambiguous_records_format(sd4_all):
    unambiguous = [r for r in sd4_all if r["is_ambiguous"] == "False"]
    assert len(unambiguous) == 3300
    for r in unambiguous:
        assert r["secondary_class"] == ""
        assert r["derived_anatomical_secondary_class"] == ""
        assert r["history_token"] == r["primary_class"]

def test_no_missing_fields_sd04(sd4_all):
    required = [
        "image_path", "fingerprint_id", "impression", "hand", "finger_num",
        "finger_name", "digit_code", "gender", "original_nist_class",
        "primary_class", "derived_anatomical_class", "history_token",
        "history_raw", "width", "height", "split"
    ]
    for r in sd4_all:
        for field in required:
            assert r.get(field) not in (None, ""), f"Missing field {field} in {r['image_path']}"

def test_socofing_manifest_counts_and_partition(socofing_eval, socofing_real, socofing_alt):
    assert len(socofing_eval) == 55270
    assert len(socofing_real) == 6000
    assert len(socofing_alt) == 49270
    assert len(socofing_real) + len(socofing_alt) == len(socofing_eval)
    
    real_paths = {r["image_path"] for r in socofing_real}
    alt_paths = {r["image_path"] for r in socofing_alt}
    assert real_paths.isdisjoint(alt_paths)
    assert real_paths | alt_paths == {r["image_path"] for r in socofing_eval}

def test_all_images_exist_on_disk(sd4_all, socofing_eval):
    for r in sd4_all:
        assert os.path.isfile(os.path.join(PROJECT_ROOT, r["image_path"]))
    for r in socofing_eval:
        assert os.path.isfile(os.path.join(PROJECT_ROOT, r["image_path"]))

def test_image_dimensions_and_resolutions(sd4_all, socofing_eval):
    # SD04 is strictly 512x512
    for r in sd4_all:
        assert int(r["width"]) == 512 and int(r["height"]) == 512

    # SOCOFing has 54,837 standard images (96x103) and 433 high-res images (241x298)
    real_odd = [r for r in socofing_eval if r["category"] == "Real" and (int(r["width"]), int(r["height"])) == (241, 298)]
    alt_odd = [r for r in socofing_eval if r["category"] == "Altered" and (int(r["width"]), int(r["height"])) == (241, 298)]
    assert len(real_odd) == 44
    assert len(alt_odd) == 389
    
    # All other images are 96x103
    std = [r for r in socofing_eval if (int(r["width"]), int(r["height"])) == (96, 103)]
    assert len(std) == 54837
    assert len(real_odd) + len(alt_odd) + len(std) == len(socofing_eval)

def test_socofing_no_pattern_labels(socofing_eval):
    first_row = socofing_eval[0]
    forbidden_keys = {"class", "primary_class", "pattern", "pattern_class", "label", "nist_class"}
    for key in first_row.keys():
        assert key.lower() not in forbidden_keys, f"Found forbidden pattern key in SOCOFing: {key}"

def test_socofing_alteration_metadata(socofing_eval):
    tiers = Counter(r["alteration_tier"] for r in socofing_eval)
    assert tiers["None"] == 6000
    assert tiers["Easy"] == 17931
    assert tiers["Medium"] == 17067
    assert tiers["Hard"] == 14272

    types = Counter(r["alteration_type"] for r in socofing_eval)
    assert types["None"] == 6000
    assert types["Obl"] == 5982 + 5939 + 5933
    assert types["CR"] == 5969 + 5516 + 4954
    assert types["Zcut"] == 5980 + 5612 + 3385
