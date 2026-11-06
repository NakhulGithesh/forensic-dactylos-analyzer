"""
Unit and Integration Tests for Stage 9: External Evaluation, Robustness & Unified Pipeline.
Verifies:
1. Unified FingerprintSystemAnalyzer end-to-end execution across Stages 4 through 8.
2. Quality gating on pristine, degraded, and degenerate blank/noise inputs.
3. Serialization integrity: 100% JSON compliance for Stage 10 Web API.
4. Multi-layer diagnostic overlay rendering.
5. Robustness evaluator, pairing logic, and metric calculations across SOCOFing.
"""

import os
import sys
import json
import pytest
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import FingerprintSystemAnalyzer, FingerprintAnalysisResult
from src.evaluation import FingerprintRobustnessEvaluator, PairEvaluation, compute_robustness_metrics
from src.subtypes.taxonomy import TAXONOMY_REGISTRY, SubtypeLabel, PrimaryCategory


@pytest.fixture(scope="module")
def analyzer():
    """Module-scoped unified analyzer instance."""
    return FingerprintSystemAnalyzer(model_checkpoint="models/best_model.pt")


@pytest.fixture
def clean_nist_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0012_01.png")
    if not os.path.exists(path):
        pytest.skip(f"NIST clean image not found: {path}")
    return path


@pytest.fixture
def ambiguous_nist_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0026_03.png")
    if not os.path.exists(path):
        pytest.skip(f"NIST ambiguous image not found: {path}")
    return path


@pytest.fixture
def socofing_real_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/SOCOFing/Real/100__M_Left_index_finger.BMP")
    if not os.path.exists(path):
        pytest.skip(f"SOCOFing real image not found: {path}")
    return path


@pytest.fixture
def socofing_obl_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/SOCOFing/Altered/Altered-Hard/100__M_Left_index_finger_Obl.BMP")
    if not os.path.exists(path):
        pytest.skip(f"SOCOFing altered obl image not found: {path}")
    return path


# ---------------------------------------------------------------------------
# PIPELINE INITIALIZATION & CAPABILITIES
# ---------------------------------------------------------------------------

def test_analyzer_initialization(analyzer):
    """Verify analyzer initialization, device selection, and default gates."""
    assert analyzer.device is not None
    assert analyzer.min_quality_score == 40.0
    assert analyzer.min_usable_area == 40.0
    assert analyzer.min_coherence_threshold == 0.25
    assert analyzer.min_confidence_gate == 0.45
    assert analyzer.model is not None


# ---------------------------------------------------------------------------
# REAL SAMPLES END-TO-END VERIFICATION
# ---------------------------------------------------------------------------

def test_clean_nist_analysis(analyzer, clean_nist_path):
    """Verify end-to-end execution on clean rolled NIST sample."""
    res = analyzer.analyze(clean_nist_path, hand="Right", finger_num=1, digit_code="01")

    assert isinstance(res, FingerprintAnalysisResult)
    assert res.decision_gate == "ACCEPTED"
    assert res.is_acceptable is True
    assert res.classification["primary_class"] == "A"
    assert res.classification["confidence"] > 0.85
    assert res.subtype["subtype"] in [s.value for s in TAXONOMY_REGISTRY.keys()]
    assert res.subtype["status"] == "Resolved"
    assert res.quality["overall_score"] >= 70.0
    assert res.quality["usable_area_percentage"] >= 85.0
    assert res.features["num_cores"] == 0
    assert res.features["num_deltas"] == 0
    assert len(res.explanation) == 5
    assert "total_ms" in res.execution_time_ms
    assert res.execution_time_ms["total_ms"] > 0


def test_ambiguous_nist_analysis(analyzer, ambiguous_nist_path):
    """Verify permissive confidence and runner-up tracking on ambiguous NIST sample."""
    res = analyzer.analyze(ambiguous_nist_path, hand="Right", finger_num=3, digit_code="03")

    assert isinstance(res, FingerprintAnalysisResult)
    assert res.classification["primary_class"] == "T"
    assert len(res.classification["top_k"]) >= 2
    # Verify runner-up class is R (as cross-referenced in ground truth)
    runner_up = res.classification["top_k"][1]
    assert runner_up["class"] == "R"
    assert runner_up["prob"] > 0.05
    assert len(res.explanation) == 5


def test_socofing_real_analysis(analyzer, socofing_real_path):
    """Verify touch-sensor handling and delta truncation recognition on SOCOFing."""
    res = analyzer.analyze(socofing_real_path, hand="Left", finger_num=7)

    assert isinstance(res, FingerprintAnalysisResult)
    assert res.quality["overall_score"] > 60.0
    assert res.features["delta_truncated"] is True or res.subtype["status"] == "Uncertain"
    assert res.subtype["status"] in ("Resolved", "Uncertain")
    assert res.execution_time_ms["total_ms"] < 100.0  # Fast touch sensor print


def test_socofing_altered_damage_detection(analyzer, socofing_obl_path, socofing_real_path):
    """Verify defect segmentation, quality drop, and warning generation on altered print."""
    res_real = analyzer.analyze(socofing_real_path, hand="Left", finger_num=7)
    res_alt = analyzer.analyze(socofing_obl_path, hand="Left", finger_num=7)

    # Altered print should detect damage
    assert res_alt.quality["damage_detected"] is True
    assert len(res_alt.quality["damage_types"]) > 0
    assert "Obliteration / Abrasion" in res_alt.quality["damage_types"] or res_alt.quality["damaged_pixels"] > 0
    assert res_alt.quality["quality_warning"] != "None"

    # Quality score should drop relative to Real
    assert res_alt.quality["overall_score"] < res_real.quality["overall_score"]


# ---------------------------------------------------------------------------
# QUALITY GATING & DEGENERATE INPUT GUARDS
# ---------------------------------------------------------------------------

def test_quality_gate_uniform_black(analyzer):
    """Verify uniform black image triggers degenerate quality gate without crashing."""
    black_img = np.zeros((256, 256), dtype=np.float32)
    res = analyzer.analyze(black_img)

    assert res.decision_gate == "REJECTED_LOW_QUALITY"
    assert res.is_acceptable is False
    assert res.quality["overall_score"] == 0.0
    assert res.subtype["is_uncertain"] is True
    assert res.classification["confidence"] == 0.0
    assert "Degenerate" in res.quality["quality_warning"] or "zero variance" in res.quality["quality_warning"].lower()


def test_quality_gate_uniform_white(analyzer):
    """Verify uniform white image triggers degenerate quality gate without crashing."""
    white_img = np.ones((256, 256), dtype=np.float32) * 255.0
    res = analyzer.analyze(white_img)

    assert res.decision_gate == "REJECTED_LOW_QUALITY"
    assert res.is_acceptable is False
    assert res.quality["overall_score"] == 0.0
    assert res.subtype["is_uncertain"] is True


def test_quality_gate_gaussian_noise(analyzer):
    """Verify random unstructured noise is gated and flagged as Poor / Low Quality."""
    rng = np.random.default_rng(1234)
    noise_img = rng.normal(loc=128.0, scale=30.0, size=(256, 256)).astype(np.float32)
    noise_img = np.clip(noise_img, 0.0, 255.0)

    res = analyzer.analyze(noise_img)

    # Coherence of pure Gaussian noise is near zero
    assert res.quality["overall_score"] < 40.0
    assert res.is_acceptable is False
    assert res.decision_gate == "REJECTED_LOW_QUALITY"
    assert res.subtype["is_uncertain"] is True


# ---------------------------------------------------------------------------
# SERIALIZATION & WEB API READINESS
# ---------------------------------------------------------------------------

def test_json_dict_serialization(analyzer, clean_nist_path):
    """Verify to_dict() and to_json() return clean primitives with zero serialization errors."""
    res = analyzer.analyze(clean_nist_path, hand="Right", finger_num=1)

    d = res.to_dict()
    assert isinstance(d, dict)
    assert isinstance(d["image_path"], str)
    assert isinstance(d["input_shape"], list)
    assert isinstance(d["quality"]["overall_score"], float)
    assert isinstance(d["classification"]["probabilities"], dict)
    assert isinstance(d["explanation"], list)

    # Test complete JSON encoding roundtrip
    json_str = res.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["decision_gate"] == res.decision_gate
    assert parsed["classification"]["primary_class"] == res.classification["primary_class"]


# ---------------------------------------------------------------------------
# VISUAL OVERLAYS & ALTERNATIVE INPUT FORMATS
# ---------------------------------------------------------------------------

def test_create_overlay(analyzer, clean_nist_path):
    """Verify create_overlay() generates valid 3-channel RGB image."""
    res = analyzer.analyze(clean_nist_path, hand="Right", finger_num=1)
    overlay = res.create_overlay()

    assert isinstance(overlay, np.ndarray)
    assert overlay.ndim == 3
    assert overlay.shape[2] == 3
    assert overlay.dtype == np.uint8
    assert overlay.shape[:2] == res.input_shape


def test_pil_image_input(analyzer):
    """Verify PIL Image ingestion operates identically to file path."""
    arr = (np.random.default_rng(42).uniform(50, 200, (128, 128))).astype(np.uint8)
    pil_img = Image.fromarray(arr, mode="L")

    res = analyzer.analyze(pil_img)
    assert isinstance(res, FingerprintAnalysisResult)
    assert res.input_shape == (128, 128)


def test_batch_analysis(analyzer, clean_nist_path, ambiguous_nist_path):
    """Verify analyze_batch() handles multiple inputs."""
    images = [clean_nist_path, ambiguous_nist_path]
    meta = [{"hand": "Right", "finger_num": 1}, {"hand": "Right", "finger_num": 3}]

    results = analyzer.analyze_batch(images, metadata_list=meta)
    assert len(results) == 2
    assert results[0].classification["primary_class"] == "A"
    assert results[1].classification["primary_class"] == "T"


# ---------------------------------------------------------------------------
# ROBUSTNESS EVALUATOR & METRICS VERIFICATION
# ---------------------------------------------------------------------------

def test_robustness_evaluator_pairing():
    """Verify FingerprintRobustnessEvaluator correctly pairs Real with Altered impressions."""
    evaluator = FingerprintRobustnessEvaluator()
    assert len(evaluator.paired_cohorts) > 1000

    # Inspect first cohort
    first_key = next(iter(evaluator.paired_cohorts.keys()))
    cohort = evaluator.paired_cohorts[first_key]
    assert "real_path" in cohort
    assert len(cohort["altered"]) == 9  # 3 tiers x 3 types


def test_robustness_mini_cohort_run(analyzer):
    """Verify robustness evaluation on a 1-finger mini cohort (9 alteration pairs)."""
    evaluator = FingerprintRobustnessEvaluator(analyzer=analyzer)
    pair_evals, metrics = evaluator.evaluate_cohort(cohort_size=1, seed=42)

    assert len(pair_evals) == 9
    assert "summary" in metrics
    assert "macro_stability" in metrics
    assert "feature_degradation" in metrics
    assert "correlations" in metrics
    assert metrics["summary"]["unique_fingers_evaluated"] == 1
    assert metrics["summary"]["total_pair_evaluations"] == 9
    assert 0.0 <= metrics["summary"]["overall_stability_rate_pct"] <= 100.0


# ---------------------------------------------------------------------------
# HARDENING & LATENCY REGRESSION TESTS
# ---------------------------------------------------------------------------

def test_non_standard_aspect_ratios(analyzer, clean_nist_path):
    """Verify analyzer handles wide and tall non-standard aspect ratio crops gracefully."""
    import cv2
    img = cv2.imread(clean_nist_path, cv2.IMREAD_GRAYSCALE)

    # Wide aspect ratio crop: 100 x 512
    crop_wide = img[100:200, :]
    res_wide = analyzer.analyze(crop_wide)
    assert res_wide.input_shape == (100, 512)
    assert res_wide.decision_gate in ("ACCEPTED", "ACCEPTED_WITH_WARNINGS", "REJECTED_LOW_QUALITY")

    # Tall aspect ratio crop: 512 x 100
    crop_tall = img[:, 100:200]
    res_tall = analyzer.analyze(crop_tall)
    assert res_tall.input_shape == (512, 100)
    assert res_tall.decision_gate in ("ACCEPTED", "ACCEPTED_WITH_WARNINGS", "REJECTED_LOW_QUALITY")


def test_pytorch_tensor_and_multichannel_input(analyzer):
    """Verify PyTorch tensor (1, C, H, W) and 3-channel RGB NumPy array ingestion."""
    # 3-channel RGB image
    rgb = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    res_rgb = analyzer.analyze(rgb)
    assert res_rgb.input_shape == (256, 256)

    # 4D PyTorch tensor: (1, 1, 256, 256)
    t4d = torch.rand(1, 1, 256, 256, dtype=torch.float32) * 255.0
    res_t4d = analyzer.analyze(t4d)
    assert res_t4d.input_shape == (256, 256)


def test_invalid_inputs_raise_informative_errors(analyzer):
    """Verify None, empty string, and zero-element arrays raise appropriate exceptions."""
    with pytest.raises(ValueError, match="Input image cannot be None"):
        analyzer.analyze(None)

    with pytest.raises(FileNotFoundError):
        analyzer.analyze("")

    with pytest.raises(ValueError, match="zero elements"):
        analyzer.analyze(np.zeros((0, 0), dtype=np.float32))


def test_json_serialization_payload_compactness(analyzer, clean_nist_path):
    """Verify JSON payload does not serialize raw 2D pixel raster arrays and remains < 50 KB."""
    res = analyzer.analyze(clean_nist_path, hand="Right", finger_num=1)
    json_str = res.to_json()

    assert len(json_str) < 50_000, f"JSON payload too large: {len(json_str)} bytes (must be < 50 KB)"
    # Verify shape metadata is preserved for client-side diagnostics
    assert "details" in res.quality
    if "clarity_map_shape" in res.quality["details"]:
        assert res.quality["details"]["clarity_map_shape"] == [512, 512]


def test_nist_end_to_end_latency_under_100ms(analyzer, clean_nist_path):
    """Verify full end-to-end pipeline on 512x512 NIST image satisfies < 100 ms latency requirement."""
    import time
    # Warmup
    _ = analyzer.analyze(clean_nist_path, hand="Right", finger_num=1)

    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = analyzer.analyze(clean_nist_path, hand="Right", finger_num=1)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_latency = float(np.mean(latencies))
    assert mean_latency < 100.0, f"Pipeline latency too slow on Apple Silicon: {mean_latency:.2f} ms >= 100.0 ms"

