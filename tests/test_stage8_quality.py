"""
Automated Unit and Integration Tests for Stage 8: Image Quality Assessment & Damage Detection.
Verifies:
1. Local Ridge Clarity Index (RCI) on synthetic sine gratings, blurred gratings, and noise.
2. Focus / Blur and Laplacian variance scoring.
3. Dynamic range and saturation metrics (normal vs. dry vs. wet impressions).
4. Frequency continuity and spatial smoothness.
5. Damage segmentation and defect categorization (Scar, Obliteration, Rotation, Sensor Truncation).
6. Binary damage mask geometry and usable area percentages.
7. Acceptability gating and forensic quality warning generation.
8. End-to-end integration and latency performance (<50ms per image) on NIST SD04 and SOCOFing datasets.
"""

import os
import sys
import time
import pytest
import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.orientation import compute_orientation_field
from src.quality.assessment import (
    BiometricQualityResult,
    assess_biometric_quality,
    compute_local_ridge_clarity,
    compute_frequency_continuity,
    compute_sharpness_and_blur,
    compute_dynamic_range_saturation,
    compute_composite_quality_score,
    classify_quality_tier,
)
from src.quality.damage import (
    DamageSegmentationResult,
    segment_damage_and_defects,
    detect_ridge_flow_disruptions,
    categorize_defects,
)
from src.quality.analyzer import (
    QualityReport,
    analyze_fingerprint_quality,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_sine_grating():
    """Generates pristine 256x256 horizontal sine grating with period = 6.67 px (f0 = 0.15 cyc/px)."""
    h, w = 256, 256
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    f0 = 0.15
    grating = 128.0 + 120.0 * np.cos(2.0 * np.pi * f0 * X)
    return np.clip(grating, 0.0, 255.0).astype(np.float32)


@pytest.fixture
def synthetic_noise():
    """Generates uniform random noise in [0, 255]."""
    np.random.seed(42)
    return np.random.uniform(0.0, 255.0, (256, 256)).astype(np.float32)


@pytest.fixture
def synthetic_uniform_white():
    """Generates uniform white image (all 255.0)."""
    return np.ones((256, 256), dtype=np.float32) * 255.0


@pytest.fixture
def synthetic_uniform_black():
    """Generates uniform black image (all 0.0)."""
    return np.zeros((256, 256), dtype=np.float32)


@pytest.fixture
def sample_nist_image_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0012_01.png")
    if not os.path.exists(path):
        pytest.skip(f"NIST image not found: {path}")
    return path


@pytest.fixture
def sample_socofing_real_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/SOCOFing/Real/100__M_Left_index_finger.BMP")
    if not os.path.exists(path):
        pytest.skip(f"SOCOFing real image not found: {path}")
    return path


@pytest.fixture
def sample_socofing_obl_hard_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/SOCOFing/Altered/Altered-Hard/100__M_Left_index_finger_Obl.BMP")
    if not os.path.exists(path):
        pytest.skip(f"SOCOFing Altered-Hard Obl image not found: {path}")
    return path


@pytest.fixture
def sample_socofing_cr_hard_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/SOCOFing/Altered/Altered-Hard/100__M_Left_index_finger_CR.BMP")
    if not os.path.exists(path):
        pytest.skip(f"SOCOFing Altered-Hard CR image not found: {path}")
    return path


@pytest.fixture
def sample_socofing_zcut_hard_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/SOCOFing/Altered/Altered-Hard/100__M_Left_index_finger_Zcut.BMP")
    if not os.path.exists(path):
        pytest.skip(f"SOCOFing Altered-Hard Zcut image not found: {path}")
    return path


# ---------------------------------------------------------------------------
# UNIT TESTS: IQA ENGINE (assessment.py)
# ---------------------------------------------------------------------------

class TestBiometricIQA:
    def test_sine_grating_clarity_and_tier(self, synthetic_sine_grating):
        """Pristine sine grating must achieve high clarity score and 'Excellent' tier."""
        res = assess_biometric_quality(synthetic_sine_grating)
        assert isinstance(res, BiometricQualityResult)
        assert res.overall_score >= 80.0
        assert res.quality_tier == "Excellent"
        assert res.clarity_score >= 80.0
        assert res.coherence_score >= 90.0
        assert 0.0 <= res.overall_score <= 100.0

    def test_pure_noise_penalty(self, synthetic_noise):
        """Pure noise must be penalized to 'Poor' quality with low clarity and coherence."""
        res = assess_biometric_quality(synthetic_noise)
        assert res.overall_score < 40.0
        assert res.quality_tier == "Poor"
        assert res.clarity_score < 20.0
        assert res.coherence_score < 30.0

    def test_uniform_images_zero_quality(self, synthetic_uniform_white, synthetic_uniform_black):
        """Uniform white and black images must receive 0 quality score and 'Poor' tier."""
        res_w = assess_biometric_quality(synthetic_uniform_white)
        res_b = assess_biometric_quality(synthetic_uniform_black)
        assert res_w.overall_score == 0.0
        assert res_w.quality_tier == "Poor"
        assert res_b.overall_score == 0.0
        assert res_b.quality_tier == "Poor"

    def test_blur_penalty_on_sine_grating(self, synthetic_sine_grating):
        """Severe Gaussian blur must significantly reduce sharpness and Laplacian variance."""
        blurred = cv2.GaussianBlur(synthetic_sine_grating, (17, 17), 5.0)

        sharp_orig, lap_orig, grad_orig = compute_sharpness_and_blur(synthetic_sine_grating)
        sharp_blur, lap_blur, grad_blur = compute_sharpness_and_blur(blurred)

        assert sharp_blur < sharp_orig * 0.35
        assert lap_blur < lap_orig * 0.10
        assert grad_blur < grad_orig * 0.25

    def test_dryness_detection(self):
        """Synthetic dry print (high intensity, faint ridges, low dynamic range) flags is_dry."""
        dry_img = np.random.uniform(215.0, 250.0, (128, 128)).astype(np.float32)
        sat_score, dr, dry_r, wet_r, is_dry, is_wet = compute_dynamic_range_saturation(dry_img)
        assert is_dry is True
        assert is_wet is False
        assert dry_r > 0.30
        assert sat_score < 50.0

    def test_wetness_detection(self):
        """Synthetic wet print (flooded ink, dark saturated pixels) flags is_wet."""
        wet_img = np.random.uniform(5.0, 45.0, (128, 128)).astype(np.float32)
        sat_score, dr, dry_r, wet_r, is_dry, is_wet = compute_dynamic_range_saturation(wet_img)
        assert is_wet is True
        assert is_dry is False
        assert wet_r > 0.30
        assert sat_score < 50.0

    def test_quality_tier_boundaries(self):
        """Test exact boundary mappings for classify_quality_tier."""
        assert classify_quality_tier(100.0) == "Excellent"
        assert classify_quality_tier(80.0) == "Excellent"
        assert classify_quality_tier(79.9) == "Good"
        assert classify_quality_tier(60.0) == "Good"
        assert classify_quality_tier(59.9) == "Fair"
        assert classify_quality_tier(40.0) == "Fair"
        assert classify_quality_tier(39.9) == "Poor"
        assert classify_quality_tier(0.0) == "Poor"

    def test_frequency_continuity(self):
        """Smooth frequency grid must achieve high continuity score; erratic grid low score."""
        smooth_grid = np.ones((16, 16), dtype=np.float32) * 0.14
        erratic_grid = np.random.uniform(0.08, 0.28, (16, 16)).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.uint8)

        score_smooth = compute_frequency_continuity(smooth_grid, mask)
        score_erratic = compute_frequency_continuity(erratic_grid, mask)

        assert score_smooth >= 95.0
        assert score_erratic < 50.0


# ---------------------------------------------------------------------------
# UNIT TESTS: DAMAGE & DEFECT SEGMENTATION (damage.py)
# ---------------------------------------------------------------------------

class TestDamageSegmentation:
    def test_damage_mask_format_and_shape(self, synthetic_sine_grating):
        """Damage mask must be binary uint8 {0, 1} matching original image dimensions."""
        res = segment_damage_and_defects(synthetic_sine_grating)
        assert isinstance(res, DamageSegmentationResult)
        assert res.damage_mask.shape == synthetic_sine_grating.shape
        assert res.damage_mask.dtype == np.uint8
        assert set(np.unique(res.damage_mask)).issubset({0, 1})
        assert 0.0 <= res.usable_area_percentage <= 100.0
        assert 0.0 <= res.usable_foreground_ratio <= 1.0

    def test_intact_grating_zero_defects(self, synthetic_sine_grating):
        """Pristine sine grating must have zero damaged pixels and 100% usable area."""
        res = segment_damage_and_defects(synthetic_sine_grating)
        assert res.damaged_pixels == 0
        assert res.damage_detected is False
        assert res.usable_area_percentage >= 98.0

    def test_synthetic_cut_detection(self, synthetic_sine_grating):
        """Simulated horizontal scar / cut across ridges must be segmented into damage mask."""
        grating_with_cut = synthetic_sine_grating.copy()
        # Insert a 4-pixel wide white scar traversing columns 50 to 200
        grating_with_cut[126:130, 50:200] = 255.0

        res = segment_damage_and_defects(grating_with_cut)
        assert res.damaged_pixels > 0
        assert "Scar / Crease" in res.damage_types or "Rotation / Distortion Boundary" in res.damage_types

    def test_synthetic_obliteration_detection(self, synthetic_sine_grating):
        """Simulated circular obliteration patch must be segmented and flagged."""
        grating_with_obl = synthetic_sine_grating.copy()
        # Overwrite a 40x40 patch with uniform gray noise
        cv2.circle(grating_with_obl, (128, 128), 24, (128,), -1)

        res = segment_damage_and_defects(grating_with_obl)
        assert res.damaged_pixels > 100
        assert res.usable_area_percentage < 99.5
        assert "Obliteration / Abrasion" in res.damage_types

    def test_sensor_truncation_detection(self):
        """Impressions abutting the image boundaries must trigger 'Sensor Truncation'."""
        img = np.zeros((100, 100), dtype=np.float32)
        # Create full-width block touching top and bottom
        img[:, 10:90] = 128.0 + 100.0 * np.cos(2.0 * np.pi * 0.15 * np.arange(80)[None, :])
        res = segment_damage_and_defects(img)
        assert "Sensor Truncation" in res.damage_types

    def test_detect_ridge_flow_disruptions_empty_unpack(self):
        """Disruption detection on empty images must return a 7-element tuple without ValueError."""
        blank = np.zeros((64, 64), dtype=np.uint8)
        ores = compute_orientation_field(blank)
        ret = detect_ridge_flow_disruptions(blank, ores)
        assert isinstance(ret, tuple)
        assert len(ret) == 7
        clean, low_coh, seam, interior, adiff, local_std, fg_std = ret
        assert clean.shape == (64, 64)
        assert fg_std == 0.0


# ---------------------------------------------------------------------------
# INTEGRATION TESTS: UNIFIED ANALYZER (analyzer.py)
# ---------------------------------------------------------------------------

class TestUnifiedAnalyzer:
    def test_analyze_nist_clean_image(self, sample_nist_image_path):
        """Run analyze_fingerprint_quality on real clean NIST SD04 print."""
        report = analyze_fingerprint_quality(sample_nist_image_path)
        assert isinstance(report, QualityReport)
        assert 0.0 <= report.overall_score <= 100.0
        assert report.quality_tier in ["Good", "Excellent"]
        assert report.usable_area_percentage >= 70.0
        assert report.is_acceptable_for_classification is True
        assert report.damage_detected is False
        assert report.damage_mask.shape == (512, 512)

        # Check required metrics
        for key in ["clarity", "contrast", "sharpness", "saturation", "usable_area", "frequency_continuity"]:
            assert key in report.metrics
            assert 0.0 <= report.metrics[key] <= 100.0

    def test_analyze_socofing_real(self, sample_socofing_real_path):
        """SOCOFing Real baseline print must be acceptable for classification with high usable area."""
        report = analyze_fingerprint_quality(sample_socofing_real_path)
        assert isinstance(report, QualityReport)
        assert report.usable_area_percentage >= 90.0
        assert report.is_acceptable_for_classification is True
        assert report.damage_mask.shape == (103, 96)

    def test_analyze_socofing_obliteration(self, sample_socofing_obl_hard_path):
        """SOCOFing Altered-Hard Obliteration must detect obliteration and show degraded usable area."""
        report = analyze_fingerprint_quality(sample_socofing_obl_hard_path)
        assert isinstance(report, QualityReport)
        assert report.damage_detected is True
        assert "Obliteration / Abrasion" in report.damage_types
        assert report.damaged_pixels > 200
        assert "obliteration" in report.quality_warning.lower()

    def test_analyze_socofing_rotation(self, sample_socofing_cr_hard_path):
        """SOCOFing Altered-Hard Central Rotation must detect rotation/distortion seams."""
        report = analyze_fingerprint_quality(sample_socofing_cr_hard_path)
        assert isinstance(report, QualityReport)
        assert report.damage_detected is True
        assert "Rotation / Distortion Boundary" in report.damage_types

    def test_analyze_socofing_zcut(self, sample_socofing_zcut_hard_path):
        """SOCOFing Altered-Hard Z-cut must detect alteration damage."""
        report = analyze_fingerprint_quality(sample_socofing_zcut_hard_path)
        assert isinstance(report, QualityReport)
        assert report.damage_detected is True
        assert any(t in report.damage_types for t in ["Rotation / Distortion Boundary", "Scar / Crease", "Obliteration / Abrasion"])

    def test_overlay_generation(self, sample_socofing_obl_hard_path):
        """Diagnostic damage overlay must return valid 3-channel RGB image."""
        img = cv2.imread(sample_socofing_obl_hard_path, cv2.IMREAD_GRAYSCALE)
        report = analyze_fingerprint_quality(img)
        overlay = report.create_damage_overlay(img)
        assert overlay.shape == (103, 96, 3)
        assert overlay.dtype == np.uint8

    def test_input_type_flexibility(self, sample_socofing_real_path):
        """Pipeline must accept file path, numpy array, torch tensor, and PIL Image."""
        # 1. File path
        rep1 = analyze_fingerprint_quality(sample_socofing_real_path)

        # 2. NumPy array
        arr = cv2.imread(sample_socofing_real_path, cv2.IMREAD_GRAYSCALE)
        rep2 = analyze_fingerprint_quality(arr)

        # 3. Torch tensor
        tensor = torch.from_numpy(arr).float()
        rep3 = analyze_fingerprint_quality(tensor)

        # 4. PIL Image
        pil_img = Image.open(sample_socofing_real_path)
        rep4 = analyze_fingerprint_quality(pil_img)

        assert abs(rep1.overall_score - rep2.overall_score) < 1e-3
        assert abs(rep2.overall_score - rep3.overall_score) < 1e-3
        assert abs(rep3.overall_score - rep4.overall_score) < 0.05

    def test_acceptability_gate_logic(self):
        """Images with overall_score < 40 or usable_area < 40% must be marked unacceptable."""
        # Unacceptable due to zero variance
        blank = np.zeros((100, 100), dtype=np.float32)
        rep_blank = analyze_fingerprint_quality(blank)
        assert rep_blank.is_acceptable_for_classification is False
        assert rep_blank.overall_score == 0.0

    def test_performance_latency_under_50ms(self, sample_nist_image_path, sample_socofing_real_path):
        """Full analyze_fingerprint_quality pipeline must complete under 50ms on Apple Silicon."""
        # Warm-up run
        _ = analyze_fingerprint_quality(sample_nist_image_path)

        # Measure NIST 512x512
        t0 = time.perf_counter()
        _ = analyze_fingerprint_quality(sample_nist_image_path)
        nist_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Measure SOCOFing 103x96
        t0 = time.perf_counter()
        _ = analyze_fingerprint_quality(sample_socofing_real_path)
        socofing_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print(f"\nLatency Benchmark: NIST 512x512 = {nist_elapsed_ms:.2f}ms | SOCOFing = {socofing_elapsed_ms:.2f}ms")
        assert nist_elapsed_ms < 50.0, f"NIST latency {nist_elapsed_ms:.2f}ms exceeded 50ms budget!"
        assert socofing_elapsed_ms < 15.0, f"SOCOFing latency {socofing_elapsed_ms:.2f}ms exceeded 15ms budget!"
