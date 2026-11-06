"""
Automated Unit Tests for Stage 5: Fingerprint Feature Detection Engine.
Verifies:
1. Continuous Ridge Orientation Field bounds [0, pi), coherence [0, 1], and synthetic pattern accuracy.
2. Poincaré Index mathematical properties (+0.5 for core, -0.5 for delta, 0 for flat).
3. Complex rotational symmetry filter orthogonality and peak response.
4. Non-maximum suppression, dipole cancellation, and calibrated confidence scoring.
5. Sensor truncation flags (is_truncated, delta_truncated).
6. Local ridge frequency bounds [0.10, 0.25] cycles/pixel and streamline tracing.
7. Loop directionality and anatomical Ulnar vs. Radial mapping.
8. Execution performance on Apple Silicon (M5 Pro).
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

from src.data.transforms import FingerprintTransformPipeline
from src.features.orientation import (
    OrientationResult,
    compute_orientation_field,
    compute_coherence,
    compute_orientation_curvature,
    segment_foreground,
    to_grayscale_numpy,
)
from src.features.singular_points import (
    SingularPoint,
    SingularPointsResult,
    detect_singular_points,
    compute_poincare_index_map,
    compute_complex_phase_filters,
    compute_singular_point_confidence,
    cluster_singular_points,
    cancel_core_delta_dipoles,
    predict_delta_truncation,
)
from src.features.geometry import (
    GeometryResult,
    estimate_ridge_frequency,
    trace_ridge_streamlines,
    trace_single_streamline,
    detect_loop_directionality,
    extract_fingerprint_geometry,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nist_arch_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0005_03.png")
    assert os.path.isfile(path), f"Arch print not found: {path}"
    return path


@pytest.fixture(scope="module")
def nist_whorl_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0001_01.png")
    assert os.path.isfile(path), f"Whorl print not found: {path}"
    return path


@pytest.fixture(scope="module")
def nist_left_loop_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0003_10.png")
    assert os.path.isfile(path), f"Left loop print not found: {path}"
    return path


@pytest.fixture(scope="module")
def nist_right_loop_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0002_05.png")
    assert os.path.isfile(path), f"Right loop print not found: {path}"
    return path


@pytest.fixture(scope="module")
def socofing_real_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/SOCOFing/Real/100__M_Left_index_finger.BMP")
    assert os.path.isfile(path), f"SOCOFing sample not found: {path}"
    return path


# ---------------------------------------------------------------------------
# 1. ORIENTATION & SEGMENTATION TESTS
# ---------------------------------------------------------------------------

class TestOrientationField:
    """Verifies orientation field estimation, bounds, and mathematical correctness."""

    def test_orientation_bounds_and_shapes(self, nist_right_loop_path):
        img = cv2.imread(nist_right_loop_path, cv2.IMREAD_GRAYSCALE)[:480, :]
        res = compute_orientation_field(img, block_size=16)

        assert isinstance(res, OrientationResult)
        assert res.theta.shape == (480, 512)
        assert res.coherence.shape == (480, 512)
        assert res.mask.shape == (480, 512)
        assert res.block_theta.shape == (30, 32)
        assert res.block_coherence.shape == (30, 32)
        assert res.block_mask.shape == (30, 32)

        # Mathematical interval check: theta in [0, pi)
        assert np.all(res.theta >= 0.0)
        assert np.all(res.theta < np.pi)
        assert np.all(res.block_theta >= 0.0)
        assert np.all(res.block_theta < np.pi)

        # Degrees property: [0, 180)
        deg = res.theta_degrees
        assert np.all(deg >= 0.0)
        assert np.all(deg < 180.0)

        # Coherence bounds: [0.0, 1.0]
        assert np.all(res.coherence >= 0.0)
        assert np.all(res.coherence <= 1.0)
        assert np.all(res.block_coherence >= 0.0)
        assert np.all(res.block_coherence <= 1.0)

        # Mask values: {0, 1}
        assert set(np.unique(res.mask)).issubset({0, 1})
        assert set(np.unique(res.block_mask)).issubset({0, 1})

    def test_synthetic_horizontal_ridges(self):
        """Horizontal ridges (y-varying) must have theta ~ 0 (or ~ pi) and coherence ~ 1."""
        y, _ = np.mgrid[0:128, 0:128]
        img = (127.5 * (1 + np.sin(2 * np.pi * y / 8.0))).astype(np.uint8)
        res = compute_orientation_field(img, block_size=8)

        # Interior ridge region
        sub_th = res.theta[32:96, 32:96]
        # In [0, pi), angle close to 0 or pi
        th_dist_to_zero = np.minimum(sub_th, np.pi - sub_th)
        assert np.all(th_dist_to_zero < 0.15)
        assert np.mean(res.coherence[32:96, 32:96]) > 0.98

    def test_synthetic_vertical_ridges(self):
        """Vertical ridges (x-varying) must have theta ~ pi/2 and coherence ~ 1."""
        _, x = np.mgrid[0:128, 0:128]
        img = (127.5 * (1 + np.sin(2 * np.pi * x / 8.0))).astype(np.uint8)
        res = compute_orientation_field(img, block_size=8)

        sub_th = res.theta[32:96, 32:96]
        assert np.all(np.abs(sub_th - np.pi / 2.0) < 0.15)
        assert np.mean(res.coherence[32:96, 32:96]) > 0.98

    def test_synthetic_diagonal_ridges(self):
        """Diagonal 45-deg ridges (x + y) have normal at 45 deg -> tangent theta ~ 135 deg (3*pi/4)."""
        y, x = np.mgrid[0:128, 0:128]
        img = (127.5 * (1 + np.sin(2 * np.pi * (x + y) / (8.0 * np.sqrt(2))))).astype(np.uint8)
        res = compute_orientation_field(img, block_size=8)

        sub_th = res.theta[32:96, 32:96]
        expected_th = 3.0 * np.pi / 4.0
        assert np.all(np.abs(sub_th - expected_th) < 0.15)
        assert np.mean(res.coherence[32:96, 32:96]) > 0.98

    def test_input_type_tolerance(self, nist_right_loop_path):
        """Supports numpy, torch tensor (CHW / HW), and PIL Image."""
        pil_img = Image.open(nist_right_loop_path)
        res_pil = compute_orientation_field(pil_img, block_size=16)
        assert isinstance(res_pil, OrientationResult)

        arr = np.array(pil_img)
        res_np = compute_orientation_field(arr, block_size=16)
        assert np.allclose(res_pil.theta, res_np.theta, atol=1e-5)

        t_chw = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
        res_torch = compute_orientation_field(t_chw, block_size=16)
        assert np.allclose(res_np.theta, res_torch.theta, atol=1e-4)

    def test_constant_image_edge_case(self):
        """Zero / flat white image produces valid bounds with zero coherence and empty mask."""
        flat = np.full((128, 128), 255, dtype=np.uint8)
        res = compute_orientation_field(flat, block_size=16)
        assert not np.any(np.isnan(res.theta))
        assert not np.any(np.isnan(res.coherence))
        assert np.max(res.coherence) == 0.0
        assert np.count_nonzero(res.mask) == 0

    def test_orientation_curvature_properties(self):
        """Orientation curvature ||grad(theta)|| peaks at singularities and vanishes on uniform fields."""
        # 1. Analytical core
        r, c = np.mgrid[0:65, 0:65]
        x = (c - 32).astype(float)
        y = (r - 32).astype(float)
        z = x + 1j * y
        th_core = (0.5 * np.angle(z)) % np.pi
        curv_core = compute_orientation_curvature(th_core.astype(np.float32), sigma=0.0)
        assert curv_core[32, 32] > 3.0
        assert curv_core[32, 52] < 0.30

        # 2. Uniform orientation field
        uni = np.full((65, 65), np.pi / 3.0, dtype=np.float32)
        curv_uni = compute_orientation_curvature(uni, sigma=0.0)
        assert np.all(curv_uni < 1e-4)

    def test_standardized_tensor_input_to_orientation_field(self, nist_right_loop_path):
        """Verifies that standardized float32 tensors (mean 0, std 1) are cleanly handled."""
        pipe = FingerprintTransformPipeline(target_size=(256, 256), normalization="standardize")
        pil_img = Image.open(nist_right_loop_path)
        tensor = pipe(pil_img)
        assert tensor.dtype == torch.float32

        res = compute_orientation_field(tensor, block_size=8)
        assert isinstance(res, OrientationResult)
        assert np.count_nonzero(res.mask) > 1000
        assert np.mean(res.coherence[res.mask > 0]) > 0.40


# ---------------------------------------------------------------------------
# 2. SINGULAR POINTS & POINCARÉ INDEX TESTS
# ---------------------------------------------------------------------------

class TestSingularPoints:
    """Verifies Poincaré index calculation, complex filters, NMS, and confidence."""

    def test_poincare_mathematical_core(self):
        """Sherlock-Monro analytical core model: theta(z) = 0.5 * arg(z) yields Poincare = +0.5."""
        r, c = np.mgrid[0:41, 0:41]
        x = (c - 20).astype(float)
        y = (r - 20).astype(float)  # Image coordinates (row index)
        z = x + 1j * y
        theta_core = (0.5 * np.angle(z)) % np.pi

        pmap = compute_poincare_index_map(theta_core.astype(np.float32), radius=1)
        # Center cell (row 20, col 20)
        assert abs(pmap[20, 20] - 0.5) < 1e-4

    def test_poincare_mathematical_delta(self):
        """Sherlock-Monro analytical delta model: theta(z) = -0.5 * arg(z) yields Poincare = -0.5."""
        r, c = np.mgrid[0:41, 0:41]
        x = (c - 20).astype(float)
        y = (r - 20).astype(float)
        z = x + 1j * y
        theta_delta = (-0.5 * np.angle(z)) % np.pi

        pmap = compute_poincare_index_map(theta_delta.astype(np.float32), radius=1)
        assert abs(pmap[20, 20] - (-0.5)) < 1e-4

    def test_poincare_uniform_field(self):
        """Uniform orientation field yields Poincare = 0.0 everywhere."""
        uni = np.full((30, 30), np.pi / 4.0, dtype=np.float32)
        pmap = compute_poincare_index_map(uni, radius=1)
        assert np.all(np.abs(pmap) < 1e-5)

    def test_complex_filter_orthogonality(self):
        """Core symmetry filter peaks on core and delta filter peaks on delta."""
        r, c = np.mgrid[0:65, 0:65]
        x = (c - 32).astype(float)
        y = (r - 32).astype(float)
        z = x + 1j * y
        th_core = (0.5 * np.angle(z)) % np.pi
        th_delta = (-0.5 * np.angle(z)) % np.pi

        # On core field:
        rc_core, rd_core = compute_complex_phase_filters(th_core.astype(np.float32), sigma=8.0)
        assert rc_core[32, 32] > 0.85
        assert rd_core[32, 32] < 0.15

        # On delta field:
        rc_delta, rd_delta = compute_complex_phase_filters(th_delta.astype(np.float32), sigma=8.0)
        assert rd_delta[32, 32] > 0.85
        assert rc_delta[32, 32] < 0.15

    def test_dipole_cancellation(self):
        """Candidate core and delta within dipole distance are successfully cancelled."""
        cores = [(100.0, 100.0, 0.5, 0.8), (200.0, 200.0, 0.5, 0.9)]
        deltas = [(105.0, 105.0, -0.5, 0.7), (300.0, 300.0, -0.5, 0.85)]

        c_clean, d_clean = cancel_core_delta_dipoles(cores, deltas, dipole_distance=20.0)
        assert len(c_clean) == 1
        assert len(d_clean) == 1
        assert c_clean[0][0] == 200.0
        assert d_clean[0][0] == 300.0

    def test_nms_clustering(self):
        """Duplicate detections within cluster radius are collapsed to weighted centroid."""
        raw_pts = [
            (50.0, 50.0, 0.5, 0.8),
            (54.0, 52.0, 0.5, 0.9),
            (48.0, 51.0, 0.5, 0.7),
            (150.0, 150.0, 0.5, 0.95),
        ]
        clustered = cluster_singular_points(raw_pts, kind="core", radius=20.0)
        assert len(clustered) == 2
        # First cluster near (50, 50)
        assert 48.0 <= clustered[0][0] <= 54.0
        # Second cluster at (150, 150)
        assert clustered[1][0] == 150.0

    def test_fingerprint_class_singularities(self, nist_arch_path, nist_whorl_path, nist_right_loop_path):
        """Tests topological properties: Arch = 0 points, Whorl >= 2 deltas, Loop = 1 core."""
        # 1. Arch
        img_a = cv2.imread(nist_arch_path, cv2.IMREAD_GRAYSCALE)[:480, :]
        of_a = compute_orientation_field(img_a, block_size=16, sigma_smooth=6.0)
        sp_a = detect_singular_points(of_a)
        assert sp_a.num_cores == 0
        assert sp_a.num_deltas == 0

        # 2. Right Loop
        img_r = cv2.imread(nist_right_loop_path, cv2.IMREAD_GRAYSCALE)[:480, :]
        of_r = compute_orientation_field(img_r, block_size=16, sigma_smooth=6.0)
        sp_r = detect_singular_points(of_r)
        assert sp_r.num_cores == 1
        assert sp_r.num_deltas >= 1
        # Confidence score validation
        assert sp_r.cores[0].confidence >= 0.70
        assert 0.0 <= sp_r.cores[0].confidence <= 1.0

        # 3. Whorl
        img_w = cv2.imread(nist_whorl_path, cv2.IMREAD_GRAYSCALE)[:480, :]
        of_w = compute_orientation_field(img_w, block_size=16, sigma_smooth=6.0)
        sp_w = detect_singular_points(of_w)
        assert sp_w.num_cores >= 1
        assert sp_w.num_deltas >= 2

    def test_truncation_detection_on_socofing(self, socofing_real_path):
        """Live-scan optical print detects boundary proximity and flags truncated features."""
        img = cv2.imread(socofing_real_path, cv2.IMREAD_GRAYSCALE)
        of = compute_orientation_field(img, block_size=8, sigma_smooth=4.0)
        sp = detect_singular_points(of)

        assert isinstance(sp, SingularPointsResult)
        # Any singular point within 16 px of sensor boundary must be marked is_truncated=True
        for p in sp.all_points:
            if p.x <= 16 or p.x >= img.shape[1] - 16 or p.y <= 16 or p.y >= img.shape[0] - 16:
                assert p.is_truncated is True
                assert p.feature_visible is False

    def test_predict_delta_truncation_sectors(self):
        """Right loop predicts missing delta on bottom-left; Left loop on bottom-right."""
        # Right loop field: sub-theta has sin(2*theta) > 0
        h, w = 100, 100
        mask = np.ones((h, w), dtype=np.uint8)
        th_right = np.full((h, w), np.pi / 4.0, dtype=np.float32) # sin(2*th) = 1.0 > 0
        core_r = [SingularPoint(50.0, 30.0, "core", 0.85, 0.5, 0.8, False, True)]
        is_trunc_r, sector_r = predict_delta_truncation(core_r, [], th_right, mask)
        assert is_trunc_r is True
        assert sector_r == "bottom-left"

        # Left loop field: sub-theta has sin(2*theta) < 0
        th_left = np.full((h, w), 3.0 * np.pi / 4.0, dtype=np.float32) # sin(2*th) = -1.0 < 0
        core_l = [SingularPoint(50.0, 30.0, "core", 0.85, 0.5, 0.8, False, True)]
        is_trunc_l, sector_l = predict_delta_truncation(core_l, [], th_left, mask)
        assert is_trunc_l is True
        assert sector_l == "bottom-right"

        # Whorl with 2 cores and 1 delta on left -> missing delta on right
        cores_w = [
            SingularPoint(40.0, 40.0, "core", 0.85, 0.5, 0.8, False, True),
            SingularPoint(60.0, 40.0, "core", 0.85, 0.5, 0.8, False, True),
        ]
        delta_left = [SingularPoint(25.0, 65.0, "delta", 0.80, -0.5, 0.7, False, True)]
        is_trunc_w, sector_w = predict_delta_truncation(cores_w, delta_left, th_right, mask)
        assert is_trunc_w is True
        assert sector_w == "bottom-right"

    def test_all_four_singular_point_methods_execute(self, nist_right_loop_path):
        """Verifies poincare, complex_filter, hybrid, and curvature methods on real print."""
        img = cv2.imread(nist_right_loop_path, cv2.IMREAD_GRAYSCALE)[:480, :]
        of = compute_orientation_field(img, block_size=16)

        sp_hybrid = detect_singular_points(of, method="hybrid")
        sp_poinc = detect_singular_points(of, method="poincare")
        sp_filter = detect_singular_points(of, method="complex_filter")
        sp_curv = detect_singular_points(of, method="curvature")

        for sp, m in [(sp_hybrid, "hybrid"), (sp_poinc, "poincare"), (sp_filter, "complex_filter"), (sp_curv, "curvature")]:
            assert isinstance(sp, SingularPointsResult)
            assert sp.method == m
            assert sp.poincare_map.shape == of.block_theta.shape
            assert sp.core_response_map.shape == of.theta.shape
            assert sp.delta_response_map.shape == of.theta.shape
            assert sp.curvature_map is not None
            assert sp.curvature_map.shape == of.theta.shape

    def test_crease_and_subdelta_pseudo_core_suppression(self):
        """Verifies detect_singular_points() prunes candidate cores in bottom 20% crease zone, below deltas, or within 45px of deltas."""
        h, w = 256, 256
        y, x = np.mgrid[0:h, 0:w]

        # 1. Test field with central core (128, 98), pseudo-core at bottom crease (128, 230), and delta at (72, 170)
        z1 = (x - 128) + 1j * (y - 98)
        z2 = (x - 128) + 1j * (y - 230)
        zd = (x - 72) + 1j * (y - 170)

        phase = np.exp(1j * np.angle(z1)) * np.exp(1j * np.angle(z2)) * np.exp(-1j * np.angle(zd))
        theta = (0.5 * np.angle(phase)) % np.pi
        mask = np.ones((h, w), dtype=np.uint8) * 255
        coherence = np.ones((h, w), dtype=np.float32) * 0.90

        block_size = 16
        b_th = cv2.resize(theta, (w // block_size, h // block_size))
        b_coh = np.ones((h // block_size, w // block_size), dtype=np.float32)
        b_mask = np.ones((h // block_size, w // block_size), dtype=np.uint8)

        of = OrientationResult(
            theta=theta.astype(np.float32),
            coherence=coherence,
            mask=mask,
            vx=np.cos(2 * theta).astype(np.float32),
            vy=np.sin(2 * theta).astype(np.float32),
            block_theta=b_th.astype(np.float32),
            block_coherence=b_coh,
            block_mask=b_mask,
            block_size=block_size,
        )

        sp = detect_singular_points(of, method="poincare")
        # Central core at ~98 should be preserved
        assert sp.num_cores == 1
        assert sp.cores[0].y < 120.0
        # Pseudo-core at y=230 (in bottom 20% > 204.8 and below delta at 170) must be pruned
        assert not any(c.y >= 204.8 for c in sp.cores)
        if sp.deltas:
            max_delta_y = max(d.y for d in sp.deltas)
            assert not any(c.y > max_delta_y for c in sp.cores)
            for c in sp.cores:
                assert all(np.hypot(c.x - d.x, c.y - d.y) >= 45.0 for d in sp.deltas)

        # 2. Test bottom 20% crease zone suppression without deltas
        z_crease = (x - 128) + 1j * (y - 230)
        th_crease = (0.5 * np.angle(z_crease)) % np.pi
        b_th_crease = cv2.resize(th_crease, (w // block_size, h // block_size))
        of_crease = OrientationResult(
            theta=th_crease.astype(np.float32),
            coherence=coherence,
            mask=mask,
            vx=np.cos(2 * th_crease).astype(np.float32),
            vy=np.sin(2 * th_crease).astype(np.float32),
            block_theta=b_th_crease.astype(np.float32),
            block_coherence=b_coh,
            block_mask=b_mask,
            block_size=block_size,
        )
        sp_crease = detect_singular_points(of_crease, method="poincare")
        assert sp_crease.num_cores == 0

        # 3. Test sub-delta pseudo-core suppression where core is strictly above the crease cutoff (150 < 204.8)
        # Delta at (60, 100), core at (128, 60), pseudo-core at (128, 150) [below delta but not in bottom 20%]
        z_c1_sub = (x - 128) + 1j * (y - 60)
        z_c2_sub = (x - 128) + 1j * (y - 150)
        z_d_sub = (x - 60) + 1j * (y - 100)
        phase_sub = np.exp(1j * np.angle(z_c1_sub)) * np.exp(1j * np.angle(z_c2_sub)) * np.exp(-1j * np.angle(z_d_sub))
        th_sub = (0.5 * np.angle(phase_sub)) % np.pi
        b_th_sub = cv2.resize(th_sub, (w // block_size, h // block_size))
        of_sub = OrientationResult(
            theta=th_sub.astype(np.float32),
            coherence=coherence,
            mask=mask,
            vx=np.cos(2 * th_sub).astype(np.float32),
            vy=np.sin(2 * th_sub).astype(np.float32),
            block_theta=b_th_sub.astype(np.float32),
            block_coherence=b_coh,
            block_mask=b_mask,
            block_size=block_size,
        )
        sp_sub = detect_singular_points(of_sub, method="poincare")
        assert sp_sub.num_cores == 1
        assert sp_sub.cores[0].y < 100.0  # Only the core above the delta line is retained
        assert not any(c.y > 100.0 for c in sp_sub.cores)




# ---------------------------------------------------------------------------
# 3. GEOMETRY, FREQUENCY & LOOP DIRECTIONALITY TESTS
# ---------------------------------------------------------------------------

class TestGeometryAndDirectionality:
    """Verifies ridge frequency estimation, streamline tracing, and Ulnar/Radial loop detection."""

    def test_frequency_estimation_bounds(self, nist_right_loop_path):
        """Frequency estimates must strictly fall within physiological range [0.10, 0.25] cyc/px."""
        img = cv2.imread(nist_right_loop_path, cv2.IMREAD_GRAYSCALE)[:480, :]
        of = compute_orientation_field(img, block_size=16)
        freq_map, med_freq, med_wl = estimate_ridge_frequency(img, of, min_freq=0.10, max_freq=0.25)

        assert 0.10 <= med_freq <= 0.25
        assert 4.0 <= med_wl <= 10.0
        # Foreground points in freq_map
        fg_freqs = freq_map[of.block_mask > 0]
        assert np.all(fg_freqs >= 0.10)
        assert np.all(fg_freqs <= 0.25)
        # Verifies variation across fingerprint
        assert np.std(fg_freqs) > 0.001

    def test_synthetic_frequency_accuracy(self):
        """Synthetic sinusoidal grating with known wavelength 6 px (0.1667 cyc/px) is accurately recovered."""
        # Horizontal ridges with wavelength 6.0 pixels -> frequency = 1 / 6.0 = 0.1667
        y, _ = np.mgrid[0:128, 0:128]
        target_wavelength = 6.0
        target_freq = 1.0 / target_wavelength
        synth = (127.5 * (1 + np.sin(2 * np.pi * y / target_wavelength))).astype(np.uint8)

        of = compute_orientation_field(synth, block_size=16)
        freq_map, med_freq, med_wl = estimate_ridge_frequency(synth, of, min_freq=0.10, max_freq=0.25)
        assert abs(med_freq - target_freq) < 0.02
        assert abs(med_wl - target_wavelength) < 0.8

    def test_streamline_tracing(self, nist_right_loop_path, nist_left_loop_path):
        """Streamlines must be non-empty, stay inside bounds, and trace on both Left and Right loops."""
        for path in [nist_right_loop_path, nist_left_loop_path]:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)[:480, :]
            of = compute_orientation_field(img, block_size=16)
            sp = detect_singular_points(of)
            streamlines = trace_ridge_streamlines(of, singular_points_res=sp, seed_grid_stride=32)

            assert len(streamlines) > 20
            h, w = img.shape
            for sl in streamlines:
                assert sl.ndim == 2 and sl.shape[1] == 2
                assert len(sl) >= 15
                # Points strictly within image coordinates
                assert np.all(sl[:, 0] >= 0.0) and np.all(sl[:, 0] < w)
                assert np.all(sl[:, 1] >= 0.0) and np.all(sl[:, 1] < h)


    def test_loop_directionality_and_anatomical_classification(
        self, nist_right_loop_path, nist_left_loop_path
    ):
        """Tests loop directionality (sloping left vs. right) and Ulnar/Radial anatomical classification."""
        # 1. Right Little (f0002_05) is Right Hand, Digit 05, Right Loop -> Ulnar Loop
        img_r = cv2.imread(nist_right_loop_path, cv2.IMREAD_GRAYSCALE)[:480, :]
        of_r = compute_orientation_field(img_r, block_size=16, sigma_smooth=6.0)
        sp_r = detect_singular_points(of_r)
        geo_r = extract_fingerprint_geometry(img_r, of_r, sp_r, hand="Right", finger_name="Right Little")

        assert geo_r.flow_direction == "right"
        assert geo_r.loop_type_estimate == "right_loop"
        assert geo_r.anatomical_class == "Ulnar Loop"
        assert geo_r.confidence >= 0.70

        # 2. Left Little (f0003_10) is Left Hand, Digit 10, Left Loop -> Ulnar Loop
        img_l = cv2.imread(nist_left_loop_path, cv2.IMREAD_GRAYSCALE)[:480, :]
        of_l = compute_orientation_field(img_l, block_size=16, sigma_smooth=6.0)
        sp_l = detect_singular_points(of_l)
        geo_l = extract_fingerprint_geometry(img_l, of_l, sp_l, hand="Left", finger_name="Left Little")

        assert geo_l.flow_direction == "left"
        assert geo_l.loop_type_estimate == "left_loop"
        assert geo_l.anatomical_class == "Ulnar Loop"
        assert geo_l.confidence >= 0.70

    def test_radial_loop_derivation(self, nist_right_loop_path, nist_left_loop_path):
        """Tests Radial Loop mapping when hand and slope oppose: Right Hand + Left Loop = Radial."""
        img_l = cv2.imread(nist_left_loop_path, cv2.IMREAD_GRAYSCALE)[:480, :]
        of_l = compute_orientation_field(img_l, block_size=16, sigma_smooth=6.0)
        sp_l = detect_singular_points(of_l)
        # If this left-sloping loop were on the Right hand, it would be a Radial Loop
        geo_rad = extract_fingerprint_geometry(img_l, of_l, sp_l, hand="Right", finger_name="Right Index")
        assert geo_rad.anatomical_class == "Radial Loop"


# ---------------------------------------------------------------------------
# 4. EXECUTION LATENCY & THROUGHPUT BENCHMARK (M5 PRO)
# ---------------------------------------------------------------------------

class TestAppleSiliconPerformance:
    """Verifies feature extraction latency and throughput on Apple Silicon."""

    def test_orientation_latency_target(self):
        """Orientation field extraction on 256x256 image must complete in < 25 ms."""
        synth = (np.random.rand(256, 256) * 255).astype(np.uint8)
        # Warmup
        _ = compute_orientation_field(synth, block_size=16)

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = compute_orientation_field(synth, block_size=16)
            times.append(time.perf_counter() - t0)

        mean_ms = float(np.mean(times) * 1000.0)
        assert mean_ms < 25.0, f"Orientation too slow: {mean_ms:.2f} ms > 25 ms"

    def test_end_to_end_latency_target(self):
        """Full feature extraction pipeline on 256x256 image must complete in < 60 ms."""
        synth = (np.random.rand(256, 256) * 255).astype(np.uint8)
        # Warmup
        of = compute_orientation_field(synth, block_size=16)
        sp = detect_singular_points(of)
        _ = extract_fingerprint_geometry(synth, of, sp)

        times = []
        for _ in range(15):
            t0 = time.perf_counter()
            of = compute_orientation_field(synth, block_size=16)
            sp = detect_singular_points(of)
            _ = extract_fingerprint_geometry(synth, of, sp)
            times.append(time.perf_counter() - t0)

        mean_ms = float(np.mean(times) * 1000.0)
        assert mean_ms < 60.0, f"End-to-end feature pipeline too slow: {mean_ms:.2f} ms > 60 ms"
