"""
Unit and Integration Tests for Stage 7: Subtype Classification & 20-Label Engine.
Verifies:
1. 20-label taxonomy definitions, category mappings, and registry integrity.
2. Geometric decision rules with synthetic/mocked feature vectors (Loops, Arches, Whorls, Accidental).
3. Uncertainty handling for missing hand metadata, truncated deltas, and low coherence.
4. End-to-end classification pipeline on real NIST SD04 fingerprint impressions.
"""

import os
import sys
import pytest
import numpy as np
import torch
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.subtypes.taxonomy import (
    PrimaryCategory,
    ForensicVerifiability,
    SubtypeLabel,
    SubtypeDefinition,
    TAXONOMY_REGISTRY,
    get_subtypes_for_category,
    get_subtypes_for_nist_class,
    parse_subtype_label,
    normalize_subtype_str,
    validate_taxonomy_integrity,
)
from src.subtypes.rules import (
    RuleDecision,
    LoopRuleEngine,
    ArchRuleEngine,
    WhorlRuleEngine,
    compute_core_streamline_geometry,
    compute_ridge_count,
)
from src.subtypes.classifier import (
    HierarchicalSubtypeClassifier,
    SubtypeClassificationResult,
)
from src.features.orientation import OrientationResult
from src.features.singular_points import SingularPoint, SingularPointsResult
from src.features.geometry import GeometryResult


# ---------------------------------------------------------------------------
# FIXTURES & MOCK GENERATORS
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_orientation():
    """Generates synthetic 256x256 horizontal orientation result."""
    h, w = 256, 256
    theta = np.zeros((h, w), dtype=np.float32)
    coherence = np.ones((h, w), dtype=np.float32) * 0.85
    mask = np.ones((h, w), dtype=np.uint8)
    vx = np.cos(2 * theta)
    vy = np.sin(2 * theta)
    block_theta = np.zeros((h // 16, w // 16), dtype=np.float32)
    block_coherence = np.ones((h // 16, w // 16), dtype=np.float32) * 0.85
    block_mask = np.ones((h // 16, w // 16), dtype=np.uint8)

    return OrientationResult(
        theta=theta,
        coherence=coherence,
        mask=mask,
        vx=vx,
        vy=vy,
        block_theta=block_theta,
        block_coherence=block_coherence,
        block_mask=block_mask,
        block_size=16,
    )


def create_mock_singular_points(cores=None, deltas=None, delta_truncated=False):
    """Helper to build SingularPointsResult with mock coordinates."""
    cores_list = []
    if cores:
        for x, y in cores:
            cores_list.append(
                SingularPoint(
                    x=float(x), y=float(y), kind="core",
                    confidence=0.88, poincare_index=0.5, filter_response=0.85,
                    is_truncated=False, feature_visible=True,
                )
            )

    deltas_list = []
    if deltas:
        for x, y in deltas:
            deltas_list.append(
                SingularPoint(
                    x=float(x), y=float(y), kind="delta",
                    confidence=0.85, poincare_index=-0.5, filter_response=0.80,
                    is_truncated=False, feature_visible=True,
                )
            )

    return SingularPointsResult(
        cores=cores_list,
        deltas=deltas_list,
        all_points=cores_list + deltas_list,
        poincare_map=np.zeros((16, 16), dtype=np.float32),
        core_response_map=np.zeros((256, 256), dtype=np.float32),
        delta_response_map=np.zeros((256, 256), dtype=np.float32),
        delta_truncated=delta_truncated,
        expected_delta_sector="bottom-right" if delta_truncated else None,
        method="mock",
        curvature_map=np.zeros((256, 256), dtype=np.float32),
    )


def create_mock_geometry(flow_asym=0.5, exit_ang=20.0, flow_dir="right", wavelength=10.0):
    """Helper to build GeometryResult with custom flow asymmetry and exit trajectory."""
    return GeometryResult(
        frequency_map=np.ones((16, 16), dtype=np.float32) * (1.0 / wavelength),
        median_frequency=1.0 / wavelength,
        median_wavelength=wavelength,
        streamlines=[],
        flow_direction=flow_dir,
        loop_type_estimate="right_loop" if flow_asym > 0 else "left_loop",
        flow_asymmetry=flow_asym,
        exit_angle_deg=exit_ang,
        confidence=0.85,
        anatomical_class=None,
    )


# ---------------------------------------------------------------------------
# 1. TAXONOMY SPECIFICATION TESTS
# ---------------------------------------------------------------------------

class TestTaxonomySpecification:
    """Verifies that the 4 categories and 20 labels are strictly defined."""

    def test_taxonomy_integrity(self):
        """Verifies exact count of 20 subtypes and valid category partitioning."""
        assert validate_taxonomy_integrity() is True
        assert len(TAXONOMY_REGISTRY) == 20

    def test_category_counts(self):
        """Ensure Loop(4), Arch(4), Whorl(11), Accidental(1)."""
        loops = get_subtypes_for_category(PrimaryCategory.LOOP)
        arches = get_subtypes_for_category(PrimaryCategory.ARCH)
        whorls = get_subtypes_for_category(PrimaryCategory.WHORL)
        accidentals = get_subtypes_for_category(PrimaryCategory.ACCIDENTAL)

        assert len(loops) == 4, f"Expected 4 loops, got {len(loops)}"
        assert len(arches) == 4, f"Expected 4 arches, got {len(arches)}"
        assert len(whorls) == 11, f"Expected 11 whorls, got {len(whorls)}"
        assert len(accidentals) == 1, f"Expected 1 accidental, got {len(accidentals)}"

    def test_nist_class_compatibility(self):
        """Ensure all 5 NIST pattern classes map to compatible subtypes."""
        for nist_cls in ["A", "T", "W", "L", "R"]:
            compatible = get_subtypes_for_nist_class(nist_cls)
            assert len(compatible) > 0, f"No subtypes mapped for NIST class {nist_cls}"

    def test_label_normalization(self):
        """Verify parsing handles en-dashes, hyphens, and whitespace variations."""
        # En-dash vs hyphen
        dash_str = "Elongated Whorl – Target Centric"
        hyphen_str = "Elongated Whorl - Target Centric"
        assert parse_subtype_label(dash_str) == SubtypeLabel.ELONGATED_WHORL_TARGET
        assert parse_subtype_label(hyphen_str) == SubtypeLabel.ELONGATED_WHORL_TARGET

        # Case insensitivity
        assert parse_subtype_label("simple arch") == SubtypeLabel.SIMPLE_ARCH
        assert parse_subtype_label("ULNAR LOOP") == SubtypeLabel.ULNAR_LOOP
        assert parse_subtype_label("invalid label") is None

    def test_forensic_verifiability_levels(self):
        """Verify every subtype is assigned a valid ForensicVerifiability tier."""
        for label, defn in TAXONOMY_REGISTRY.items():
            assert isinstance(defn.forensic_verifiability, ForensicVerifiability)
            assert defn.min_cores <= defn.max_cores
            assert defn.min_deltas <= defn.max_deltas
            d = defn.to_dict()
            assert "label" in d and "forensic_verifiability" in d


# ---------------------------------------------------------------------------
# 2. GEOMETRIC RULE DISCRIMINATOR TESTS (SYNTHETIC VECTORS)
# ---------------------------------------------------------------------------

class TestGeometricRuleDiscriminators:
    """Verifies geometric logic for Loops, Arches, Whorls, and Accidental."""

    def test_loop_ulnar_vs_radial_right_hand(self, mock_orientation):
        """Right hand: Right slant is Ulnar, Left slant is Radial."""
        sp = create_mock_singular_points(cores=[(128, 100)], deltas=[(80, 180)])

        # Right slant -> Ulnar
        geo_right = create_mock_geometry(flow_asym=0.6, exit_ang=20.0, flow_dir="right")
        dec_u = LoopRuleEngine.evaluate("R", 0.90, mock_orientation, sp, geo_right, hand="Right")
        assert dec_u.subtype == SubtypeLabel.ULNAR_LOOP
        assert dec_u.is_uncertain is False

        # Left slant -> Radial
        geo_left = create_mock_geometry(flow_asym=-0.6, exit_ang=160.0, flow_dir="left")
        dec_r = LoopRuleEngine.evaluate("L", 0.90, mock_orientation, sp, geo_left, hand="Right")
        assert dec_r.subtype == SubtypeLabel.RADIAL_LOOP
        assert dec_r.is_uncertain is False

    def test_loop_ulnar_vs_radial_left_hand(self, mock_orientation):
        """Left hand: Left slant is Ulnar, Right slant is Radial."""
        sp = create_mock_singular_points(cores=[(128, 100)], deltas=[(170, 180)])

        # Left slant -> Ulnar
        geo_left = create_mock_geometry(flow_asym=-0.6, exit_ang=160.0, flow_dir="left")
        dec_u = LoopRuleEngine.evaluate("L", 0.90, mock_orientation, sp, geo_left, hand="Left")
        assert dec_u.subtype == SubtypeLabel.ULNAR_LOOP
        assert dec_u.is_uncertain is False

        # Right slant -> Radial
        geo_right = create_mock_geometry(flow_asym=0.6, exit_ang=20.0, flow_dir="right")
        dec_r = LoopRuleEngine.evaluate("R", 0.90, mock_orientation, sp, geo_right, hand="Left")
        assert dec_r.subtype == SubtypeLabel.RADIAL_LOOP
        assert dec_r.is_uncertain is False

    def test_loop_missing_hand_uncertainty(self, mock_orientation):
        """Without hand metadata, loop must return Uncertain with explicit reason."""
        sp = create_mock_singular_points(cores=[(128, 100)], deltas=[(80, 180)])
        geo = create_mock_geometry(flow_asym=0.6, exit_ang=20.0, flow_dir="right")

        dec = LoopRuleEngine.evaluate("R", 0.90, mock_orientation, sp, geo, hand=None)
        assert dec.is_uncertain is True
        assert "Hand metadata required" in dec.reason

    def test_falling_loop_detection(self, mock_orientation):
        """Steep downward exit angle and high delta elevation triggers Falling Loop."""
        # High delta near core elevation (y_delta - y_core = 30 px < 42 px)
        sp = create_mock_singular_points(cores=[(128, 100)], deltas=[(80, 130)])
        # Downward exit trajectory (exit angle 65 deg)
        geo = create_mock_geometry(flow_asym=0.6, exit_ang=65.0, flow_dir="right")

        dec = LoopRuleEngine.evaluate("R", 0.90, mock_orientation, sp, geo, hand="Right")
        assert dec.subtype == SubtypeLabel.ULNAR_FALLING_LOOP
        assert dec.evidence["is_falling"] is True

    def test_simple_arch_detection(self, mock_orientation):
        """Flat horizontal ridges with zero cores and deltas classify as Simple Arch."""
        sp = create_mock_singular_points(cores=[], deltas=[])
        geo = create_mock_geometry(flow_asym=0.0, exit_ang=0.0, flow_dir="symmetric")

        dec = ArchRuleEngine.evaluate("A", 0.92, mock_orientation, sp, geo, hand="Right")
        assert dec.subtype == SubtypeLabel.SIMPLE_ARCH
        assert dec.is_uncertain is False

    def test_tented_arch_vertical_thrust(self, mock_orientation):
        """Central vertical ridges trigger Tented Arch."""
        # Inject vertical ridge spine in center (|theta - pi/2| < 0.1)
        h, w = mock_orientation.theta.shape
        mock_orientation.theta[int(0.3 * h):int(0.7 * h), int(0.4 * w):int(0.6 * w)] = np.pi / 2.0
        sp = create_mock_singular_points(cores=[], deltas=[])
        geo = create_mock_geometry(flow_asym=0.0, exit_ang=0.0, flow_dir="symmetric")

        dec = ArchRuleEngine.evaluate("T", 0.90, mock_orientation, sp, geo, hand="Right")
        assert dec.subtype == SubtypeLabel.TENTED_ARCH
        assert dec.evidence["vertical_ridge_ratio"] > 0.04

    def test_tented_arch_with_loop(self, mock_orientation):
        """Core and delta with ridge count < 9 classifies as Tented Arch with Loop."""
        # Core at (128, 120), Delta at (110, 160) -> dist = 43.8 px. At wavelength 10 px, rc = 4.38 < 9.
        sp = create_mock_singular_points(cores=[(128, 120)], deltas=[(110, 160)])
        geo = create_mock_geometry(flow_asym=0.3, exit_ang=30.0, flow_dir="right", wavelength=10.0)

        dec = ArchRuleEngine.evaluate("T", 0.88, mock_orientation, sp, geo, hand="Right")
        assert dec.subtype == SubtypeLabel.TENTED_ARCH_ULNAR_LOOP
        assert dec.evidence["has_pseudo_loop"] is True
        assert dec.evidence["ridge_count"] < 9.0

    def test_whorl_double_loop(self, mock_orientation):
        """Two distinct cores with opposing angles classify as Double Loop Whorl."""
        # Core 1 at angle 15 deg, Core 2 at angle 105 deg (ang_diff = 90 deg >= 40 deg)
        mock_orientation.theta[100, 110] = np.deg2rad(15.0)
        mock_orientation.theta[150, 140] = np.deg2rad(105.0)
        sp = create_mock_singular_points(
            cores=[(110, 100), (140, 150)],
            deltas=[(60, 200), (200, 200)]
        )
        geo = create_mock_geometry(flow_asym=0.0, exit_ang=0.0, flow_dir="circular")

        dec = WhorlRuleEngine.evaluate("W", 0.95, mock_orientation, sp, geo, hand="Right")
        assert dec.subtype == SubtypeLabel.DOUBLE_LOOP_WHORL
        assert dec.evidence["num_cores"] == 2
        assert dec.evidence["core_angle_difference_deg"] >= 40.0

    def test_whorl_peacock_eye(self, mock_orientation):
        """Delta proximate to core (<= 45 px) and outer delta >= 55 px triggers Peacock's Eye."""
        # Core at (128, 128), Proximate delta at (105, 145) [dist ~30 px], Outer delta at (220, 210) [dist ~123 px]
        sp = create_mock_singular_points(
            cores=[(128, 128)],
            deltas=[(105, 145), (220, 210)]
        )
        geo = create_mock_geometry(flow_asym=0.1, exit_ang=10.0, flow_dir="circular", wavelength=10.0)

        dec = WhorlRuleEngine.evaluate("W", 0.92, mock_orientation, sp, geo, hand="Right")
        assert "Peacock's Eye" in dec.subtype.value
        assert dec.evidence["proximate_delta_dist_px"] <= 45.0

    def test_whorl_accidental_trigger(self, mock_orientation):
        """>= 3 confident deltas triggers Accidental classification."""
        sp = create_mock_singular_points(
            cores=[(128, 120)],
            deltas=[(60, 180), (190, 180), (130, 210)]
        )
        geo = create_mock_geometry(flow_asym=0.0, exit_ang=0.0, flow_dir="circular")

        dec = WhorlRuleEngine.evaluate("W", 0.90, mock_orientation, sp, geo, hand="Right")
        assert dec.subtype == SubtypeLabel.ACCIDENTAL
        assert dec.evidence["num_deltas"] >= 3

    def test_whorl_sensor_truncation_uncertainty(self, mock_orientation):
        """Delta truncated outside sensor FOV marks Whorl as Uncertain."""
        sp = create_mock_singular_points(
            cores=[(128, 128)],
            deltas=[(60, 180)],
            delta_truncated=True
        )
        geo = create_mock_geometry(flow_asym=0.0, exit_ang=0.0, flow_dir="circular")

        dec = WhorlRuleEngine.evaluate("W", 0.88, mock_orientation, sp, geo, hand="Right")
        assert dec.is_uncertain is True
        assert "delta truncated" in dec.reason.lower()

    def test_falling_loop_upward_angle_safeguard(self, mock_orientation):
        """Left loop with upward exit angle (e.g. -140 deg) must NOT trigger Falling Loop."""
        sp = create_mock_singular_points(cores=[(128, 100)], deltas=[(170, 180)])
        # Upward exit angle (-140 deg, dy < 0)
        geo = create_mock_geometry(flow_asym=-0.6, exit_ang=-140.0, flow_dir="left")

        dec = LoopRuleEngine.evaluate("L", 0.92, mock_orientation, sp, geo, hand="Left")
        assert dec.evidence["is_falling"] is False
        assert dec.evidence["downward_deviation_deg"] == 0.0
        assert dec.subtype == SubtypeLabel.ULNAR_LOOP

    def test_simple_arch_with_border_smudges(self, mock_orientation):
        """Plain arch with peripheral smudge singular points at edge of image must remain Simple Arch."""
        # Core at (18, 18) and Delta at (240, 15) near borders outside central area
        sp = create_mock_singular_points(cores=[(18, 18)], deltas=[(240, 15)])
        geo = create_mock_geometry(flow_asym=0.0, exit_ang=0.0, flow_dir="symmetric")

        dec = ArchRuleEngine.evaluate("A", 0.94, mock_orientation, sp, geo, hand="Right")
        assert dec.subtype == SubtypeLabel.SIMPLE_ARCH
        assert dec.evidence["has_central_singularity"] is False

    def test_tented_arch_with_loop_left_hand(self, mock_orientation):
        """Verify Tented Arch with Loop resolves Ulnar vs. Radial properly on Left Hand."""
        # Recurve with delta on right of core (delta.x > core.x) -> Left slant
        # On Left Hand, Left slant is Ulnar
        sp_u = create_mock_singular_points(cores=[(120, 120)], deltas=[(140, 160)])
        geo = create_mock_geometry(flow_asym=-0.3, exit_ang=150.0, flow_dir="left", wavelength=10.0)
        dec_u = ArchRuleEngine.evaluate("T", 0.88, mock_orientation, sp_u, geo, hand="Left")
        assert dec_u.subtype == SubtypeLabel.TENTED_ARCH_ULNAR_LOOP

        # Recurve with delta on left of core (delta.x < core.x) -> Right slant
        # On Left Hand, Right slant is Radial
        sp_r = create_mock_singular_points(cores=[(140, 120)], deltas=[(110, 160)])
        geo_r = create_mock_geometry(flow_asym=0.3, exit_ang=30.0, flow_dir="right", wavelength=10.0)
        dec_r = ArchRuleEngine.evaluate("T", 0.88, mock_orientation, sp_r, geo_r, hand="Left")
        assert dec_r.subtype == SubtypeLabel.TENTED_ARCH_RADIAL_LOOP

    def test_target_centric_vs_spiral_synthetic_discrimination(self):
        """Synthetic concentric circles vs. Archimedean spiral orientation fields."""
        h, w = 256, 256
        y_g, x_g = np.ogrid[:h, :w]
        dx = x_g - 128.0
        dy = y_g - 128.0
        phi = np.arctan2(dy, dx)

        # Concentric circles: theta = phi + pi/2
        th_concentric = np.mod(phi + np.pi / 2.0, np.pi).astype(np.float32)
        mask = np.ones((h, w), dtype=np.uint8)
        coh = np.ones((h, w), dtype=np.float32) * 0.90
        vx = np.cos(2 * th_concentric)
        vy = np.sin(2 * th_concentric)

        ori_concentric = OrientationResult(
            theta=th_concentric, coherence=coh, mask=mask, vx=vx, vy=vy,
            block_theta=cv2.resize(th_concentric, (16, 16)),
            block_coherence=cv2.resize(coh, (16, 16)),
            block_mask=cv2.resize(mask, (16, 16)),
            block_size=16,
        )

        sp = create_mock_singular_points(cores=[(128, 128)], deltas=[(60, 200), (200, 200)])
        geo = create_mock_geometry(flow_asym=0.0, exit_ang=0.0, flow_dir="circular", wavelength=10.0)

        dec_c = WhorlRuleEngine.evaluate("W", 0.94, ori_concentric, sp, geo, hand="Right")
        assert dec_c.subtype == SubtypeLabel.TARGET_CENTRIC_WHORL
        assert dec_c.evidence["core_geom"]["is_spiral"] is False

        # Open Spiral: theta = phi + pi/2 + 0.38 rad
        th_spiral = np.mod(phi + np.pi / 2.0 + 0.38, np.pi).astype(np.float32)
        vx_s = np.cos(2 * th_spiral)
        vy_s = np.sin(2 * th_spiral)
        ori_spiral = OrientationResult(
            theta=th_spiral, coherence=coh, mask=mask, vx=vx_s, vy=vy_s,
            block_theta=cv2.resize(th_spiral, (16, 16)),
            block_coherence=cv2.resize(coh, (16, 16)),
            block_mask=cv2.resize(mask, (16, 16)),
            block_size=16,
        )

        dec_s = WhorlRuleEngine.evaluate("W", 0.94, ori_spiral, sp, geo, hand="Right")
        assert dec_s.subtype == SubtypeLabel.SPIRAL_WHORL
        assert dec_s.evidence["core_geom"]["is_spiral"] is True

    def test_elongated_whorl_synthetic(self):
        """Synthetic stretched elliptical concentric whorl classifies as Elongated Whorl."""
        h, w = 256, 256
        y_g, x_g = np.ogrid[:h, :w]
        a = 1.8
        dx = (x_g - 128.0)
        dy = (y_g - 128.0)
        th_ellipse = np.mod(np.arctan2(dx / (a**2), -dy), np.pi).astype(np.float32)
        mask = np.ones((h, w), dtype=np.uint8)
        coh = np.ones((h, w), dtype=np.float32) * 0.90
        vx = np.cos(2 * th_ellipse)
        vy = np.sin(2 * th_ellipse)

        ori_ellipse = OrientationResult(
            theta=th_ellipse, coherence=coh, mask=mask, vx=vx, vy=vy,
            block_theta=cv2.resize(th_ellipse, (16, 16)),
            block_coherence=cv2.resize(coh, (16, 16)),
            block_mask=cv2.resize(mask, (16, 16)),
            block_size=16,
        )
        sp = create_mock_singular_points(cores=[(128, 128)], deltas=[(60, 200), (200, 200)])
        geo = create_mock_geometry(flow_asym=0.0, exit_ang=0.0, flow_dir="circular", wavelength=10.0)

        dec = WhorlRuleEngine.evaluate("W", 0.92, ori_ellipse, sp, geo, hand="Right")
        assert dec.subtype in (SubtypeLabel.ELONGATED_WHORL_TARGET, SubtypeLabel.ELONGATED_WHORL_SPIRAL)
        assert dec.evidence["core_geom"]["is_elongated"] is True

    def test_crease_pseudo_core_suppressed_for_elongated_whorl(self):
        """Artifact pseudo-core in crease / below deltas is pruned, classifying as Elongated Whorl Target Centric."""
        h, w = 256, 256
        y_g, x_g = np.ogrid[:h, :w]
        a = 1.8
        dx = (x_g - 128.0)
        dy = (y_g - 110.0)
        th_ellipse = np.mod(np.arctan2(dx / (a**2), -dy), np.pi).astype(np.float32)
        # Set opposing angle at the pseudo-core location (230, 195) so without pruning it would trigger Double Loop
        th_ellipse[230, 195] = np.deg2rad(90.0)
        mask = np.ones((h, w), dtype=np.uint8)
        coh = np.ones((h, w), dtype=np.float32) * 0.90
        vx = np.cos(2 * th_ellipse)
        vy = np.sin(2 * th_ellipse)

        ori_ellipse = OrientationResult(
            theta=th_ellipse, coherence=coh, mask=mask, vx=vx, vy=vy,
            block_theta=cv2.resize(th_ellipse, (16, 16)),
            block_coherence=cv2.resize(coh, (16, 16)),
            block_mask=cv2.resize(mask, (16, 16)),
            block_size=16,
        )
        # Setup: 1 central core at (128, 110), 2 deltas at (60, 180) and (200, 185)
        # 1 artifact pseudo-core near bottom crease / delta at (195, 230) (below deltas / in bottom 20%)
        sp = create_mock_singular_points(
            cores=[(128, 110), (195, 230)],
            deltas=[(60, 180), (200, 185)]
        )
        geo = create_mock_geometry(flow_asym=0.0, exit_ang=0.0, flow_dir="circular", wavelength=10.0)

        dec = WhorlRuleEngine.evaluate("W", 0.871, ori_ellipse, sp, geo, hand="Right")
        assert dec.subtype != SubtypeLabel.DOUBLE_LOOP_WHORL
        assert dec.subtype == SubtypeLabel.ELONGATED_WHORL_TARGET
        assert dec.evidence["core_geom"]["is_elongated"] is True

    def test_crease_pseudo_core_asymmetric_confidence_not_starving_central_core(self):
        """Asymmetric confidence (high conf crease artifact, lower conf central core) preserves central core."""
        h, w = 256, 256
        y_g, x_g = np.ogrid[:h, :w]
        a = 1.8
        dx = (x_g - 128.0)
        dy = (y_g - 110.0)
        th_ellipse = np.mod(np.arctan2(dx / (a**2), -dy), np.pi).astype(np.float32)
        th_ellipse[230, 195] = np.deg2rad(90.0)
        mask = np.ones((h, w), dtype=np.uint8)
        coh = np.ones((h, w), dtype=np.float32) * 0.90
        vx = np.cos(2 * th_ellipse)
        vy = np.sin(2 * th_ellipse)

        ori_ellipse = OrientationResult(
            theta=th_ellipse, coherence=coh, mask=mask, vx=vx, vy=vy,
            block_theta=cv2.resize(th_ellipse, (16, 16)),
            block_coherence=cv2.resize(coh, (16, 16)),
            block_mask=cv2.resize(mask, (16, 16)),
            block_size=16,
        )
        c_true = SingularPoint(x=128, y=110, kind="core", confidence=0.58, poincare_index=0.5, filter_response=0.5, is_truncated=False, feature_visible=True)
        c_crease = SingularPoint(x=195, y=230, kind="core", confidence=0.88, poincare_index=0.5, filter_response=0.5, is_truncated=False, feature_visible=True)
        d1 = SingularPoint(x=60, y=180, kind="delta", confidence=0.80, poincare_index=-0.5, filter_response=0.5, is_truncated=False, feature_visible=True)
        d2 = SingularPoint(x=200, y=185, kind="delta", confidence=0.80, poincare_index=-0.5, filter_response=0.5, is_truncated=False, feature_visible=True)

        sp = SingularPointsResult(
            cores=[c_crease, c_true],
            deltas=[d1, d2],
            all_points=[c_crease, c_true, d1, d2],
            poincare_map=np.zeros((16, 16)),
            core_response_map=np.zeros((h, w)),
            delta_response_map=np.zeros((h, w)),
            delta_truncated=False,
            expected_delta_sector=None,
            method="hybrid",
        )
        geo = create_mock_geometry(flow_asym=0.0, exit_ang=0.0, flow_dir="circular", wavelength=10.0)

        dec = WhorlRuleEngine.evaluate("W", 0.871, ori_ellipse, sp, geo, hand="Right")
        assert dec.subtype == SubtypeLabel.ELONGATED_WHORL_TARGET
        assert dec.evidence["num_cores"] == 1
        assert dec.evidence["core_geom"]["is_elongated"] is True

    def test_double_loop_with_crease_artifact_and_split_confidence(self):
        """Genuine double loop with cores at 0.85 and 0.58 plus a crease core at 0.75 correctly identifies double loop."""
        h, w = 256, 256
        ori = OrientationResult(
            theta=np.zeros((h, w), dtype=np.float32), coherence=np.ones((h, w), dtype=np.float32),
            mask=np.ones((h, w), dtype=np.uint8), vx=np.ones((h, w), dtype=np.float32), vy=np.zeros((h, w), dtype=np.float32),
            block_theta=np.zeros((16, 16), dtype=np.float32), block_coherence=np.ones((16, 16), dtype=np.float32),
            block_mask=np.ones((16, 16), dtype=np.uint8), block_size=16
        )
        ori.theta[100, 110] = np.deg2rad(15.0)
        ori.theta[150, 140] = np.deg2rad(105.0)

        c1 = SingularPoint(x=110, y=100, kind="core", confidence=0.85, poincare_index=0.5, filter_response=0.5, is_truncated=False, feature_visible=True)
        c2 = SingularPoint(x=140, y=150, kind="core", confidence=0.58, poincare_index=0.5, filter_response=0.5, is_truncated=False, feature_visible=True)
        c_crease = SingularPoint(x=195, y=230, kind="core", confidence=0.75, poincare_index=0.5, filter_response=0.5, is_truncated=False, feature_visible=True)
        d1 = SingularPoint(x=60, y=200, kind="delta", confidence=0.80, poincare_index=-0.5, filter_response=0.5, is_truncated=False, feature_visible=True)
        d2 = SingularPoint(x=200, y=200, kind="delta", confidence=0.80, poincare_index=-0.5, filter_response=0.5, is_truncated=False, feature_visible=True)

        sp = SingularPointsResult(
            cores=[c1, c_crease, c2],
            deltas=[d1, d2],
            all_points=[c1, c_crease, c2, d1, d2],
            poincare_map=np.zeros((16, 16)),
            core_response_map=np.zeros((h, w)),
            delta_response_map=np.zeros((h, w)),
            delta_truncated=False,
            expected_delta_sector=None,
            method="hybrid",
        )
        geo = create_mock_geometry(flow_asym=0.0, exit_ang=0.0, flow_dir="circular")
        dec = WhorlRuleEngine.evaluate("W", 0.95, ori, sp, geo, hand="Right")
        assert dec.subtype == SubtypeLabel.DOUBLE_LOOP_WHORL

    def test_double_loop_multi_candidate_pair_selection(self):
        """When 3 candidate cores exist, double loop pair search correctly matches opposing pair."""
        h, w = 256, 256
        ori = OrientationResult(
            theta=np.zeros((h, w), dtype=np.float32), coherence=np.ones((h, w), dtype=np.float32),
            mask=np.ones((h, w), dtype=np.uint8), vx=np.ones((h, w), dtype=np.float32), vy=np.zeros((h, w), dtype=np.float32),
            block_theta=np.zeros((16, 16), dtype=np.float32), block_coherence=np.ones((16, 16), dtype=np.float32),
            block_mask=np.ones((16, 16), dtype=np.uint8), block_size=16
        )
        ori.theta[100, 110] = np.deg2rad(15.0)
        ori.theta[102, 112] = np.deg2rad(15.0)
        ori.theta[150, 140] = np.deg2rad(105.0)

        c0 = SingularPoint(x=110, y=100, kind="core", confidence=0.88, poincare_index=0.5, filter_response=0.5, is_truncated=False, feature_visible=True)
        c_neighbor = SingularPoint(x=112, y=102, kind="core", confidence=0.85, poincare_index=0.5, filter_response=0.5, is_truncated=False, feature_visible=True)
        c_opposing = SingularPoint(x=140, y=150, kind="core", confidence=0.80, poincare_index=0.5, filter_response=0.5, is_truncated=False, feature_visible=True)
        d1 = SingularPoint(x=60, y=200, kind="delta", confidence=0.80, poincare_index=-0.5, filter_response=0.5, is_truncated=False, feature_visible=True)
        d2 = SingularPoint(x=200, y=200, kind="delta", confidence=0.80, poincare_index=-0.5, filter_response=0.5, is_truncated=False, feature_visible=True)

        sp = SingularPointsResult(
            cores=[c0, c_neighbor, c_opposing],
            deltas=[d1, d2],
            all_points=[c0, c_neighbor, c_opposing, d1, d2],
            poincare_map=np.zeros((16, 16)),
            core_response_map=np.zeros((h, w)),
            delta_response_map=np.zeros((h, w)),
            delta_truncated=False,
            expected_delta_sector=None,
            method="hybrid",
        )
        geo = create_mock_geometry(flow_asym=0.0, exit_ang=0.0, flow_dir="circular")
        dec = WhorlRuleEngine.evaluate("W", 0.95, ori, sp, geo, hand="Right")
        assert dec.subtype == SubtypeLabel.DOUBLE_LOOP_WHORL


# ---------------------------------------------------------------------------
# 3. END-TO-END INTEGRATION TESTS ON REAL NIST SAMPLE IMAGES
# ---------------------------------------------------------------------------

class TestHierarchicalClassifierEndToEnd:
    """End-to-end inference on real NIST SD04 prints and validation of dictionary output."""

    @pytest.fixture(scope="module")
    def classifier(self):
        return HierarchicalSubtypeClassifier(model_checkpoint="models/best_model.pt")

    def test_classifier_output_schema(self, classifier):
        """Verify returned dataclass and to_dict() schema."""
        img_path = "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0044_03.png"
        res = classifier.classify(img_path, hand="Right", finger_num=3, digit_code="03")

        assert isinstance(res, SubtypeClassificationResult)
        assert res.primary_class == "R"
        assert res.subtype == SubtypeLabel.ULNAR_LOOP
        assert res.status == "Resolved"
        assert res.is_uncertain is False
        assert 0.0 <= res.confidence <= 1.0

        d = res.to_dict(include_features=True)
        assert "primary_class" in d
        assert "subtype" in d
        assert "status" in d
        assert "is_uncertain" in d
        assert "confidence" in d
        assert "reason" in d
        assert "forensic_verifiability" in d
        assert "evidence_summary" in d
        assert "cores" in d
        assert "deltas" in d

    @pytest.mark.parametrize("img_rel,hand,exp_cat", [
        ("Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0012_01.png", "Right", PrimaryCategory.ARCH),
        ("Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/s0021_07.png", "Left", PrimaryCategory.ARCH),
        ("Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0044_03.png", "Right", PrimaryCategory.LOOP),
        ("Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0061_08.png", "Left", PrimaryCategory.WHORL),
    ])
    def test_real_nist_category_resolution(self, classifier, img_rel, hand, exp_cat):
        """Verify robust classification across diverse NIST samples."""
        res = classifier.classify(img_rel, hand=hand)
        assert res.primary_category == exp_cat
        assert res.subtype in TAXONOMY_REGISTRY
        assert res.confidence > 0.40

    def test_headless_fallback_inference(self):
        """Verify headless classifier infers pattern category from geometry when model is None."""
        headless = HierarchicalSubtypeClassifier(model=None, model_checkpoint=None)
        arch_path = "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0012_01.png"
        res = headless.classify(arch_path, hand="Right")
        assert res.primary_category == PrimaryCategory.ARCH
        assert res.subtype in (SubtypeLabel.SIMPLE_ARCH, SubtypeLabel.TENTED_ARCH)

    def test_conflicting_evidence_uncertainty(self, classifier):
        """Verify conflicting neural network vs. geometric evidence activates uncertainty safeguard."""
        # An arch image evaluated with artificial neural override or severe conflict
        socofing_path = "Fingerprint Data/SOCOFing/Real/100__M_Left_index_finger.BMP"
        res = classifier.classify(socofing_path, hand="Left")
        assert res.is_uncertain is True
        assert res.status == "Uncertain"

    def test_socofing_truncation_uncertainty(self, classifier):
        """Verify SOCOFing touch-sensor print with truncated delta triggers Uncertain status."""
        socofing_path = "Fingerprint Data/SOCOFing/Real/100__M_Left_index_finger.BMP"
        res = classifier.classify(socofing_path, hand="Left")
        assert res.is_uncertain is True
        assert res.status == "Uncertain"
        assert res.reason is not None
        assert "truncated" in res.reason.lower() or "confidence" in res.reason.lower()

    def test_batch_classification(self, classifier):
        """Verify batch inference consistency."""
        batch_paths = [
            "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0012_01.png",
            "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0044_03.png",
        ]
        meta = [{"hand": "Right"}, {"hand": "Right"}]
        results = classifier.classify_batch(batch_paths, metadata_list=meta)

        assert len(results) == 2
        assert results[0].primary_category == PrimaryCategory.ARCH
        assert results[1].primary_category == PrimaryCategory.LOOP
