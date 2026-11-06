"""
Singular Point Extraction (Core and Delta) for Fingerprints.
Implements:
1. Poincaré Index method on smoothed orientation fields.
2. Complex phase symmetry filter (Nilsson & Bigun) and orientation curvature methods.
3. Non-maximum suppression (NMS), dipole cancellation, and spatial clustering.
4. Calibrated Confidence Scoring [0.0, 1.0] for every detected point.
5. Detection truncation flags (is_truncated, feature_visible, delta_truncated).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np

from .orientation import OrientationResult, compute_orientation_field, compute_orientation_curvature


@dataclass
class SingularPoint:
    """
    Representation of an individual fingerprint singular point (Core or Delta).
    """
    x: float                     # X coordinate in image pixel space (column)
    y: float                     # Y coordinate in image pixel space (row)
    kind: str                    # 'core', 'delta', or 'whorl_core'
    confidence: float            # Calibrated confidence score in [0.0, 1.0]
    poincare_index: float        # Measured Poincaré index (+0.5, -0.5, +1.0)
    filter_response: float       # Complex symmetry filter response magnitude
    is_truncated: bool           # True if point lies within sensor margin / truncated boundary
    feature_visible: bool        # True if point is reliably inside active impression area
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": float(round(self.x, 2)),
            "y": float(round(self.y, 2)),
            "kind": self.kind,
            "confidence": float(round(self.confidence, 4)),
            "poincare_index": float(round(self.poincare_index, 3)),
            "filter_response": float(round(self.filter_response, 4)),
            "is_truncated": self.is_truncated,
            "feature_visible": self.feature_visible,
        }


@dataclass
class SingularPointsResult:
    """
    Container for all detected singular points and diagnostic response maps.
    """
    cores: List[SingularPoint]
    deltas: List[SingularPoint]
    all_points: List[SingularPoint]
    poincare_map: np.ndarray       # 2D block Poincaré index map (H_b, W_b)
    core_response_map: np.ndarray  # Complex phase filter core response (H, W)
    delta_response_map: np.ndarray # Complex phase filter delta response (H, W)
    delta_truncated: bool          # Flag if loop core exists but delta is truncated outside sensor
    expected_delta_sector: Optional[str] # Predicted truncation sector (e.g. 'bottom-right')
    method: str                    # 'poincare', 'complex_filter', 'hybrid', or 'curvature'
    curvature_map: Optional[np.ndarray] = None # Optional orientation curvature map (H, W)

    @property
    def num_cores(self) -> int:
        return len(self.cores)

    @property
    def num_deltas(self) -> int:
        return len(self.deltas)

    def core_coords(self) -> List[Tuple[float, float]]:
        return [(p.x, p.y) for p in self.cores]

    def delta_coords(self) -> List[Tuple[float, float]]:
        return [(p.x, p.y) for p in self.deltas]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "num_cores": self.num_cores,
            "num_deltas": self.num_deltas,
            "cores": [c.to_dict() for c in self.cores],
            "deltas": [d.to_dict() for d in self.deltas],
            "delta_truncated": self.delta_truncated,
            "expected_delta_sector": self.expected_delta_sector,
        }


def compute_poincare_index_map(
    block_theta: np.ndarray,
    block_mask: Optional[np.ndarray] = None,
    radius: int = 1,
) -> np.ndarray:
    """
    Compute Poincaré index map on a 2D orientation field.
    Integrates orientation differences along a counter-clockwise perimeter.

    Sign convention:
    - Core: +0.5
    - Delta: -0.5
    - Double Core / Concentric Whorl: +1.0
    """
    gh, gw = block_theta.shape
    poincare_map = np.zeros((gh, gw), dtype=np.float32)

    if gh < 3 or gw < 3:
        return poincare_map

    # Define perimeter offset sequence in counter-clockwise order:
    # East -> North-East -> North -> North-West -> West -> South-West -> South -> South-East
    if radius == 1:
        # Standard 8-neighborhood
        nbrs = [
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
            (0, -1), (1, -1), (1, 0), (1, 1),
        ]
    elif radius == 2:
        # 16-neighborhood on 5x5 square perimeter
        nbrs = [
            (0, 2), (-1, 2), (-2, 2), (-2, 1), (-2, 0), (-2, -1), (-2, -2),
            (-1, -2), (0, -2), (1, -2), (2, -2), (2, -1), (2, 0), (2, 1),
            (2, 2), (1, 2)
        ]
    else:
        raise ValueError(f"Unsupported Poincaré neighborhood radius: {radius}")

    n_pts = len(nbrs)
    r_margin = radius
    c_margin = radius

    # Vectorized / inner loop traversal
    for r in range(r_margin, gh - r_margin):
        for c in range(c_margin, gw - c_margin):
            if block_mask is not None and not block_mask[r, c]:
                continue

            sum_diff = 0.0
            for k in range(n_pts):
                dr1, dc1 = nbrs[k]
                dr2, dc2 = nbrs[(k + 1) % n_pts]
                t1 = block_theta[r + dr1, c + dc1]
                t2 = block_theta[r + dr2, c + dc2]
                # Acute difference in [-pi/2, pi/2)
                diff = (t2 - t1 + np.pi / 2.0) % np.pi - np.pi / 2.0
                sum_diff += diff

            # Note the negation: counter-clockwise in image coordinates (where row is down)
            # gives negative sum for cores; negating maps Core to +0.5, Delta to -0.5.
            poincare_map[r, c] = -sum_diff / (2.0 * np.pi)

    return poincare_map


def compute_complex_phase_filters(
    theta: np.ndarray,
    mask: Optional[np.ndarray] = None,
    sigma: float = 8.0,
    ksize: int = 31,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute complex rotational symmetry filter responses (Nilsson & Bigun method).
    - Core matched filter: (x - i y) * exp(-(x^2+y^2)/(2*sigma^2))
    - Delta matched filter: (x + i y) * exp(-(x^2+y^2)/(2*sigma^2))

    Returns:
        Tuple[np.ndarray, np.ndarray]: (core_response_map, delta_response_map) in [0.0, 1.0].
    """
    h, w = theta.shape
    k_half = ksize // 2
    y, x = np.mgrid[-k_half:k_half + 1, -k_half:k_half + 1]
    gauss = np.exp(-(x**2 + y**2) / (2.0 * sigma**2)).astype(np.float32)

    # Core filter: conjugate of exp(i*phi) -> (x - i y)
    h_core_re = (x * gauss).astype(np.float32)
    h_core_im = (-y * gauss).astype(np.float32)

    # Delta filter: conjugate of exp(-i*phi) -> (x + i y)
    h_delta_re = (x * gauss).astype(np.float32)
    h_delta_im = (y * gauss).astype(np.float32)

    # Double-angle representation of orientation field
    z_re = np.cos(2.0 * theta).astype(np.float32)
    z_im = np.sin(2.0 * theta).astype(np.float32)

    # Convolve: (z_re + i*z_im) * (h_re + i*h_im)
    # Core response
    r_core_re = cv2.filter2D(z_re, -1, h_core_re) - cv2.filter2D(z_im, -1, h_core_im)
    r_core_im = cv2.filter2D(z_re, -1, h_core_im) + cv2.filter2D(z_im, -1, h_core_re)
    resp_core = np.sqrt(r_core_re**2 + r_core_im**2)

    # Delta response
    r_delta_re = cv2.filter2D(z_re, -1, h_delta_re) - cv2.filter2D(z_im, -1, h_delta_im)
    r_delta_im = cv2.filter2D(z_re, -1, h_delta_im) + cv2.filter2D(z_im, -1, h_delta_re)
    resp_delta = np.sqrt(r_delta_re**2 + r_delta_im**2)

    if mask is not None:
        m = (mask > 0).astype(np.float32)
        resp_core = resp_core * m
        resp_delta = resp_delta * m

    # Theoretical maximum filter response for ideal singularity:
    # norm_factor = sum(||(x, y)|| * gauss)
    norm_factor = float(np.sum(np.hypot(x, y) * gauss))
    norm_factor = max(norm_factor, 1e-6)

    resp_core = np.clip(resp_core / norm_factor, 0.0, 1.0)
    resp_delta = np.clip(resp_delta / norm_factor, 0.0, 1.0)

    return resp_core.astype(np.float32), resp_delta.astype(np.float32)


def compute_singular_point_confidence(
    x: float,
    y: float,
    kind: str,
    poincare_val: float,
    filter_response: float,
    coherence_map: np.ndarray,
    dist_to_boundary: np.ndarray,
    r_inner: int = 8,
    r_outer: int = 24,
    method: str = "hybrid",
) -> Tuple[float, bool]:
    """
    Calculate calibrated confidence score in [0.0, 1.0] and truncation flag.
    Combines:
    1. Poincaré index fidelity (target 0.5 or 1.0)
    2. Complex phase filter response
    3. Annular ridge coherence (ridges around true singularity have high coherence)
    4. Distance to mask/sensor boundary
    """
    h, w = coherence_map.shape
    px = int(np.clip(round(x), 0, w - 1))
    py = int(np.clip(round(y), 0, h - 1))

    # 1. Poincaré fidelity
    abs_p = abs(poincare_val)
    if kind == "whorl_core":
        p_fidelity = max(0.0, 1.0 - abs(abs_p - 1.0) * 2.0)
    else:
        p_fidelity = max(0.0, 1.0 - abs(abs_p - 0.5) * 2.0)

    # 2. Filter response fidelity
    f_score = float(np.clip(filter_response, 0.0, 1.0))

    # 3. Annular coherence
    # True singular points are surrounded by an annular zone of well-defined parallel ridges
    y_min, y_max = max(0, py - r_outer), min(h, py + r_outer + 1)
    x_min, x_max = max(0, px - r_outer), min(w, px + r_outer + 1)
    yy, xx = np.ogrid[y_min - py : y_max - py, x_min - px : x_max - px]
    dist_sq = xx**2 + yy**2
    annular_mask = (dist_sq >= r_inner**2) & (dist_sq <= r_outer**2)

    sub_coh = coherence_map[y_min:y_max, x_min:x_max]
    if np.any(annular_mask):
        coh_score = float(np.mean(sub_coh[annular_mask]))
    else:
        coh_score = float(coherence_map[py, px])
    coh_score = float(np.clip(coh_score, 0.0, 1.0))

    # 4. Boundary clearance
    dist_px = float(dist_to_boundary[py, px])
    boundary_score = float(np.clip(dist_px / 24.0, 0.0, 1.0))

    # Truncation: point is near sensor margin or image boundary
    is_truncated = (dist_px < 16.0) or (px < 16) or (px > w - 16) or (py < 16) or (py > h - 16)

    # Method-tailored calibrated confidence combination
    if method == "poincare":
        confidence = 0.50 * p_fidelity + 0.30 * coh_score + 0.20 * boundary_score
    elif method in ("complex_filter", "curvature"):
        confidence = 0.55 * f_score + 0.30 * coh_score + 0.15 * boundary_score
    else:
        # Hybrid fusion
        base_conf = (
            0.30 * p_fidelity
            + 0.35 * f_score
            + 0.20 * coh_score
            + 0.15 * boundary_score
        )
        # Penalize candidate if rotational symmetry response is weak
        symmetry_gate = min(1.0, f_score / 0.20) if f_score < 0.20 else 1.0
        confidence = base_conf * symmetry_gate

    confidence = float(np.clip(confidence, 0.0, 1.0))

    return confidence, is_truncated


def cluster_singular_points(
    points: List[Tuple[float, float, float, float]],  # (x, y, poincare, filter_resp)
    kind: str,
    radius: float = 24.0,
) -> List[Tuple[float, float, float, float]]:
    """
    Cluster adjacent / duplicate singular points using response-weighted centroiding.
    """
    if not points:
        return []

    clusters = []
    used = [False] * len(points)

    for i in range(len(points)):
        if used[i]:
            continue

        group = [points[i]]
        used[i] = True

        for j in range(i + 1, len(points)):
            if not used[j]:
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                if np.hypot(dx, dy) <= radius:
                    group.append(points[j])
                    used[j] = True

        # Weighted centroid using filter response + poincare magnitude as weight
        weights = [max(p[3], 0.1) * abs(p[2]) for p in group]
        tot_w = sum(weights)
        cx = sum(p[0] * w for p, w in zip(group, weights)) / tot_w
        cy = sum(p[1] * w for p, w in zip(group, weights)) / tot_w
        mean_p = sum(p[2] * w for p, w in zip(group, weights)) / tot_w
        max_f = max(p[3] for p in group)

        clusters.append((cx, cy, mean_p, max_f))

    return clusters


def cancel_core_delta_dipoles(
    cores: List[Tuple[float, float, float, float]],
    deltas: List[Tuple[float, float, float, float]],
    dipole_distance: float = 36.0,
) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]]]:
    """
    Cancel spurious Core-Delta dipoles caused by ridge creases, scars, or local noise breaks.
    """
    filtered_cores = []
    for c in cores:
        is_dipole = False
        for d in deltas:
            if np.hypot(c[0] - d[0], c[1] - d[1]) <= dipole_distance:
                is_dipole = True
                break
        if not is_dipole:
            filtered_cores.append(c)

    filtered_deltas = []
    for d in deltas:
        is_dipole = False
        for c in cores:
            if np.hypot(d[0] - c[0], d[1] - c[1]) <= dipole_distance:
                is_dipole = True
                break
        if not is_dipole:
            filtered_deltas.append(d)

    return filtered_cores, filtered_deltas


def predict_delta_truncation(
    cores: List[SingularPoint],
    deltas: List[SingularPoint],
    theta_field: np.ndarray,
    mask: np.ndarray,
) -> Tuple[bool, Optional[str]]:
    """
    Detect if a live-scan print has a truncated delta outside the sensor glass.
    If 1 core exists (standard loop) but 0 deltas are inside the impression area,
    projects the exit ridge flow to identify expected delta sector.
    Also handles whorls with truncated deltas (>= 2 cores, < 2 deltas).
    """
    # 1. Whorl with truncated deltas (expects 2 deltas)
    if len(cores) >= 2 and len(deltas) < 2:
        cx = float(np.mean([c.x for c in cores]))
        if len(deltas) == 1:
            # One delta detected: the missing delta is on the opposite side
            return True, "bottom-right" if deltas[0].x < cx else "bottom-left"
        else:
            return True, "bilateral-bottom"

    # 2. Standard Loop with missing delta (expects 1 delta)
    if len(cores) != 1 or len(deltas) > 0:
        return False, None

    core = cores[0]
    h, w = mask.shape
    cx, cy = int(round(core.x)), int(round(core.y))

    # Examine flow asymmetry below the core (rows cy+8 to cy+60)
    y_start = min(h - 1, cy + 8)
    y_end = min(h, cy + 60)
    if y_end <= y_start:
        return True, "bottom"

    sub_theta = theta_field[y_start:y_end, max(0, cx - 40) : min(w, cx + 40)]
    if sub_theta.size == 0:
        return True, "bottom"

    # Evaluate orientation slope:
    # Right loops slope down to the right (positive dx/dy) -> Delta on lower-left
    # Left loops slope down to the left (negative dx/dy) -> Delta on lower-right
    sin2 = np.sin(2.0 * sub_theta)
    mean_sin2 = float(np.mean(sin2))

    if mean_sin2 > 0.15:
        return True, "bottom-left"
    elif mean_sin2 < -0.15:
        return True, "bottom-right"
    else:
        return True, "bottom"


def detect_singular_points(
    orientation_res: OrientationResult,
    method: str = "hybrid",  # 'hybrid', 'poincare', 'complex_filter'
    block_threshold: float = 0.38,
    min_confidence: float = 0.30,
    cluster_radius: float = 24.0,
    dipole_radius: float = 36.0,
) -> SingularPointsResult:
    """
    Extract singular points (cores and deltas) with confidence scoring and truncation flags.

    Args:
        orientation_res: OrientationResult from compute_orientation_field.
        method: Detection engine ('hybrid', 'poincare', 'complex_filter').
        block_threshold: Minimum absolute Poincaré index to trigger candidate detection.
        min_confidence: Minimum calibrated confidence [0.0, 1.0] to accept candidate.
        cluster_radius: Euclidean pixel radius for non-maximum suppression / clustering.
        dipole_radius: Pixel distance within which core-delta dipoles are cancelled.

    Returns:
        SingularPointsResult containing verified cores, deltas, maps, and truncation flags.
    """
    h, w = orientation_res.theta.shape
    block_size = orientation_res.block_size
    b_th = orientation_res.block_theta
    b_mask = orientation_res.block_mask

    # Distance transform from mask boundary to penalize points near edges
    dist_to_boundary = cv2.distanceTransform(orientation_res.mask, cv2.DIST_L2, 5)

    # Compute complex phase filter responses across full image
    resp_core, resp_delta = compute_complex_phase_filters(
        orientation_res.theta,
        mask=orientation_res.mask,
        sigma=8.0,
        ksize=31,
    )

    # Erode block mask slightly to eliminate peripheral boundary noise
    eroded_b_mask = cv2.erode(
        b_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )

    # Compute curvature map across full image
    curvature_map = compute_orientation_curvature(orientation_res.theta, mask=orientation_res.mask)

    # Compute Poincaré Index Map
    poincare_map = compute_poincare_index_map(b_th, block_mask=eroded_b_mask, radius=1)
    gh, gw = poincare_map.shape

    raw_cores = []
    raw_deltas = []

    if method == "poincare":
        # Pure Poincaré candidate detection (no rotational symmetry gating)
        for r in range(1, gh - 1):
            for c in range(1, gw - 1):
                if not eroded_b_mask[r, c]:
                    continue

                idx_val = float(poincare_map[r, c])
                px = float(c * block_size + block_size // 2)
                py = float(r * block_size + block_size // 2)
                ix, iy = int(round(px)), int(round(py))

                if idx_val >= block_threshold:
                    f_resp = float(resp_core[min(h - 1, iy), min(w - 1, ix)])
                    raw_cores.append((px, py, idx_val, f_resp))
                elif idx_val <= -block_threshold:
                    f_resp = float(resp_delta[min(h - 1, iy), min(w - 1, ix)])
                    raw_deltas.append((px, py, idx_val, f_resp))

    elif method == "hybrid":
        # Hybrid fusion: Poincaré topological candidate gated by complex rotational symmetry filter
        for r in range(1, gh - 1):
            for c in range(1, gw - 1):
                if not eroded_b_mask[r, c]:
                    continue

                idx_val = float(poincare_map[r, c])
                px = float(c * block_size + block_size // 2)
                py = float(r * block_size + block_size // 2)
                ix, iy = int(round(px)), int(round(py))

                if idx_val >= block_threshold:
                    f_resp = float(resp_core[min(h - 1, iy), min(w - 1, ix)])
                    # True core requires confirmed rotational symmetry
                    if f_resp >= 0.20:
                        raw_cores.append((px, py, idx_val, f_resp))
                elif idx_val <= -block_threshold:
                    f_resp = float(resp_delta[min(h - 1, iy), min(w - 1, ix)])
                    # True delta requires confirmed triradius symmetry
                    if f_resp >= 0.18:
                        raw_deltas.append((px, py, idx_val, f_resp))

    elif method == "complex_filter":
        # Pure complex filter local maxima detection
        dil_c = cv2.dilate(resp_core, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        peaks_c = (resp_core == dil_c) & (resp_core > 0.40) & (dist_to_boundary > 16)
        yc, xc = np.nonzero(peaks_c)
        for y, x in zip(yc, xc):
            raw_cores.append((float(x), float(y), 0.5, float(resp_core[y, x])))

        dil_d = cv2.dilate(resp_delta, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        peaks_d = (resp_delta == dil_d) & (resp_delta > 0.40) & (dist_to_boundary > 16)
        yd, xd = np.nonzero(peaks_d)
        for y, x in zip(yd, xd):
            raw_deltas.append((float(x), float(y), -0.5, float(resp_delta[y, x])))

    elif method == "curvature":
        # Pure orientation curvature local maxima detection
        dil_k = cv2.dilate(curvature_map, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        peaks_k = (curvature_map == dil_k) & (curvature_map > 1.20) & (dist_to_boundary > 16)
        yk, xk = np.nonzero(peaks_k)
        for y, x in zip(yk, xk):
            fc = float(resp_core[y, x])
            fd = float(resp_delta[y, x])
            if fc >= fd and fc >= 0.20:
                raw_cores.append((float(x), float(y), 0.5, fc))
            elif fd > fc and fd >= 0.18:
                raw_deltas.append((float(x), float(y), -0.5, fd))

    # 1. Dipole Cancellation (suppress false pairs from ridge breaks/scars)
    filtered_cores, filtered_deltas = cancel_core_delta_dipoles(
        raw_cores, raw_deltas, dipole_distance=dipole_radius
    )

    # 2. Non-Maximum Suppression & Spatial Clustering
    clustered_cores = cluster_singular_points(filtered_cores, kind="core", radius=cluster_radius)
    clustered_deltas = cluster_singular_points(filtered_deltas, kind="delta", radius=cluster_radius)

    # 3. Confidence Scoring & Truncation Tagging
    final_cores: List[SingularPoint] = []
    for cx, cy, p_val, f_resp in clustered_cores:
        kind = "whorl_core" if p_val >= 0.8 else "core"
        conf, is_trunc = compute_singular_point_confidence(
            cx, cy, kind, p_val, f_resp,
            coherence_map=orientation_res.coherence,
            dist_to_boundary=dist_to_boundary,
            method=method,
        )
        if conf >= min_confidence:
            final_cores.append(
                SingularPoint(
                    x=cx,
                    y=cy,
                    kind=kind,
                    confidence=conf,
                    poincare_index=p_val,
                    filter_response=f_resp,
                    is_truncated=is_trunc,
                    feature_visible=(conf >= 0.35 and not is_trunc),
                )
            )

    final_deltas: List[SingularPoint] = []
    for dx, dy, p_val, f_resp in clustered_deltas:
        conf, is_trunc = compute_singular_point_confidence(
            dx, dy, "delta", p_val, f_resp,
            coherence_map=orientation_res.coherence,
            dist_to_boundary=dist_to_boundary,
            method=method,
        )
        if conf >= min_confidence:
            final_deltas.append(
                SingularPoint(
                    x=dx,
                    y=dy,
                    kind="delta",
                    confidence=conf,
                    poincare_index=p_val,
                    filter_response=f_resp,
                    is_truncated=is_trunc,
                    feature_visible=(conf >= 0.35 and not is_trunc),
                )
            )

    # 4. Basilar / Crease Boundary Suppression & Delta Proximity Filter for Core Candidates
    fg_y, _ = np.nonzero(orientation_res.mask)
    if fg_y.size > 0:
        y_min_fg = float(np.min(fg_y))
        y_max_fg = float(np.max(fg_y))
        crease_cutoff_y = y_min_fg + 0.80 * (y_max_fg - y_min_fg)
    else:
        crease_cutoff_y = 0.80 * h

    max_delta_y = max((d.y for d in final_deltas), default=None)

    pruned_cores: List[SingularPoint] = []
    for c in final_cores:
        # a) Discard core candidates in bottom 20% of active foreground mask (crease zone)
        if c.y >= crease_cutoff_y:
            continue
        # b) Discard core candidates located below the lower delta line
        if max_delta_y is not None and c.y > max_delta_y:
            continue
        # c) Prune pseudo-cores that appear within the peripheral delta zone (< 45 px)
        if any(np.hypot(c.x - d.x, c.y - d.y) < 45.0 for d in final_deltas):
            continue
        pruned_cores.append(c)

    final_cores = pruned_cores

    # Sort descending by confidence
    final_cores.sort(key=lambda p: p.confidence, reverse=True)
    final_deltas.sort(key=lambda p: p.confidence, reverse=True)

    # Predict sensor glass delta truncation
    delta_truncated, exp_sector = predict_delta_truncation(
        final_cores, final_deltas, orientation_res.theta, orientation_res.mask
    )

    all_points = final_cores + final_deltas

    return SingularPointsResult(
        cores=final_cores,
        deltas=final_deltas,
        all_points=all_points,
        poincare_map=poincare_map,
        core_response_map=resp_core,
        delta_response_map=resp_delta,
        delta_truncated=delta_truncated,
        expected_delta_sector=exp_sector,
        method=method,
        curvature_map=curvature_map,
    )

