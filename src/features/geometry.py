"""
Ridge Geometry, Frequency Estimation, Trajectory Analysis, and Loop Directionality.
Vectorized implementation for Apple Silicon (macOS).
Includes:
1. Local ridge frequency field estimation in valid range [0.10, 0.25] cycles/pixel.
2. Ridge flow trajectory analysis (streamlines tracing from cores and grid points).
3. Loop directionality detection (sloping left vs. sloping right) and Ulnar vs. Radial mapping.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
import math
import cv2
import numpy as np

from .orientation import OrientationResult, to_grayscale_numpy
from .singular_points import SingularPointsResult, SingularPoint


@dataclass
class GeometryResult:
    """
    Container for fingerprint geometric, topological, and frequency properties.
    """
    frequency_map: np.ndarray        # Local ridge frequency map (H_b, W_b), in cycles/pixel
    median_frequency: float          # Median ridge frequency across foreground in cycles/pixel
    median_wavelength: float         # Median ridge period (wavelength) in pixels
    streamlines: List[np.ndarray]    # Traced streamline trajectories (each shape (N, 2))
    flow_direction: str              # 'left', 'right', 'symmetric', 'circular', or 'unknown'
    loop_type_estimate: str          # 'left_loop', 'right_loop', 'whorl', 'arch', or 'unknown'
    flow_asymmetry: float            # Horizontal exit drift (+ for right, - for left)
    exit_angle_deg: float            # Mean trajectory exit angle in degrees [-180, 180]
    confidence: float                # Overall confidence in geometry / loop classification [0.0, 1.0]
    anatomical_class: Optional[str]  # Anatomical type ('Ulnar Loop', 'Radial Loop', etc.)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "median_frequency_cyc_per_px": float(round(self.median_frequency, 4)),
            "median_wavelength_px": float(round(self.median_wavelength, 2)),
            "num_streamlines": len(self.streamlines),
            "flow_direction": self.flow_direction,
            "loop_type_estimate": self.loop_type_estimate,
            "flow_asymmetry": float(round(self.flow_asymmetry, 3)),
            "exit_angle_deg": float(round(self.exit_angle_deg, 2)),
            "confidence": float(round(self.confidence, 4)),
            "anatomical_class": self.anatomical_class,
        }


def estimate_ridge_frequency(
    img: Union[np.ndarray, Any],
    orientation_res: OrientationResult,
    block_size: int = 16,
    min_freq: float = 0.10,
    max_freq: float = 0.25,
    window_size: int = 32,
    default_fallback_freq: float = 0.16,
) -> Tuple[np.ndarray, float, float]:
    """
    Estimate local ridge frequency using orthogonal 1D profile spectral peak analysis.
    Constrains estimates to valid physiological fingerprint range [min_freq, max_freq] cycles/pixel.
    Optimized with fast local bounding box patch extraction, linear detrending, and Hann windowing.

    Returns:
        Tuple[np.ndarray, float, float]: (frequency_map (H_b, W_b), median_freq, median_wavelength)
    """
    gray = to_grayscale_numpy(img)
    h, w = gray.shape
    b_th = orientation_res.block_theta
    b_mask = orientation_res.block_mask
    gh, gw = b_th.shape

    freq_map = np.zeros((gh, gw), dtype=np.float32)
    valid_freqs: List[float] = []

    # Frequency bin grid for zero-padded 1D FFT of length max(64, window_size)
    n_fft = max(64, window_size)
    freq_bins = np.fft.rfftfreq(n_fft, d=1.0)
    search_idx = np.where((freq_bins >= min_freq * 0.85) & (freq_bins <= max_freq * 1.15))[0]

    half_w = window_size // 2
    crop_r = int(np.ceil(half_w * np.sqrt(2))) + 2
    hann_win = np.hanning(window_size).astype(np.float32)
    x_coords = np.arange(window_size, dtype=np.float32)
    x_mean = float(np.mean(x_coords))
    x_diff = x_coords - x_mean
    x_var_sum = float(np.sum(x_diff ** 2))

    for r in range(gh):
        for c in range(gw):
            if not b_mask[r, c]:
                continue

            cx = c * block_size + block_size // 2
            cy = r * block_size + block_size // 2

            # Check boundary clearance for fast local bounding box crop
            if cx < crop_r or cx >= w - crop_r or cy < crop_r or cy >= h - crop_r:
                continue

            # Ridge normal direction (perpendicular to ridge tangent)
            theta_rad = b_th[r, c]
            normal_rad = theta_rad + np.pi / 2.0

            # Rotate local patch around center so ridge normal aligns with horizontal x-axis
            sub_img = gray[cy - crop_r : cy + crop_r, cx - crop_r : cx + crop_r]
            rot_mat = cv2.getRotationMatrix2D((crop_r, crop_r), np.rad2deg(-normal_rad), 1.0)
            rot_patch = cv2.warpAffine(sub_img, rot_mat, (2 * crop_r, 2 * crop_r), flags=cv2.INTER_LINEAR)
            patch = rot_patch[crop_r - half_w : crop_r + half_w, crop_r - half_w : crop_r + half_w]

            if patch.shape != (window_size, window_size):
                continue

            # Project across rows (mean along ridge tangent direction to get orthogonal 1D profile)
            profile = np.mean(patch, axis=0)

            # Fast linear detrending using precomputed 1D least-squares formula
            p_mean = float(np.mean(profile))
            slope = float(np.dot(x_diff, profile)) / x_var_sum
            intercept = p_mean - slope * x_mean
            profile_detrend = profile - (slope * x_coords + intercept)
            profile_windowed = profile_detrend * hann_win

            # Compute zero-padded power spectrum
            fft_mag = np.abs(np.fft.rfft(profile_windowed, n=n_fft))

            if len(search_idx) > 0:
                best_sub_idx = search_idx[np.argmax(fft_mag[search_idx])]
                peak_f = float(freq_bins[best_sub_idx])

                # Accept if within or very near physiological bounds
                if min_freq * 0.85 <= peak_f <= max_freq * 1.15:
                    clamped_f = float(np.clip(peak_f, min_freq, max_freq))
                    freq_map[r, c] = clamped_f
                    valid_freqs.append(clamped_f)

    # Compute foreground median frequency
    if valid_freqs:
        med_freq = float(np.median(valid_freqs))
    else:
        med_freq = default_fallback_freq

    # Infill missing / invalid foreground cells with median frequency
    for r in range(gh):
        for c in range(gw):
            if b_mask[r, c] and freq_map[r, c] == 0.0:
                freq_map[r, c] = med_freq

    med_wavelength = float(1.0 / med_freq) if med_freq > 1e-4 else 0.0
    return freq_map, med_freq, med_wavelength


def trace_single_streamline(
    x0: float,
    y0: float,
    theta_field: np.ndarray,
    mask: np.ndarray,
    coherence: np.ndarray,
    step_size: float = 2.0,
    max_steps: int = 150,
    min_coherence: float = 0.04,
) -> np.ndarray:
    """
    Trace a bidirectional ridge streamline from a seed point using Runge-Kutta 2 (Heun).
    Optimized with scalar math for high-throughput execution on Apple Silicon.
    """
    h, w = theta_field.shape
    ix0, iy0 = int(round(x0)), int(round(y0))

    if ix0 < 4 or ix0 >= w - 4 or iy0 < 4 or iy0 >= h - 4 or not mask[iy0, ix0]:
        return np.empty((0, 2), dtype=np.float32)

    init_th = float(theta_field[iy0, ix0])
    base_u = math.cos(init_th)
    base_v = math.sin(init_th)

    def trace_branch(init_u: float, init_v: float) -> List[Tuple[float, float]]:
        branch = []
        curr_x = float(x0)
        curr_y = float(y0)
        curr_u = init_u
        curr_v = init_v

        for _ in range(max_steps):
            ix, iy = int(round(curr_x)), int(round(curr_y))
            if ix < 4 or ix >= w - 4 or iy < 4 or iy >= h - 4:
                break
            if not mask[iy, ix] or coherence[iy, ix] < min_coherence:
                break

            th = float(theta_field[iy, ix])
            vx = math.cos(th)
            vy = math.sin(th)

            # Bidirectional alignment
            if vx * curr_u + vy * curr_v < 0:
                vx, vy = -vx, -vy
            curr_u, curr_v = vx, vy

            # RK2 Heun step
            k1_x = curr_u * step_size
            k1_y = curr_v * step_size
            mid_x = curr_x + k1_x
            mid_y = curr_y + k1_y

            imid_x, imid_y = int(round(mid_x)), int(round(mid_y))
            if imid_x < 4 or imid_x >= w - 4 or imid_y < 4 or imid_y >= h - 4:
                branch.append((curr_x, curr_y))
                break

            th_mid = float(theta_field[imid_y, imid_x])
            vmid_x = math.cos(th_mid)
            vmid_y = math.sin(th_mid)
            if vmid_x * curr_u + vmid_y * curr_v < 0:
                vmid_x, vmid_y = -vmid_x, -vmid_y

            step_dx = 0.5 * (k1_x + vmid_x * step_size)
            step_dy = 0.5 * (k1_y + vmid_y * step_size)

            curr_x += step_dx
            curr_y += step_dy
            branch.append((curr_x, curr_y))

            # Early loop-closure check
            if len(branch) > 20:
                dx_loop = curr_x - branch[0][0]
                dy_loop = curr_y - branch[0][1]
                if dx_loop * dx_loop + dy_loop * dy_loop < 16.0:
                    break

        return branch

    # Trace forward and backward branches
    fwd = trace_branch(base_u, base_v)
    bwd = trace_branch(-base_u, -base_v)

    full_path = bwd[::-1] + [(x0, y0)] + fwd
    return np.array(full_path, dtype=np.float32)


def trace_ridge_streamlines(
    orientation_res: OrientationResult,
    singular_points_res: Optional[SingularPointsResult] = None,
    step_size: float = 2.0,
    max_steps: int = 120,
    seed_grid_stride: Optional[int] = None,
    min_coherence: float = 0.04,
) -> List[np.ndarray]:
    """
    Trace full set of ridge streamlines across the fingerprint impression.
    Seeds include detected singular points and an interior regular grid.
    """
    h, w = orientation_res.theta.shape
    if seed_grid_stride is None:
        seed_grid_stride = 32 if max(h, w) <= 300 else max(32, min(h, w) // 10)
    dist_boundary = cv2.distanceTransform(orientation_res.mask, cv2.DIST_L2, 5)

    seeds: List[Tuple[float, float]] = []

    # Priority 1: Singular points (cores)
    if singular_points_res is not None:
        for core in singular_points_res.cores:
            # Seed slightly offset from the exact core singularity to follow loop trajectory
            seeds.append((core.x, core.y + 12.0))
            seeds.append((core.x, core.y - 12.0))
            seeds.append((core.x - 12.0, core.y))
            seeds.append((core.x + 12.0, core.y))

    # Priority 2: Interior regular grid
    for y in range(seed_grid_stride, h - seed_grid_stride, seed_grid_stride):
        for x in range(seed_grid_stride, w - seed_grid_stride, seed_grid_stride):
            if orientation_res.mask[y, x] and dist_boundary[y, x] > 20 and orientation_res.coherence[y, x] > 0.35:
                seeds.append((float(x), float(y)))

    streamlines: List[np.ndarray] = []
    for sx, sy in seeds:
        sl = trace_single_streamline(
            sx, sy,
            theta_field=orientation_res.theta,
            mask=orientation_res.mask,
            coherence=orientation_res.coherence,
            step_size=step_size,
            max_steps=max_steps,
            min_coherence=min_coherence,
        )
        if len(sl) >= 15:
            streamlines.append(sl)

    return streamlines



def detect_loop_directionality(
    orientation_res: OrientationResult,
    singular_points_res: Optional[SingularPointsResult] = None,
    hand: Optional[str] = None,
    finger_name: Optional[str] = None,
    img: Optional[Union[np.ndarray, Any]] = None,
) -> GeometryResult:
    """
    Detect fingerprint loop directionality (sloping left vs. sloping right)
    derived directly from orientation field vectors and streamline trajectories.
    Validates Stage 1/2 Ulnar vs. Radial orientation findings.
    """
    h, w = orientation_res.theta.shape

    # 1. Estimate Ridge Frequency Map (using image if provided, else orientation field)
    freq_source = img if img is not None else orientation_res.theta
    freq_map, med_freq, med_wavelength = estimate_ridge_frequency(
        freq_source, orientation_res
    )

    # 2. Extract Streamlines
    streamlines = trace_ridge_streamlines(
        orientation_res, singular_points_res=singular_points_res, min_coherence=0.04
    )

    # 3. Analyze Directionality
    cores = singular_points_res.cores if singular_points_res else []
    deltas = singular_points_res.deltas if singular_points_res else []

    flow_asymmetry = 0.0
    exit_angles: List[float] = []
    flow_direction = "unknown"
    loop_type = "unknown"
    confidence = 0.50

    # High-confidence singular point filtering for structural pattern checks
    strong_cores = [c for c in cores if c.confidence >= 0.70 and c.filter_response >= 0.25]
    strong_deltas = [d for d in deltas if d.confidence >= 0.55 and d.filter_response >= 0.20]

    # Bilateral delta check for whorls (core flanked bilaterally by two deltas below core)
    has_bilateral_deltas = False
    if len(cores) >= 1 and len(strong_deltas) >= 2:
        cx = cores[0].x
        cy = cores[0].y
        flank_deltas = [d for d in strong_deltas if d.y >= cy - 16.0]
        has_left_delta = any(d.x < cx - 16.0 for d in flank_deltas)
        has_right_delta = any(d.x > cx + 16.0 for d in flank_deltas)
        has_bilateral_deltas = (has_left_delta and has_right_delta)

    # Check for genuine two-core whorl (two separated strong cores within core region)
    has_dual_cores = False
    if len(strong_cores) >= 2:
        d_cores = np.hypot(strong_cores[0].x - strong_cores[1].x, strong_cores[0].y - strong_cores[1].y)
        if 15.0 <= d_cores <= 120.0:
            has_dual_cores = True

    # Directionality Indicator 1: Sub-core Orientation Field Vector Tilt
    core_ref = cores[0] if cores else None
    if core_ref is not None:
        cy_ref, cx_ref = int(round(core_ref.y)), int(round(core_ref.x))
        y0, y1 = max(0, cy_ref - 10), min(h, cy_ref + 80)
        x0, x1 = max(0, cx_ref - 60), min(w, cx_ref + 60)
    else:
        y0, y1 = int(h * 0.25), int(h * 0.85)
        x0, x1 = int(w * 0.15), int(w * 0.85)

    sub_mask = orientation_res.mask[y0:y1, x0:x1] > 0
    sub_th = orientation_res.theta[y0:y1, x0:x1]
    sin2_tilt = float(np.mean(np.sin(2.0 * sub_th[sub_mask]))) if np.any(sub_mask) else 0.0

    # Directionality Indicator 2: Core-Delta Relative Position (only deltas below core)
    delta_indicator = 0  # +1 for right loop, -1 for left loop
    if core_ref is not None and deltas:
        cx = core_ref.x
        lower_d = [d for d in deltas if d.confidence >= 0.45 and d.y >= core_ref.y - 16.0]
        if lower_d:
            best_d = min(lower_d, key=lambda d: np.hypot(d.x - cx, d.y - core_ref.y))
            if best_d.x > cx + 12.0:
                delta_indicator = -1  # Left Loop (delta on right)
            elif best_d.x < cx - 12.0:
                delta_indicator = +1  # Right Loop (delta on left)

    # Directionality Indicator 3: Streamlines exiting from core (or center of mass)
    if core_ref is not None:
        sx, sy = core_ref.x, core_ref.y + 12.0
    else:
        # Fallback to foreground center of mass
        m_pts = np.nonzero(orientation_res.mask)
        sy = float(np.mean(m_pts[0])) if m_pts[0].size > 0 else float(h // 2)
        sx = float(np.mean(m_pts[1])) if m_pts[1].size > 0 else float(w // 2)

    core_stream = trace_single_streamline(
        sx, sy,
        theta_field=orientation_res.theta,
        mask=orientation_res.mask,
        coherence=orientation_res.coherence,
        step_size=2.0,
        max_steps=120,
        min_coherence=0.04,
    )

    traj_dx = 0.0
    if len(core_stream) > 10:
        term_pt = core_stream[-1]
        start_pt = core_stream[0]
        dx_end = float(term_pt[0] - sx)
        dx_start = float(start_pt[0] - sx)
        traj_dx = (dx_end + dx_start) / 2.0
        angle_rad = np.arctan2(term_pt[1] - sy, term_pt[0] - sx)
        exit_angles.append(float(np.rad2deg(angle_rad)))

    # Combined Continuous Flow Asymmetry Score
    # Positive -> flows Right (Right Loop), Negative -> flows Left (Left Loop)
    flow_asymmetry = 0.45 * sin2_tilt + 0.35 * float(np.clip(traj_dx / 40.0, -1.0, 1.0)) + 0.20 * float(delta_indicator)

    # Topological Pattern Classification
    if (has_dual_cores or has_bilateral_deltas) and abs(flow_asymmetry) < 0.25:
        # Genuine Whorl pattern (symmetric / circular)
        flow_direction = "circular"
        loop_type = "whorl"
        confidence = 0.88
    elif len(cores) == 0 and len(deltas) == 0 and abs(flow_asymmetry) < 0.08:
        # Plain Arch pattern
        flow_direction = "symmetric"
        loop_type = "arch"
        confidence = 0.82
    elif flow_asymmetry > 0.05 or (delta_indicator == +1 and flow_asymmetry > -0.05):
        # Right Loop
        flow_direction = "right"
        loop_type = "right_loop"
        confidence = float(np.clip(0.65 + abs(flow_asymmetry) * 0.35, 0.65, 0.98))
    elif flow_asymmetry < -0.05 or (delta_indicator == -1 and flow_asymmetry < 0.05):
        # Left Loop
        flow_direction = "left"
        loop_type = "left_loop"
        confidence = float(np.clip(0.65 + abs(flow_asymmetry) * 0.35, 0.65, 0.98))
    else:
        # Symmetric / Tented Arch
        flow_direction = "symmetric"
        loop_type = "tented_arch" if (len(deltas) > 0 or len(cores) > 0) else "arch"
        confidence = 0.70

    mean_exit_angle = float(np.mean(exit_angles)) if exit_angles else 0.0

    # 4. Derived Anatomical Ulnar vs. Radial Mapping
    anatomical_class: Optional[str] = None
    if hand and loop_type in ("left_loop", "right_loop"):
        hand_clean = str(hand).strip().lower()
        if "right" in hand_clean:
            # Right Hand: Left Loop slopes toward Thumb (Radial), Right Loop slopes toward Pinky (Ulnar)
            anatomical_class = "Radial Loop" if loop_type == "left_loop" else "Ulnar Loop"
        elif "left" in hand_clean:
            # Left Hand: Left Loop slopes toward Pinky (Ulnar), Right Loop slopes toward Thumb (Radial)
            anatomical_class = "Ulnar Loop" if loop_type == "left_loop" else "Radial Loop"

    return GeometryResult(
        frequency_map=freq_map,
        median_frequency=med_freq,
        median_wavelength=med_wavelength,
        streamlines=streamlines,
        flow_direction=flow_direction,
        loop_type_estimate=loop_type,
        flow_asymmetry=flow_asymmetry,
        exit_angle_deg=mean_exit_angle,
        confidence=confidence,
        anatomical_class=anatomical_class,
    )


def extract_fingerprint_geometry(
    img: Union[np.ndarray, Any],
    orientation_res: OrientationResult,
    singular_points_res: Optional[SingularPointsResult] = None,
    hand: Optional[str] = None,
    finger_name: Optional[str] = None,
) -> GeometryResult:
    """
    High-level facade to extract full fingerprint geometry and loop directionality.
    """
    return detect_loop_directionality(
        orientation_res,
        singular_points_res=singular_points_res,
        hand=hand,
        finger_name=finger_name,
        img=img,
    )

