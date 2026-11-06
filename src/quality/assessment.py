"""
Biometric Image Quality Assessment (IQA) Engine.
Computes standardized biometric fingerprint quality metrics inspired by NFIQ and ISO/IEC 29794-4:
1. Local Ridge Clarity Index (RCI): orthogonal profile contrast and sinusoidal regularity.
2. Frequency Continuity: smooth spatial continuity in local ridge frequency field.
3. Sharpness & Blur Score: Laplacian variance and edge gradient magnitude.
4. Dynamic Range & Saturation: underexposed (dry finger) and overexposed (wet finger) detection.
5. Composite Overall Quality Score Q in [0, 100] and 4-tier biometric classification.
Optimized for Apple Silicon with vectorized NumPy and OpenCV routines.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, Union
import cv2
import numpy as np

from src.features.orientation import (
    OrientationResult,
    compute_orientation_field,
    to_grayscale_numpy,
)


@dataclass
class BiometricQualityResult:
    """Container for biometric image quality metrics and classification."""
    overall_score: float               # Composite Quality Score Q in [0.0, 100.0]
    quality_tier: str                  # 'Excellent' | 'Good' | 'Fair' | 'Poor'
    clarity_score: float               # Ridge clarity index in [0.0, 100.0]
    coherence_score: float             # Mean foreground coherence in [0.0, 100.0]
    contrast_score: float              # Dynamic range & illumination balance in [0.0, 100.0]
    sharpness_score: float             # Edge crispness and Laplacian variance in [0.0, 100.0]
    frequency_continuity_score: float  # Smoothness of spatial frequency field in [0.0, 100.0]
    dynamic_range: float               # Foreground 95th - 5th percentile span [0.0, 255.0]
    dryness_ratio: float               # Fraction of faint / high-intensity foreground pixels
    wetness_ratio: float               # Fraction of saturated dark pixels
    is_dry: bool                       # True if dry finger artifact detected
    is_wet: bool                       # True if wet finger / ink flooding detected
    laplacian_variance: float          # Variance of Laplacian operator
    mean_gradient: float               # Mean gradient magnitude across foreground
    clarity_map: np.ndarray            # 2D pixel-wise clarity heatmap (H, W), dtype float32
    block_clarity: np.ndarray          # 2D block-level clarity array (H_b, W_b), dtype float32
    freq_map: np.ndarray               # 2D local ridge frequency map (H_b, W_b), in cycles/px
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert quality metrics to dictionary."""
        return {
            "overall_score": float(round(self.overall_score, 2)),
            "quality_tier": self.quality_tier,
            "clarity_score": float(round(self.clarity_score, 2)),
            "coherence_score": float(round(self.coherence_score, 2)),
            "contrast_score": float(round(self.contrast_score, 2)),
            "sharpness_score": float(round(self.sharpness_score, 2)),
            "frequency_continuity_score": float(round(self.frequency_continuity_score, 2)),
            "dynamic_range": float(round(self.dynamic_range, 2)),
            "dryness_ratio": float(round(self.dryness_ratio, 4)),
            "wetness_ratio": float(round(self.wetness_ratio, 4)),
            "is_dry": self.is_dry,
            "is_wet": self.is_wet,
            "laplacian_variance": float(round(self.laplacian_variance, 2)),
            "mean_gradient": float(round(self.mean_gradient, 2)),
        }


def compute_local_ridge_clarity(
    img: np.ndarray,
    orientation_res: OrientationResult,
    block_size: int = 16,
    window_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Measure Local Ridge Clarity Index (RCI) across foreground blocks.
    Extracts 1D intensity profiles orthogonal to ridge tangent, detrends illumination gradients,
    evaluates peak-to-valley contrast, and performs zero-padded FFT to measure sinusoidal regularity.
    Fully vectorized across all foreground blocks.

    Returns:
        Tuple:
            - clarity_map: np.ndarray (H, W) float32 in [0.0, 1.0]
            - block_clarity: np.ndarray (H_b, W_b) float32 in [0.0, 1.0]
            - freq_map: np.ndarray (H_b, W_b) float32 in cycles/pixel
            - clarity_score: float in [0.0, 100.0]
    """
    gray = to_grayscale_numpy(img)
    h, w = gray.shape
    b_th = orientation_res.block_theta
    b_mask = orientation_res.block_mask
    gh, gw = b_th.shape

    clarity_map = np.zeros((h, w), dtype=np.float32)
    block_clarity = np.zeros((gh, gw), dtype=np.float32)
    freq_map = np.zeros((gh, gw), dtype=np.float32)

    # Empty foreground check
    r_idx, c_idx = np.where(b_mask > 0)
    if len(r_idx) == 0:
        return clarity_map, block_clarity, freq_map, 0.0

    half_w = window_size // 2
    margin = half_w + 2

    # Block center coordinates
    cx = c_idx * block_size + block_size // 2
    cy = r_idx * block_size + block_size // 2

    valid = (cx >= margin) & (cx < w - margin) & (cy >= margin) & (cy < h - margin)
    if not np.any(valid):
        return clarity_map, block_clarity, freq_map, 0.0

    r_idx = r_idx[valid]
    c_idx = c_idx[valid]
    cx = cx[valid]
    cy = cy[valid]

    # Ridge normal orientation (perpendicular to ridge tangent)
    normal_rad = b_th[r_idx, c_idx] + np.pi / 2.0
    N = len(cx)

    # Precomputed sampling grid along normal direction
    t_offsets = np.arange(-half_w, half_w, dtype=np.float32)
    hann = np.hanning(window_size).astype(np.float32)
    x_axis = np.arange(window_size, dtype=np.float32)
    x_c = x_axis - np.mean(x_axis)
    x_norm_sq = np.sum(x_c**2)

    cos_n = np.cos(normal_rad)[:, None]
    sin_n = np.sin(normal_rad)[:, None]

    # Sample central orthogonal line
    sample_x = cx[:, None] + t_offsets[None, :] * cos_n
    sample_y = cy[:, None] + t_offsets[None, :] * sin_n

    # Also sample two parallel lines offset by -2, +2 along tangent for noise-robust averaging
    sin_t = -np.sin(normal_rad)[:, None]
    cos_t = np.cos(normal_rad)[:, None]

    sample_x_all = np.stack([sample_x - 2.0 * sin_t, sample_x, sample_x + 2.0 * sin_t], axis=0)
    sample_y_all = np.stack([sample_y - 2.0 * cos_t, sample_y, sample_y + 2.0 * cos_t], axis=0)

    # Fast vectorized bilinear interpolation
    x0 = np.clip(sample_x_all.astype(int), 0, w - 2)
    y0 = np.clip(sample_y_all.astype(int), 0, h - 2)
    wx1 = sample_x_all - x0
    wx0 = 1.0 - wx1
    wy1 = sample_y_all - y0
    wy0 = 1.0 - wy1

    sampled_vals = (
        gray[y0, x0] * wx0 * wy0
        + gray[y0, x0 + 1] * wx1 * wy0
        + gray[y0 + 1, x0] * wx0 * wy1
        + gray[y0 + 1, x0 + 1] * wx1 * wy1
    )  # shape (3, N, window_size)

    # Mean orthogonal 1D profile
    prof = np.mean(sampled_vals, axis=0)  # shape (N, window_size)

    # Linear detrending: profile - mean - slope * x_c
    prof_mean = np.mean(prof, axis=1, keepdims=True)
    prof_centered = prof - prof_mean
    slope = np.sum(prof_centered * x_c[None, :], axis=1, keepdims=True) / max(x_norm_sq, 1e-6)
    prof_dt = prof_centered - slope * x_c[None, :]

    # Profile contrast: 90th percentile - 10th percentile
    p90 = np.percentile(prof, 90, axis=1)
    p10 = np.percentile(prof, 10, axis=1)
    p_amp = p90 - p10
    # Scaled profile contrast: ~25-45 gray levels represents standard biometric ridge contrast
    contrast_norm = np.clip(p_amp / 32.0, 0.0, 1.0)

    # Windowed FFT for sinusoidal regularity and ridge frequency
    n_fft = max(64, window_size)
    prof_win = prof_dt * hann[None, :]
    spec = np.abs(np.fft.rfft(prof_win, n=n_fft, axis=-1))

    freq_bins = np.fft.rfftfreq(n_fft, d=1.0)
    # Physiological frequency range: [0.08, 0.28] cycles/pixel (ridge wavelength 3.5 to 12.5 px)
    search_idx = np.where((freq_bins >= 0.08) & (freq_bins <= 0.28))[0]

    if len(search_idx) > 0:
        band_spec = spec[:, search_idx]
        best_idx = np.argmax(band_spec, axis=1)
        peak_f = freq_bins[search_idx[best_idx]]
        peak_bin_indices = search_idx[best_idx]
        # Sum 3-bin fundamental mainlobe power to avoid spectral leakage suppression
        peak_power = np.zeros(N, dtype=np.float32)
        for i in range(N):
            p_bin = peak_bin_indices[i]
            b_min = max(1, p_bin - 1)
            b_max = min(spec.shape[1], p_bin + 2)
            peak_power[i] = np.sum(spec[i, b_min:b_max]**2)
    else:
        peak_f = np.full(N, 0.16, dtype=np.float32)
        peak_power = np.zeros(N, dtype=np.float32)

    total_ac_power = np.sum(spec[:, 1:]**2, axis=1)
    sin_regularity = np.clip(peak_power / np.maximum(total_ac_power, 1e-6), 0.0, 1.0)

    # Coherence at block locations
    coh_vals = orientation_res.block_coherence[r_idx, c_idx]

    # Composite Local Ridge Clarity Index (RCI)
    # Combines profile contrast, harmonic sinusoidal regularity, and local coherence
    rci_vals = np.clip(contrast_norm * np.sqrt(sin_regularity) * coh_vals, 0.0, 1.0)

    block_clarity[r_idx, c_idx] = rci_vals
    freq_map[r_idx, c_idx] = peak_f

    # Infill missing foreground cells with median frequency and median clarity
    fg_freqs = freq_map[b_mask > 0]
    valid_fg_freqs = fg_freqs[fg_freqs > 0.0]
    med_f = float(np.median(valid_fg_freqs)) if len(valid_fg_freqs) > 0 else 0.16

    fg_clarities = block_clarity[b_mask > 0]
    valid_fg_clarities = fg_clarities[fg_clarities > 0.0]
    med_c = float(np.median(valid_fg_clarities)) if len(valid_fg_clarities) > 0 else 0.0

    for r in range(gh):
        for c in range(gw):
            if b_mask[r, c]:
                if freq_map[r, c] == 0.0:
                    freq_map[r, c] = med_f
                if block_clarity[r, c] == 0.0:
                    block_clarity[r, c] = med_c

    # Interpolate block clarity to full-resolution clarity map
    clarity_resized = cv2.resize(block_clarity, (w, h), interpolation=cv2.INTER_LINEAR)
    clarity_map = np.clip(clarity_resized * orientation_res.mask.astype(np.float32), 0.0, 1.0)

    # Clarity score [0, 100]: scaled foreground mean
    # High clarity prints with distinct sinusoidal ridges achieve 70-95, smudged/noise prints < 30
    fg_clarity_vals = block_clarity[b_mask > 0]
    if len(fg_clarity_vals) > 0:
        raw_mean = float(np.mean(fg_clarity_vals))
        # Scaled calibration: raw mean of ~0.45 maps to ~80 score
        clarity_score = float(np.clip(raw_mean * 180.0, 0.0, 100.0))
    else:
        clarity_score = 0.0

    return clarity_map, block_clarity, freq_map, clarity_score


def compute_frequency_continuity(
    freq_map: np.ndarray,
    block_mask: np.ndarray,
) -> float:
    """
    Quantify smooth spatial continuity in local ridge frequency f0.
    Penalizes sudden jumps caused by noise, obliteration, or smudges.

    Returns:
        float: Frequency Continuity Score in [0.0, 100.0].
    """
    gh, gw = freq_map.shape
    if gh < 2 or gw < 2:
        return 100.0

    # Horizontal and vertical differences where adjacent blocks are both in foreground
    h_valid = (block_mask[:, 1:] > 0) & (block_mask[:, :-1] > 0)
    v_valid = (block_mask[1:, :] > 0) & (block_mask[:-1, :] > 0)

    diffs = []
    if np.any(h_valid):
        dx = np.abs(freq_map[:, 1:] - freq_map[:, :-1])[h_valid]
        diffs.extend(dx.tolist())
    if np.any(v_valid):
        dy = np.abs(freq_map[1:, :] - freq_map[:-1, :])[v_valid]
        diffs.extend(dy.tolist())

    if not diffs:
        return 100.0

    mean_diff = float(np.mean(diffs))
    # Natural spatial frequency variation across adjacent blocks is ~0.005-0.015 cycles/pixel
    # Damage or noise creates erratic jumps (>0.05)
    score = float(np.clip(100.0 * np.exp(-mean_diff / 0.035), 0.0, 100.0))
    return score


def compute_sharpness_and_blur(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Compute edge gradient magnitude and Laplacian variance across foreground ridges.
    Quantifies focus sharpness vs. blur degradation.

    Returns:
        Tuple:
            - sharpness_score: float in [0.0, 100.0]
            - laplacian_variance: float
            - mean_gradient: float
    """
    gray = to_grayscale_numpy(img)
    if mask is None:
        fg_mask = np.ones_like(gray, dtype=bool)
    else:
        fg_mask = mask > 0

    if not np.any(fg_mask):
        return 0.0, 0.0, 0.0

    # Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(np.var(lap[fg_mask]))

    # Edge gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    mean_grad = float(np.mean(grad_mag[fg_mask]))

    # Normalized score using soft saturation curve
    s_lap = lap_var / (lap_var + 60.0)
    s_grad = mean_grad / (mean_grad + 20.0)
    sharpness_score = float(np.clip(100.0 * (0.6 * s_lap + 0.4 * s_grad), 0.0, 100.0))

    return sharpness_score, lap_var, mean_grad


def compute_dynamic_range_saturation(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, bool, bool]:
    """
    Evaluate dynamic range, contrast, and detect over/underexposure:
    - Dry finger / faint ridges: high mean intensity, lack of dark ridge pigment.
    - Wet finger / ink blotches: flooded valleys, high fraction of saturated dark pixels.

    Returns:
        Tuple:
            - saturation_score: float in [0.0, 100.0]
            - dynamic_range: float [0.0, 255.0]
            - dryness_ratio: float [0.0, 1.0]
            - wetness_ratio: float [0.0, 1.0]
            - is_dry: bool
            - is_wet: bool
    """
    gray = to_grayscale_numpy(img)
    if mask is None:
        fg_mask = np.ones_like(gray, dtype=bool)
    else:
        fg_mask = mask > 0

    if not np.any(fg_mask):
        return 0.0, 0.0, 0.0, 0.0, False, False

    fg_pixels = gray[fg_mask]
    p5 = float(np.percentile(fg_pixels, 5))
    p95 = float(np.percentile(fg_pixels, 95))
    dyn_range = float(p95 - p5)

    # Dryness: faint ridges, washed out, lacking dark ridge pigment
    dry_ratio = float(np.mean(fg_pixels > 225.0))
    mean_val = float(np.mean(fg_pixels))

    # Wetness: ink flooded valleys, saturated black pixels
    wet_ratio = float(np.mean(fg_pixels < 35.0))

    # Physiological biometric illumination diagnostics:
    # A dry finger has faint ridges without dark pigment (high mean and low dynamic range, or p5 > 120)
    is_dry = bool(
        (mean_val > 185.0 and dyn_range < 90.0)
        or (p5 > 120.0 and dyn_range < 80.0)
        or (dry_ratio > 0.40 and wet_ratio < 0.05)
    )
    # A wet finger has valleys flooded dark without bright valley reflections (low mean and low dynamic range, or p95 < 90)
    is_wet = bool(
        (mean_val < 75.0 and dyn_range < 90.0)
        or (p95 < 90.0 and dyn_range < 80.0)
        or (wet_ratio > 0.40 and dry_ratio < 0.05)
    )

    # Dynamic range score normalized to [0, 100]
    dr_score = float(np.clip(dyn_range / 160.0 * 100.0, 0.0, 100.0))

    # Penalty for extreme illumination imbalance (dry faintness, wet ink flooding, or collapsed DR)
    penalty = 0.0
    if is_dry:
        penalty = float(np.clip((185.0 - dyn_range) / 100.0, 0.25, 0.8))
    elif is_wet:
        penalty = float(np.clip((185.0 - dyn_range) / 100.0, 0.25, 0.8))
    elif dyn_range < 70.0:
        penalty = float(np.clip((70.0 - dyn_range) / 70.0, 0.1, 0.6))

    saturation_score = float(np.clip(dr_score * (1.0 - 0.7 * penalty), 0.0, 100.0))
    return saturation_score, dyn_range, dry_ratio, wet_ratio, is_dry, is_wet


def compute_composite_quality_score(
    clarity_score: float,
    coherence_score: float,
    contrast_score: float,
    usable_area_score: float,
    frequency_continuity_score: float,
    sharpness_score: float,
) -> float:
    """
    Weighted combination of clarity, coherence, usable area, contrast, frequency continuity, and sharpness.
    Applies biometric ridge validity gating so pure noise and uniform non-fingerprints are correctly penalized.
    Weights:
        - Clarity: 0.25
        - Coherence: 0.20
        - Usable Area: 0.25
        - Contrast / Dynamic Range: 0.15
        - Frequency Continuity: 0.08
        - Sharpness: 0.07
    """
    raw_q = (
        0.25 * clarity_score
        + 0.20 * coherence_score
        + 0.25 * usable_area_score
        + 0.15 * contrast_score
        + 0.08 * frequency_continuity_score
        + 0.07 * sharpness_score
    )

    # Biometric validity gating: authentic fingerprints possess coherent ridge flow and clarity.
    # Unstructured noise (coherence < 25) or uniform blanks (coherence = 0) cannot achieve high quality.
    coh_factor = float(np.clip((coherence_score - 15.0) / 30.0, 0.0, 1.0))
    clarity_factor = float(np.clip(clarity_score / 20.0, 0.0, 1.0))
    validity = float(np.clip(0.70 * coh_factor + 0.30 * clarity_factor, 0.0, 1.0))

    final_q = raw_q * validity
    return float(np.clip(final_q, 0.0, 100.0))


def classify_quality_tier(score: float) -> str:
    """
    Map composite quality score Q in [0, 100] to biometric quality tier:
    - 'Excellent' (>= 80.0)
    - 'Good' (60.0 <= Q < 80.0)
    - 'Fair' (40.0 <= Q < 60.0)
    - 'Poor' (< 40.0)
    """
    if score >= 80.0:
        return "Excellent"
    elif score >= 60.0:
        return "Good"
    elif score >= 40.0:
        return "Fair"
    else:
        return "Poor"


def assess_biometric_quality(
    img: Union[np.ndarray, Any],
    orientation_res: Optional[OrientationResult] = None,
    usable_area_percentage: float = 100.0,
    block_size: int = 16,
) -> BiometricQualityResult:
    """
    Run complete Biometric Image Quality Assessment on a fingerprint image.
    """
    gray = to_grayscale_numpy(img)
    if orientation_res is None:
        orientation_res = compute_orientation_field(gray, block_size=block_size)

    # 1. Local Ridge Clarity & Ridge Frequency Map
    clarity_map, block_clarity, freq_map, clarity_score = compute_local_ridge_clarity(
        gray, orientation_res, block_size=block_size
    )

    # 2. Frequency Continuity
    freq_continuity = compute_frequency_continuity(freq_map, orientation_res.block_mask)

    # 3. Coherence Score
    fg_mask = orientation_res.mask > 0
    if np.any(fg_mask):
        coherence_score = float(np.mean(orientation_res.coherence[fg_mask]) * 100.0)
    else:
        coherence_score = 0.0

    # 4. Sharpness & Blur
    sharpness_score, lap_var, mean_grad = compute_sharpness_and_blur(gray, orientation_res.mask)

    # 5. Dynamic Range & Saturation
    (
        sat_score,
        dyn_range,
        dry_ratio,
        wet_ratio,
        is_dry,
        is_wet,
    ) = compute_dynamic_range_saturation(gray, orientation_res.mask)

    # 6. Composite Score
    overall_score = compute_composite_quality_score(
        clarity_score=clarity_score,
        coherence_score=coherence_score,
        contrast_score=sat_score,
        usable_area_score=usable_area_percentage,
        frequency_continuity_score=freq_continuity,
        sharpness_score=sharpness_score,
    )

    quality_tier = classify_quality_tier(overall_score)

    metrics = {
        "clarity": clarity_score,
        "coherence": coherence_score,
        "contrast": sat_score,
        "sharpness": sharpness_score,
        "frequency_continuity": freq_continuity,
        "usable_area": usable_area_percentage,
        "dynamic_range": dyn_range,
        "dryness_ratio": dry_ratio,
        "wetness_ratio": wet_ratio,
        "laplacian_variance": lap_var,
        "mean_gradient": mean_grad,
    }

    return BiometricQualityResult(
        overall_score=overall_score,
        quality_tier=quality_tier,
        clarity_score=clarity_score,
        coherence_score=coherence_score,
        contrast_score=sat_score,
        sharpness_score=sharpness_score,
        frequency_continuity_score=freq_continuity,
        dynamic_range=dyn_range,
        dryness_ratio=dry_ratio,
        wetness_ratio=wet_ratio,
        is_dry=is_dry,
        is_wet=is_wet,
        laplacian_variance=lap_var,
        mean_gradient=mean_grad,
        clarity_map=clarity_map,
        block_clarity=block_clarity,
        freq_map=freq_map,
        metrics=metrics,
    )
