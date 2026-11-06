"""
Continuous Ridge Orientation Field, Coherence, and Foreground Segmentation.
Optimized for Apple Silicon (macOS) with vectorized NumPy and OpenCV routines.
Implements normalized squared gradients (Bazen & Gerez / Rao method) with Gaussian window smoothing.
"""

from dataclasses import dataclass
from typing import Union, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import torch


@dataclass
class OrientationResult:
    """
    Container for continuous orientation field and segmentation metrics.
    """
    theta: np.ndarray             # Continuous pixel-wise orientation in [0, pi) radians (H, W)
    coherence: np.ndarray         # Continuous local coherence in [0.0, 1.0] (H, W)
    mask: np.ndarray              # Binary foreground mask {0, 1} (H, W), dtype uint8
    vx: np.ndarray                # Smoothed double-angle horizontal component (H, W)
    vy: np.ndarray                # Smoothed double-angle vertical component (H, W)
    block_theta: np.ndarray       # Block-level orientation field in [0, pi) (H_b, W_b)
    block_coherence: np.ndarray   # Block-level coherence (H_b, W_b)
    block_mask: np.ndarray        # Block-level foreground mask {0, 1} (H_b, W_b)
    block_size: int               # Block stride/dimension used for block-level fields

    @property
    def theta_degrees(self) -> np.ndarray:
        """Returns orientation field in degrees in range [0, 180)."""
        return np.rad2deg(self.theta)

    @property
    def vector_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns unit vector components (u, v) aligned with ridge tangents:
        u = cos(theta), v = sin(theta).
        """
        return np.cos(self.theta), np.sin(self.theta)

    @property
    def block_vector_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns block unit vector components (u_b, v_b)."""
        return np.cos(self.block_theta), np.sin(self.block_theta)

    def summary(self) -> Dict[str, Any]:
        """Returns high-level summary statistics of orientation field and foreground."""
        fg_mask = self.mask > 0
        fg_pixels = int(np.count_nonzero(fg_mask))
        total_pixels = int(self.mask.size)
        fg_ratio = float(fg_pixels / max(total_pixels, 1))

        if fg_pixels > 0:
            mean_coh = float(np.mean(self.coherence[fg_mask]))
            median_coh = float(np.median(self.coherence[fg_mask]))
            std_coh = float(np.std(self.coherence[fg_mask]))
            mean_theta_deg = float(np.rad2deg(np.mean(self.theta[fg_mask])))
        else:
            mean_coh = 0.0
            median_coh = 0.0
            std_coh = 0.0
            mean_theta_deg = 0.0

        return {
            "image_shape": list(self.theta.shape),
            "block_grid_shape": list(self.block_theta.shape),
            "block_size": self.block_size,
            "foreground_pixels": fg_pixels,
            "foreground_ratio": fg_ratio,
            "mean_coherence_fg": mean_coh,
            "median_coherence_fg": median_coh,
            "std_coherence_fg": std_coh,
            "mean_theta_deg_fg": mean_theta_deg,
        }


def to_grayscale_numpy(img: Union[np.ndarray, torch.Tensor, Any]) -> np.ndarray:
    """
    Standardize arbitrary image input (numpy, torch tensor, PIL) to float32 2D grayscale array [0, 255].
    """
    if img is None:
        raise ValueError("Input image cannot be None or empty.")

    # Convert torch tensor
    if hasattr(img, "detach") and hasattr(img, "cpu"):
        arr = img.detach().cpu().numpy()
        # Handle (C, H, W) or (1, C, H, W)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
    elif hasattr(img, "size") and not isinstance(img, np.ndarray):
        # PIL Image
        arr = np.array(img)
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    if arr.size == 0:
        raise ValueError("Image array contains zero elements.")

    # Convert multi-channel to single-channel grayscale
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr.astype(np.uint8) if arr.dtype != np.uint8 else arr, cv2.COLOR_BGR2GRAY)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr.astype(np.uint8) if arr.dtype != np.uint8 else arr, cv2.COLOR_BGRA2GRAY)

    arr = arr.astype(np.float32)

    # Scale to [0, 255] if normalized in [0, 1], [-1, 1], standardized, or arbitrary float range
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if min_v >= -0.01 and max_v <= 1.01 and max_v > min_v:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    elif min_v >= -1.01 and max_v <= 1.01 and min_v < -0.01:
        arr = ((np.clip(arr, -1.0, 1.0) + 1.0) / 2.0) * 255.0
    elif max_v > 1.01 and max_v <= 255.0 and min_v >= 0.0:
        # Already in standard uint8 [0, 255] range
        pass
    elif max_v > min_v:
        # Standardized (zero mean, unit variance) or arbitrary float range
        # Map [min_v, max_v] cleanly to [0, 255]
        arr = np.clip((arr - min_v) / (max_v - min_v) * 255.0, 0.0, 255.0)

    return arr


def segment_foreground(
    img: np.ndarray,
    block_size: int = 16,
    var_threshold_ratio: float = 0.12,
    min_area_ratio: float = 0.04,
) -> np.ndarray:
    """
    Segment fingerprint ridge impression from white paper / sensor margin.
    Uses local intensity variance, smoothed gradient energy, and morphological reconstruction.

    Returns:
        np.ndarray: Binary mask of shape (H, W), dtype uint8 with values in {0, 1}.
    """
    gray = to_grayscale_numpy(img)
    h, w = gray.shape

    # 1. Local Standard Deviation / Variance Filter
    k_size = (block_size, block_size)
    mean = cv2.boxFilter(gray, cv2.CV_32F, k_size)
    sq_mean = cv2.boxFilter(gray**2, cv2.CV_32F, k_size)
    local_var = np.maximum(sq_mean - mean**2, 0.0)
    local_std = np.sqrt(local_var)

    # 2. Local Gradient Energy
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.boxFilter(gx**2 + gy**2, cv2.CV_32F, k_size)

    # Thresholding: fingerprints have strong alternating ridge/valley contrast
    std_max = float(np.percentile(local_std, 95))
    energy_max = float(np.percentile(energy, 95))

    if std_max < 1e-3 or energy_max < 1e-3:
        # Uniform image - no fingerprint
        return np.zeros((h, w), dtype=np.uint8)

    std_thresh = max(std_max * var_threshold_ratio, 4.0)
    energy_thresh = max(energy_max * 0.08, 10.0)

    raw_mask = ((local_std > std_thresh) | (energy > energy_thresh)).astype(np.uint8)

    # 3. Morphological Cleanup (close gaps between ridges, remove stray speckles)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block_size + 1, block_size + 1))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, block_size // 2), max(5, block_size // 2)))

    mask_closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, open_kernel)

    # 4. Connected Components & Hole Filling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_opened)
    total_area = h * w
    min_area = total_area * min_area_ratio

    clean_mask = np.zeros((h, w), dtype=np.uint8)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == lbl] = 1

    # If no component reached min_area, keep the largest non-background component if > 1%
    if np.count_nonzero(clean_mask) == 0 and num_labels > 1:
        largest_lbl = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if stats[largest_lbl, cv2.CC_STAT_AREA] > total_area * 0.01:
            clean_mask[labels == largest_lbl] = 1

    # Fill internal holes using contour reconstruction
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled_mask, contours, -1, 1, thickness=cv2.FILLED)

    return filled_mask


def compute_coherence(
    img: Union[np.ndarray, torch.Tensor],
    ksize: int = 3,
    sigma_smooth: float = 3.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute continuous local ridge coherence map in [0.0, 1.0].
    Coherence = sqrt(Phi_x^2 + Phi_y^2) / (Phi_norm + eps)
    """
    gray = to_grayscale_numpy(img)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)

    gxx = gx**2
    gyy = gy**2
    gxy = gx * gy

    vx = gxx - gyy
    vy = 2.0 * gxy
    vmag = gxx + gyy

    # Gaussian smoothing
    phi_x = cv2.GaussianBlur(vx, (0, 0), sigmaX=sigma_smooth, sigmaY=sigma_smooth)
    phi_y = cv2.GaussianBlur(vy, (0, 0), sigmaX=sigma_smooth, sigmaY=sigma_smooth)
    phi_mag = cv2.GaussianBlur(vmag, (0, 0), sigmaX=sigma_smooth, sigmaY=sigma_smooth)

    num = np.sqrt(phi_x**2 + phi_y**2)
    coherence = num / (phi_mag + eps)
    return np.clip(coherence, 0.0, 1.0)


def compute_orientation_curvature(
    theta: np.ndarray,
    mask: Optional[np.ndarray] = None,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Compute continuous orientation field curvature magnitude kappa = ||grad(theta)||.
    Uses continuous double-angle vectors (vx = cos(2*theta), vy = sin(2*theta))
    to avoid branch-cut 0/pi wrap-around discontinuities:
        kappa = 0.5 * sqrt((dvx/dx)^2 + (dvx/dy)^2 + (dvy/dx)^2 + (dvy/dy)^2)
    Gaussian smoothed with parameter sigma.

    Returns:
        np.ndarray: Curvature magnitude map of shape (H, W), dtype float32.
    """
    vx = np.cos(2.0 * theta).astype(np.float32)
    vy = np.sin(2.0 * theta).astype(np.float32)

    dx_vx = cv2.Sobel(vx, cv2.CV_32F, 1, 0, ksize=3)
    dy_vx = cv2.Sobel(vx, cv2.CV_32F, 0, 1, ksize=3)
    dx_vy = cv2.Sobel(vy, cv2.CV_32F, 1, 0, ksize=3)
    dy_vy = cv2.Sobel(vy, cv2.CV_32F, 0, 1, ksize=3)

    kappa_raw = 0.5 * np.sqrt(dx_vx**2 + dy_vx**2 + dx_vy**2 + dy_vy**2)

    if sigma > 0.0:
        kappa = cv2.GaussianBlur(kappa_raw, (0, 0), sigmaX=sigma, sigmaY=sigma)
    else:
        kappa = kappa_raw

    if mask is not None:
        kappa = kappa * (mask > 0).astype(np.float32)

    return kappa.astype(np.float32)


def compute_orientation_field(
    img: Union[np.ndarray, torch.Tensor],
    block_size: int = 16,
    ksize: int = 3,
    sigma_smooth: float = 3.0,
    foreground_mask: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> OrientationResult:
    """
    Continuous Ridge Orientation Field (OF) estimation using normalized squared gradients
    (Bazen & Gerez / Rao method) with Gaussian window smoothing.

    Args:
        img: Input image (2D grayscale numpy array, torch tensor, or PIL Image).
        block_size: Spatial block size for block-level field aggregation (e.g. 8 or 16).
        ksize: Sobel kernel size (3 or 5).
        sigma_smooth: Gaussian smoothing standard deviation for orientation tensor integration.
        foreground_mask: Optional pre-computed binary mask {0, 1} (H, W). If None, computed automatically.
        eps: Small constant for numerical stability.

    Returns:
        OrientationResult: Full orientation fields, coherence maps, masks, and components.
    """
    gray = to_grayscale_numpy(img)
    h, w = gray.shape

    # 1. Image Gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)

    # 2. Squared Gradients / Double-Angle Components
    gxx = gx**2
    gyy = gy**2
    gxy = gx * gy

    vx_raw = gxx - gyy
    vy_raw = 2.0 * gxy
    vmag_raw = gxx + gyy

    # 3. Continuous Gaussian Window Smoothing (Bazen & Gerez integration)
    phi_x = cv2.GaussianBlur(vx_raw, (0, 0), sigmaX=sigma_smooth, sigmaY=sigma_smooth)
    phi_y = cv2.GaussianBlur(vy_raw, (0, 0), sigmaX=sigma_smooth, sigmaY=sigma_smooth)
    phi_mag = cv2.GaussianBlur(vmag_raw, (0, 0), sigmaX=sigma_smooth, sigmaY=sigma_smooth)

    # 4. Continuous Ridge Tangent Orientation
    # Gradient direction: phi = 0.5 * atan2(phi_y, phi_x)
    # Ridge tangent direction is orthogonal to gradient: theta = phi + pi/2
    phi_grad = 0.5 * np.arctan2(phi_y, phi_x)
    theta = np.mod(phi_grad + np.pi / 2.0, np.pi)

    # 5. Continuous Coherence Map
    tensor_norm = np.sqrt(phi_x**2 + phi_y**2)
    coherence = np.clip(tensor_norm / (phi_mag + eps), 0.0, 1.0)

    # 6. Foreground Mask
    if foreground_mask is None:
        mask = segment_foreground(gray, block_size=block_size)
    else:
        mask = (foreground_mask > 0).astype(np.uint8)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Zero-out orientation and coherence in background regions for cleanliness
    # (preserve raw continuous arrays in vx, vy)
    coherence = coherence * mask.astype(np.float32)

    # 7. Block-Level Field Aggregation
    # Compute block dimensions (rounded down to fit whole blocks)
    block_size = max(1, int(block_size))
    gh = h // block_size
    gw = w // block_size

    if gh > 0 and gw > 0:
        # Vectorized block pooling using boxFilter
        k_box = (block_size, block_size)
        b_x = cv2.boxFilter(phi_x, cv2.CV_32F, k_box)
        b_y = cv2.boxFilter(phi_y, cv2.CV_32F, k_box)
        b_mag = cv2.boxFilter(phi_mag, cv2.CV_32F, k_box)
        b_mask = cv2.boxFilter(mask.astype(np.float32), cv2.CV_32F, k_box)

        # Sample at block centers: step = block_size, offset = block_size // 2
        half_b = block_size // 2
        sub_y = slice(half_b, half_b + gh * block_size, block_size)
        sub_x = slice(half_b, half_b + gw * block_size, block_size)

        sample_x = b_x[sub_y, sub_x]
        sample_y = b_y[sub_y, sub_x]
        sample_mag = b_mag[sub_y, sub_x]
        sample_mask_f = b_mask[sub_y, sub_x]

        block_grad = 0.5 * np.arctan2(sample_y, sample_x)
        block_theta = np.mod(block_grad + np.pi / 2.0, np.pi)

        block_tensor_norm = np.sqrt(sample_x**2 + sample_y**2)
        block_coherence = np.clip(block_tensor_norm / (sample_mag + eps), 0.0, 1.0)
        block_mask = (sample_mask_f >= 0.5).astype(np.uint8)

        block_coherence = block_coherence * block_mask.astype(np.float32)
    else:
        block_theta = np.empty((0, 0), dtype=np.float32)
        block_coherence = np.empty((0, 0), dtype=np.float32)
        block_mask = np.empty((0, 0), dtype=np.uint8)

    return OrientationResult(
        theta=theta.astype(np.float32),
        coherence=coherence.astype(np.float32),
        mask=mask.astype(np.uint8),
        vx=phi_x.astype(np.float32),
        vy=phi_y.astype(np.float32),
        block_theta=block_theta.astype(np.float32),
        block_coherence=block_coherence.astype(np.float32),
        block_mask=block_mask.astype(np.uint8),
        block_size=block_size,
    )
