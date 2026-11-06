"""
Image preprocessing and transformation pipeline for Fingerprint Datasets.
Designed for NIST SD04 (rolled ink) and SOCOFing (optical live-scan).
Deterministic, modular, and optimized for Apple Silicon (macOS).
"""

from typing import Union, Tuple, Optional, Any
import cv2
import numpy as np
import torch

def pad_to_aspect_ratio(img: np.ndarray, pad_value: int = 255) -> np.ndarray:
    """
    Symmetric constant padding to make the image square (1:1 aspect ratio).
    Preserves 100% of fingerprint ridge content without vertical or horizontal clipping.
    """
    h, w = img.shape[:2]
    if h == w:
        return img
    elif h > w:
        diff = h - w
        pad_left = diff // 2
        pad_right = diff - pad_left
        return cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_value)
    else:
        diff = w - h
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        return cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=pad_value)

def center_crop_to_square(img: np.ndarray) -> np.ndarray:
    """Center crops the image to min(H, W) x min(H, W)."""
    h, w = img.shape[:2]
    if h == w:
        return img
    min_dim = min(h, w)
    start_y = (h - min_dim) // 2
    start_x = (w - min_dim) // 2
    return img[start_y:start_y + min_dim, start_x:start_x + min_dim]

def enhance_clahe(img: np.ndarray, clip_limit: float = 1.5, tile_grid_size: Tuple[int, int] = (4, 4)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def enhance_bandpass_dog(img: np.ndarray, sigma1: float = 1.0, sigma2: float = 3.0) -> np.ndarray:
    """Difference of Gaussians (DoG) spatial bandpass ridge filter."""
    f = img.astype(np.float32)
    g1 = cv2.GaussianBlur(f, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(f, (0, 0), sigma2)
    dog = g1 - g2
    dog_norm = cv2.normalize(dog, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return dog_norm.astype(np.uint8)

def enhance_baseline_minmax(img: np.ndarray) -> np.ndarray:
    """Baseline Min-Max contrast scaling to full [0, 255] uint8 range."""
    f = img.astype(np.float32)
    mi, ma = np.min(f), np.max(f)
    if ma > mi:
        res = (f - mi) / (ma - mi) * 255.0
    else:
        res = f
    return res.astype(np.uint8)

def enhance_baseline_zscore(img: np.ndarray) -> np.ndarray:
    """Standard Z-Score normalization clipped to [-3, 3] and mapped to [0, 255]."""
    f = img.astype(np.float32)
    mu, std = np.mean(f), np.std(f) + 1e-6
    z = (f - mu) / std
    z_clipped = np.clip(z, -3.0, 3.0)
    res = (z_clipped + 3.0) / 6.0 * 255.0
    return res.astype(np.uint8)

def enhance_percentile_stretch(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Contrast stretching with percentile noise-floor clipping."""
    f = img.astype(np.float32)
    v_low = np.percentile(f, p_low)
    v_high = np.percentile(f, p_high)
    if v_high > v_low:
        clipped = np.clip(f, v_low, v_high)
        res = (clipped - v_low) / (v_high - v_low) * 255.0
    else:
        res = f
    return res.astype(np.uint8)

def enhance_recommended_hybrid(img: np.ndarray) -> np.ndarray:
    """
    Stage 4 Recommended Hybrid Pipeline:
    1. CLAHE with conservative 4x4 tile grid & 1.5 clip limit (enhances local ridge-valley gradient).
    2. Soft percentile stretch [1%, 99%] (suppresses paper grain noise in NIST SD04 and optical saturation in SOCOFing).
    """
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    cl = clahe.apply(img)
    p1, p99 = np.percentile(cl, 1.0), np.percentile(cl, 99.0)
    if p99 > p1:
        st = np.clip(cl, p1, p99)
        out = (st - p1) / (p99 - p1) * 255.0
    else:
        out = cl
    return out.astype(np.uint8)

INTERPOLATION_MODES = {
    "lanczos": cv2.INTER_LANCZOS4,
    "lanczos4": cv2.INTER_LANCZOS4,
    "bicubic": cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}

class FingerprintTransformPipeline:
    """
    Deterministic Image Preprocessing Pipeline for Fingerprints.
    """

    def __init__(
        self,
        target_size: Union[int, Tuple[int, int]] = (256, 256),
        crop_mode: str = "auto",  # 'auto', 'nist_bottom', 'nist_square', 'conservative', 'none'
        aspect_mode: str = "auto",  # 'auto', 'pad', 'center_crop', 'none'
        pad_value: int = 255,
        interpolation: str = "lanczos",
        enhancement: str = "hybrid",  # 'hybrid', 'clahe_small', 'clahe_std', 'dog', 'percentile', 'minmax', 'zscore', 'none'
        channels: int = 1,  # 1 (grayscale) or 3 (replicated RGB)
        normalization: str = "zero_one",  # 'zero_one' [0, 1], 'neg_one_one' [-1, 1], 'standardize', 'imagenet', 'none' [0, 255]
    ):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = (int(target_size[0]), int(target_size[1]))

        self.crop_mode = crop_mode.lower()
        self.aspect_mode = aspect_mode.lower()
        self.pad_value = int(pad_value)
        if isinstance(interpolation, int):
            self.interpolation = interpolation
            self.interpolation_str = str(interpolation)
        else:
            self.interpolation_str = str(interpolation).lower()
            self.interpolation = INTERPOLATION_MODES.get(self.interpolation_str, cv2.INTER_LANCZOS4)
        self.enhancement = enhancement.lower()
        self.channels = int(channels)
        self.normalization = normalization.lower()

    def __call__(self, img: Any) -> torch.Tensor:
        """
        Process an input image (numpy array, PIL Image, or PyTorch tensor)
        and return a torch.FloatTensor of shape (C, H, W).
        """
        if img is None:
            raise ValueError("Input image cannot be None or empty.")

        is_pil = False
        # Convert PIL Image or PyTorch Tensor to numpy array
        if hasattr(img, "detach") and hasattr(img, "cpu"):  # torch.Tensor
            img = img.detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < min(img.shape[1], img.shape[2]):
                img = np.transpose(img, (1, 2, 0))
        elif hasattr(img, "size") and not isinstance(img, np.ndarray):  # PIL Image
            is_pil = True
            img = np.array(img)

        if not isinstance(img, np.ndarray) or img.size == 0:
            raise ValueError("Input image cannot be None or empty.")

        # Ensure single channel 2D array
        if img.ndim == 3:
            if img.shape[2] == 1:
                img = img[:, :, 0]
            elif is_pil:
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Ensure uint8 dtype for OpenCV functions
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                min_v = float(img.min())
                max_v = float(img.max())
                if min_v >= -0.0001 and max_v <= 1.0001:
                    # [0.0, 1.0] float range
                    img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
                elif min_v >= -1.0001 and max_v <= 1.0001 and min_v < -0.0001:
                    # [-1.0, 1.0] float range
                    img = ((np.clip(img, -1.0, 1.0) + 1.0) / 2.0 * 255.0).astype(np.uint8)
                elif min_v >= 0.0 and max_v <= 255.0:
                    # [0.0, 255.0] float range
                    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
                elif max_v > min_v:
                    # Standardized (e.g. [-3, 3]) or arbitrary float range -> contrast stretch to [0, 255]
                    img = ((img - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
                else:
                    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        h, w = img.shape[:2]
        was_nist_crop = False

        # 1. Boundary Cropping
        if self.crop_mode == "auto":
            # For NIST SD04 512x512 images, rows 480-511 are 100% white border padding
            if h == 512 and w == 512:
                img = img[:480, :]
                was_nist_crop = True
        elif self.crop_mode == "nist_bottom":
            if h >= 480:
                img = img[:480, :]
                was_nist_crop = True
        elif self.crop_mode == "nist_square":
            if h >= 480 and w >= 496:
                img = img[:480, 16:496]
                was_nist_crop = True
        elif self.crop_mode in ("conservative", "nist_card"):
            if h >= 480:
                img = img[16:480, :]
                was_nist_crop = True
        elif self.crop_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown crop_mode: {self.crop_mode}")

        # 2. Aspect Ratio Handling
        if self.aspect_mode == "auto":
            # Automatically pad rectangular sensor captures (e.g. SOCOFing 96x103 or 241x298) to square.
            # Do NOT re-pad NIST SD04 after cropping bottom scanner padding, avoiding artificial white bars.
            if not was_nist_crop and img.shape[0] != img.shape[1]:
                img = pad_to_aspect_ratio(img, pad_value=self.pad_value)
        elif self.aspect_mode == "pad":
            if img.shape[0] != img.shape[1]:
                img = pad_to_aspect_ratio(img, pad_value=self.pad_value)
        elif self.aspect_mode == "center_crop":
            img = center_crop_to_square(img)
        elif self.aspect_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown aspect_mode: {self.aspect_mode}")

        # 3. Spatial Resizing
        cur_h, cur_w = img.shape[:2]
        tgt_h, tgt_w = self.target_size
        if (cur_h, cur_w) != (tgt_h, tgt_w):
            img = cv2.resize(img, (tgt_w, tgt_h), interpolation=self.interpolation)

        # 4. Contrast Normalization & Enhancement
        if self.enhancement == "hybrid":
            img = enhance_recommended_hybrid(img)
        elif self.enhancement == "clahe_small":
            img = enhance_clahe(img, clip_limit=1.5, tile_grid_size=(4, 4))
        elif self.enhancement in ("clahe_std", "clahe_standard"):
            img = enhance_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
        elif self.enhancement in ("dog", "bandpass"):
            img = enhance_bandpass_dog(img, sigma1=1.0, sigma2=3.0)
        elif self.enhancement == "percentile":
            img = enhance_percentile_stretch(img, p_low=1.0, p_high=99.0)
        elif self.enhancement == "minmax":
            img = enhance_baseline_minmax(img)
        elif self.enhancement in ("zscore", "z_score"):
            img = enhance_baseline_zscore(img)
        elif self.enhancement == "none":
            pass
        else:
            raise ValueError(f"Unknown enhancement strategy: {self.enhancement}")

        # 5. Tensor Conversion & Value Range Normalization
        tensor = torch.from_numpy(img).float()  # (H, W)

        if self.channels == 1:
            tensor = tensor.unsqueeze(0)  # (1, H, W)
        elif self.channels == 3:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        else:
            raise ValueError(f"Unsupported channel count: {self.channels} (must be 1 or 3)")

        if self.normalization == "zero_one":
            tensor = tensor / 255.0  # [0.0, 1.0]
        elif self.normalization == "neg_one_one":
            tensor = (tensor / 127.5) - 1.0  # [-1.0, 1.0]
        elif self.normalization == "standardize":
            mean = tensor.mean()
            std = tensor.std() + 1e-7
            tensor = (tensor - mean) / std
        elif self.normalization == "imagenet":
            tensor = tensor / 255.0
            if self.channels == 3:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            else:
                mean = torch.tensor([0.449]).view(1, 1, 1)
                std = torch.tensor([0.226]).view(1, 1, 1)
            tensor = (tensor - mean) / std
        elif self.normalization == "none":
            pass
        else:
            raise ValueError(f"Unknown normalization mode: {self.normalization}")

        return tensor
