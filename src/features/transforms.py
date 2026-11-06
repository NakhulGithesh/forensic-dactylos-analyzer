"""
Multi-Channel Transform Pipeline with Stage 5 Auxiliary Features.
Injects continuous ridge orientation and coherence maps as auxiliary tensor channels.
"""

from typing import Union, Tuple, Optional, Any
import cv2
import numpy as np
import torch

from src.data.transforms import FingerprintTransformPipeline
from src.features.orientation import compute_orientation_field


class OrientationAuxiliaryTransform:
    """
    Transforms fingerprint images into multi-channel tensors injecting
    Stage 5 continuous orientation and coherence maps as auxiliary channels.

    Modes:
    - 'sincos' (3 channels):
        Channel 0: Hybrid Enhanced Grayscale [0, 1]
        Channel 1: Orientation Sine (sin(2*theta) + 1)/2 in [0, 1]
        Channel 2: Orientation Cosine (cos(2*theta) + 1)/2 in [0, 1]
    - 'sin_coherence' (3 channels):
        Channel 0: Hybrid Enhanced Grayscale [0, 1]
        Channel 1: Orientation Sine (sin(2*theta) + 1)/2 in [0, 1]
        Channel 2: Continuous Local Ridge Coherence in [0, 1]
    - 'sincos_coherence' (4 channels):
        Channel 0: Hybrid Enhanced Grayscale [0, 1]
        Channel 1: Orientation Sine [0, 1]
        Channel 2: Orientation Cosine [0, 1]
        Channel 3: Continuous Local Ridge Coherence [0, 1]
    """

    def __init__(
        self,
        target_size: Union[int, Tuple[int, int]] = (256, 256),
        enhancement: str = "hybrid",
        mode: str = "sincos",  # 'sincos', 'sin_coherence', 'sincos_coherence'
        block_size: int = 16,
        mask_background: bool = True,
    ):
        self.base_pipeline = FingerprintTransformPipeline(
            target_size=target_size,
            enhancement=enhancement,
            channels=1,
            normalization="zero_one",
        )
        self.mode = mode.lower()
        self.block_size = int(block_size)
        self.mask_background = bool(mask_background)

        if self.mode in ("sincos", "sin_coherence"):
            self.channels = 3
        elif self.mode == "sincos_coherence":
            self.channels = 4
        else:
            raise ValueError(f"Unknown auxiliary mode: '{mode}'. Supported: ['sincos', 'sin_coherence', 'sincos_coherence']")

    def __call__(self, img: Any) -> torch.Tensor:
        """
        Process raw input image and return multi-channel FloatTensor (C, H, W).
        """
        # 1. Base grayscale transformation
        base_tensor = self.base_pipeline(img)  # (1, H, W) in [0.0, 1.0]
        base_arr = base_tensor[0].cpu().numpy()  # (H, W) in [0.0, 1.0]

        # Convert to uint8 for OpenCV orientation computation
        u8_arr = np.clip(base_arr * 255.0, 0, 255).astype(np.uint8)

        # 2. Stage 5 continuous orientation and coherence
        orient = compute_orientation_field(u8_arr, block_size=self.block_size)
        theta = orient.theta  # in [0, pi)
        coherence = orient.coherence  # in [0.0, 1.0]
        mask = orient.mask > 0  # binary foreground

        # Continuous double angle components
        sin_2theta = (np.sin(2.0 * theta) + 1.0) / 2.0  # mapped to [0, 1]
        cos_2theta = (np.cos(2.0 * theta) + 1.0) / 2.0  # mapped to [0, 1]

        if self.mask_background:
            # Set background regions to neutral values (0.5 for sine/cosine, 0.0 for coherence)
            sin_2theta = np.where(mask, sin_2theta, 0.5)
            cos_2theta = np.where(mask, cos_2theta, 0.5)
            coherence = np.where(mask, coherence, 0.0)

        # 3. Channel assembly
        if self.mode == "sincos":
            stacked = np.stack([base_arr, sin_2theta, cos_2theta], axis=0)
        elif self.mode == "sin_coherence":
            stacked = np.stack([base_arr, sin_2theta, coherence], axis=0)
        elif self.mode == "sincos_coherence":
            stacked = np.stack([base_arr, sin_2theta, cos_2theta, coherence], axis=0)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return torch.from_numpy(stacked).float()
