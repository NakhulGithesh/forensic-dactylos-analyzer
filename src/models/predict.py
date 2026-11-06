"""
Fingerprint Inference & Anatomical Mapping Engine.
Maps NIST pattern predictions (A, T, W, L, R) to anatomical classifications
(Plain Arch, Tented Arch, Whorl, Ulnar Loop, Radial Loop) using finger and hand metadata.
"""

from dataclasses import dataclass, asdict
from typing import Union, Optional, Dict, Any, List, Tuple
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import (
    NIST_CLASSES,
    NIST_CLASS_TO_IDX,
    IDX_TO_NIST_CLASS,
    get_default_device,
)
from src.data.transforms import FingerprintTransformPipeline
from src.features.transforms import OrientationAuxiliaryTransform

DIGIT_CODE_TO_HAND = {
    "01": "Right", "02": "Right", "03": "Right", "04": "Right", "05": "Right",
    "06": "Left",  "07": "Left",  "08": "Left",  "09": "Left",  "10": "Left",
}

DIGIT_CODE_TO_FINGER = {
    "01": (1, "Thumb"),  "02": (2, "Index"),  "03": (3, "Middle"), "04": (4, "Ring"),  "05": (5, "Little"),
    "06": (6, "Thumb"),  "07": (7, "Index"),  "08": (8, "Middle"), "09": (9, "Ring"),  "10": (10, "Little"),
}

FINGER_NUM_TO_NAME = {
    1: "Thumb", 2: "Index", 3: "Middle", 4: "Ring", 5: "Little",
    6: "Thumb", 7: "Index", 8: "Middle", 9: "Ring", 10: "Little",
}


def infer_hand(
    hand: Optional[str] = None,
    finger_num: Optional[Union[int, str]] = None,
    digit_code: Optional[str] = None,
) -> Optional[str]:
    """
    Deduce hand ('Right' or 'Left') from direct argument, digit_code ('01'-'10'), or finger_num (1-10).
    """
    if hand is not None and str(hand).strip():
        h = str(hand).strip().capitalize()
        if h in ("Right", "Left"):
            return h

    if digit_code is not None:
        code_str = str(digit_code).strip().zfill(2)
        if code_str in DIGIT_CODE_TO_HAND:
            return DIGIT_CODE_TO_HAND[code_str]

    if finger_num is not None:
        try:
            f_int = int(finger_num)
            if 1 <= f_int <= 5:
                return "Right"
            elif 6 <= f_int <= 10:
                return "Left"
        except (ValueError, TypeError):
            pass

    return None


def map_to_anatomical(
    pred_char: str,
    hand: Optional[str] = None,
    finger_num: Optional[Union[int, str]] = None,
    digit_code: Optional[str] = None,
) -> str:
    """
    Map NIST pattern code to dermatoglyphic anatomical classification.
    
    Rules (Henry System / FBI / NIST Standards):
    - 'A': Plain Arch
    - 'T': Tented Arch
    - 'W': Whorl
    - 'R' (Right-slant Loop):
        Right Hand -> Ulnar Loop (flows toward little finger / ulna bone)
        Left Hand  -> Radial Loop (flows toward thumb / radius bone)
    - 'L' (Left-slant Loop):
        Right Hand -> Radial Loop (flows toward thumb / radius bone)
        Left Hand  -> Ulnar Loop (flows toward little finger / ulna bone)
    """
    pred_char = pred_char.upper().strip()
    resolved_hand = infer_hand(hand=hand, finger_num=finger_num, digit_code=digit_code)

    if pred_char == "A":
        return "Plain Arch"
    elif pred_char == "T":
        return "Tented Arch"
    elif pred_char == "W":
        return "Whorl"
    elif pred_char == "R":
        if resolved_hand == "Right":
            return "Ulnar Loop"
        elif resolved_hand == "Left":
            return "Radial Loop"
        return "Right Loop (Hand needed for Ulnar/Radial)"
    elif pred_char == "L":
        if resolved_hand == "Right":
            return "Radial Loop"
        elif resolved_hand == "Left":
            return "Ulnar Loop"
        return "Left Loop (Hand needed for Ulnar/Radial)"
    else:
        return f"Unknown ({pred_char})"


@dataclass
class FingerprintPrediction:
    """Container for model inference and anatomical translation."""
    predicted_class: str
    predicted_idx: int
    confidence: float
    probabilities: Dict[str, float]
    anatomical_label: str
    hand: Optional[str]
    finger_num: Optional[int]
    finger_name: Optional[str]
    top_k: List[Tuple[str, float, str]]  # [(class, prob, anatomical_label), ...]
    logits: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def predict_fingerprint(
    model: nn.Module,
    image: Union[str, np.ndarray, torch.Tensor],
    transform: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    hand: Optional[str] = None,
    finger_num: Optional[Union[int, str]] = None,
    digit_code: Optional[str] = None,
    top_k: int = 5,
) -> FingerprintPrediction:
    """
    Perform single-image inference with full confidence calibration and anatomical mapping.

    Args:
        model: Trained PyTorch FingerprintClassifier or compatible nn.Module.
        image: File path (str), NumPy 2D array, or pre-transformed torch.Tensor.
        transform: Image transformation callable. Defaults to Stage 4 hybrid pipeline.
        device: Target torch device ('mps', 'cpu'). Auto-selected if None.
        hand: 'Right' or 'Left' (optional).
        finger_num: 1-10 (optional).
        digit_code: '01'-'10' (optional).
        top_k: Number of ranked predictions to return (up to 5).
    """
    if device is None:
        # Infer device from model parameters
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = get_default_device()
    else:
        device = torch.device(device)

    # 1. Image loading and preprocessing
    model_in_channels = getattr(model, "in_channels", 1)
    if transform is None:
        if model_in_channels == 3:
            transform = OrientationAuxiliaryTransform(
                target_size=(256, 256),
                enhancement="hybrid",
                mode="sincos",
            )
        else:
            transform = FingerprintTransformPipeline(
                target_size=(256, 256),
                enhancement="hybrid",
                channels=model_in_channels,
            )

    if isinstance(image, str):
        if not os.path.isfile(image):
            raise FileNotFoundError(f"Image not found at path: {image}")
        img_raw = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            raise IOError(f"Could not read image from {image}")
        tensor = transform(img_raw)
    elif isinstance(image, np.ndarray):
        tensor = transform(image)
    elif isinstance(image, torch.Tensor):
        if image.ndim == 4 and image.shape[0] > 1:
            raise ValueError(
                f"predict_fingerprint expects a single image tensor (C, H, W) or (1, C, H, W). "
                f"For batch inference with B={image.shape[0]}, use predict_batch()."
            )
        tensor = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Ensure shape (1, C, H, W)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device=device, dtype=torch.float32)

    # 2. Forward inference
    model.eval()
    with torch.no_grad():
        logits_tensor = model(tensor)
        probs_tensor = F.softmax(logits_tensor, dim=1)

    logits = logits_tensor.squeeze(0).cpu().tolist()
    probs = probs_tensor.squeeze(0).cpu().tolist()

    pred_idx = int(np.argmax(probs))
    pred_char = IDX_TO_NIST_CLASS[pred_idx]
    confidence = float(probs[pred_idx])

    prob_dict = {IDX_TO_NIST_CLASS[i]: float(probs[i]) for i in range(len(probs))}

    # Resolved hand and finger metadata
    resolved_hand = infer_hand(hand=hand, finger_num=finger_num, digit_code=digit_code)
    f_num = None
    f_name = None
    if digit_code is not None and str(digit_code).zfill(2) in DIGIT_CODE_TO_FINGER:
        f_num, f_name = DIGIT_CODE_TO_FINGER[str(digit_code).zfill(2)]
    elif finger_num is not None:
        try:
            f_num = int(finger_num)
            f_name = FINGER_NUM_TO_NAME.get(f_num)
        except (ValueError, TypeError):
            pass

    anat_label = map_to_anatomical(
        pred_char=pred_char,
        hand=resolved_hand,
        finger_num=f_num,
        digit_code=digit_code,
    )

    # Top-K ranking
    k = min(max(1, top_k), len(probs))
    sorted_indices = np.argsort(probs)[::-1][:k]
    top_k_list = [
        (
            IDX_TO_NIST_CLASS[i],
            float(probs[i]),
            map_to_anatomical(
                IDX_TO_NIST_CLASS[i],
                hand=resolved_hand,
                finger_num=f_num,
                digit_code=digit_code,
            ),
        )
        for i in sorted_indices
    ]

    return FingerprintPrediction(
        predicted_class=pred_char,
        predicted_idx=pred_idx,
        confidence=confidence,
        probabilities=prob_dict,
        anatomical_label=anat_label,
        hand=resolved_hand,
        finger_num=f_num,
        finger_name=f_name,
        top_k=top_k_list,
        logits=logits,
    )


def predict_batch(
    model: nn.Module,
    images: Union[List[Union[str, np.ndarray, torch.Tensor]], torch.Tensor],
    transform: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5,
) -> List[FingerprintPrediction]:
    """
    Batched inference returning a list of FingerprintPrediction objects.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = get_default_device()
    else:
        device = torch.device(device)

    model_in_channels = getattr(model, "in_channels", 1)
    if transform is None:
        if model_in_channels == 3:
            transform = OrientationAuxiliaryTransform(
                target_size=(256, 256),
                enhancement="hybrid",
                mode="sincos",
            )
        else:
            transform = FingerprintTransformPipeline(
                target_size=(256, 256),
                enhancement="hybrid",
                channels=model_in_channels,
            )

    # Convert images to batched tensor
    if isinstance(images, torch.Tensor):
        if images.ndim == 3:
            batch_tensor = images.unsqueeze(1)
        elif images.ndim == 4:
            batch_tensor = images
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {images.shape}")
    elif isinstance(images, (list, tuple)):
        tensor_list = []
        for img in images:
            if isinstance(img, str):
                raw = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                t = transform(raw)
            elif isinstance(img, np.ndarray):
                t = transform(img)
            elif isinstance(img, torch.Tensor):
                t = img
            else:
                raise TypeError(f"Unsupported item type in batch: {type(img)}")
            if t.ndim == 2:
                t = t.unsqueeze(0)
            tensor_list.append(t)
        batch_tensor = torch.stack(tensor_list, dim=0)
    else:
        raise TypeError(f"Unsupported batch input type: {type(images)}")

    b_size = batch_tensor.size(0)
    batch_tensor = batch_tensor.to(device=device, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits_tensor = model(batch_tensor)
        probs_tensor = F.softmax(logits_tensor, dim=1)

    logits_arr = logits_tensor.cpu().tolist()
    probs_arr = probs_tensor.cpu().tolist()

    results: List[FingerprintPrediction] = []
    for i in range(b_size):
        meta = metadata_list[i] if (metadata_list and i < len(metadata_list)) else {}
        hand = meta.get("hand")
        finger_num = meta.get("finger_num")
        digit_code = meta.get("digit_code")

        cur_logits = logits_arr[i]
        cur_probs = probs_arr[i]

        pred_idx = int(np.argmax(cur_probs))
        pred_char = IDX_TO_NIST_CLASS[pred_idx]
        confidence = float(cur_probs[pred_idx])

        prob_dict = {IDX_TO_NIST_CLASS[c]: float(cur_probs[c]) for c in range(len(cur_probs))}

        resolved_hand = infer_hand(hand=hand, finger_num=finger_num, digit_code=digit_code)
        f_num = None
        f_name = None
        if digit_code is not None and str(digit_code).zfill(2) in DIGIT_CODE_TO_FINGER:
            f_num, f_name = DIGIT_CODE_TO_FINGER[str(digit_code).zfill(2)]
        elif finger_num is not None:
            try:
                f_num = int(finger_num)
                f_name = FINGER_NUM_TO_NAME.get(f_num)
            except (ValueError, TypeError):
                pass

        anat_label = map_to_anatomical(
            pred_char=pred_char,
            hand=resolved_hand,
            finger_num=f_num,
            digit_code=digit_code,
        )

        k = min(max(1, top_k), len(cur_probs))
        sorted_indices = np.argsort(cur_probs)[::-1][:k]
        top_k_list = [
            (
                IDX_TO_NIST_CLASS[c],
                float(cur_probs[c]),
                map_to_anatomical(
                    IDX_TO_NIST_CLASS[c],
                    hand=resolved_hand,
                    finger_num=f_num,
                    digit_code=digit_code,
                ),
            )
            for c in sorted_indices
        ]

        results.append(
            FingerprintPrediction(
                predicted_class=pred_char,
                predicted_idx=pred_idx,
                confidence=confidence,
                probabilities=prob_dict,
                anatomical_label=anat_label,
                hand=resolved_hand,
                finger_num=f_num,
                finger_name=f_name,
                top_k=top_k_list,
                logits=cur_logits,
            )
        )

    return results
