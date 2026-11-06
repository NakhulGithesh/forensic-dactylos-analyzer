"""
PyTorch Dataset and DataLoader Harness for Fingerprint Data.
Reads Stage 3 Manifests, guarantees read-only access to raw datasets,
and yields batches on device ('mps' or 'cpu').
"""

import os
import csv
from typing import Dict, Any, List, Optional, Union, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.transforms import FingerprintTransformPipeline

NIST_CLASS_TO_IDX = {"A": 0, "T": 1, "W": 2, "L": 3, "R": 4}
IDX_TO_NIST_CLASS = {0: "A", 1: "T", 2: "W", 3: "L", 4: "R"}

SOCOFING_ALT_TO_IDX = {"None": 0, "Obl": 1, "CR": 2, "Zcut": 3}
IDX_TO_SOCOFING_ALT = {0: "None", 1: "Obl", 2: "CR", 3: "Zcut"}

class FingerprintDataset(Dataset):
    """
    Modular PyTorch Dataset for NIST SD04 and SOCOFing.
    Loads raw images in strict read-only mode and applies deterministic transformations.
    """

    def __init__(
        self,
        manifest_path: str,
        root_dir: Optional[str] = None,
        transform: Optional[FingerprintTransformPipeline] = None,
        label_column: Optional[str] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        return_metadata: bool = True,
        cache_transformed: bool = False,
    ):
        if root_dir is None:
            # Assume project root is 2 directories up from src/data/
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.root_dir = os.path.abspath(root_dir)

        if not os.path.isabs(manifest_path):
            manifest_path = os.path.join(self.root_dir, manifest_path)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self.manifest_path = manifest_path
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.records: List[Dict[str, str]] = list(csv.DictReader(f))

        if len(self.records) == 0:
            raise ValueError(f"Manifest {manifest_path} contains 0 rows.")

        self.transform = transform if transform is not None else FingerprintTransformPipeline()
        self.return_metadata = return_metadata
        self.cache_transformed = cache_transformed
        self._cache: Dict[int, Any] = {}

        # Infer dataset schema
        sample = self.records[0]
        if "primary_class" in sample:
            self.dataset_type = "nist_sd04"
            self.default_label_col = "primary_class"
            default_mapping = NIST_CLASS_TO_IDX
        elif "alteration_type" in sample:
            self.dataset_type = "socofing"
            self.default_label_col = "alteration_type"
            default_mapping = SOCOFING_ALT_TO_IDX
        else:
            self.dataset_type = "unknown"
            self.default_label_col = label_column or list(sample.keys())[1]
            default_mapping = {}

        self.label_column = label_column or self.default_label_col

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        elif label_column is None or label_column == self.default_label_col:
            self.class_to_idx = default_mapping
        else:
            # Custom label column without explicit mapping:
            # Check if values are already integer-like
            unique_vals = sorted(list({r.get(self.label_column, "") for r in self.records if self.label_column in r}))
            all_int = True
            for v in unique_vals:
                try:
                    int(v)
                except ValueError:
                    all_int = False
                    break
            if all_int:
                self.class_to_idx = {v: int(v) for v in unique_vals}
            else:
                self.class_to_idx = {v: idx for idx, v in enumerate(unique_vals)}

    def __len__(self) -> int:
        return len(self.records)

    def preload(self) -> None:
        """Pre-populate transformed cache across all records in manifest."""
        self.cache_transformed = True
        for i in range(len(self.records)):
            _ = self[i]

    def clear_cache(self) -> None:
        """Clear the transformed memory cache."""
        self._cache.clear()

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int, Dict[str, Any]], Tuple[torch.Tensor, int]]:
        if self.cache_transformed and idx in self._cache:
            return self._cache[idx]

        record = self.records[idx]
        rel_path = record["image_path"]
        full_path = os.path.join(self.root_dir, rel_path)

        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Image not found at path: {full_path}")

        # Strict read-only image loading via cv2 (flags: IMREAD_GRAYSCALE)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to read image at: {full_path}")

        # Apply transformation pipeline
        tensor_img = self.transform(img)

        # Map target label
        raw_label = record.get(self.label_column, "")
        if raw_label in self.class_to_idx:
            target_idx = self.class_to_idx[raw_label]
        else:
            try:
                target_idx = int(raw_label)
            except (ValueError, TypeError):
                target_idx = -1

        if self.return_metadata:
            metadata = dict(record)
            result = (tensor_img, target_idx, metadata)
        else:
            result = (tensor_img, target_idx)

        if self.cache_transformed:
            self._cache[idx] = result

        return result


class DeviceDataLoader:
    """
    DataLoader wrapper that automatically transfers yielded batches to a target device ('mps' or 'cpu').
    Delegates all other DataLoader attributes (dataset, batch_size, sampler, etc.) to the wrapped loader.
    """

    def __init__(self, dataloader: DataLoader, device: Union[str, torch.device]):
        self.dataloader = dataloader
        self.device = torch.device(device)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataloader, name)

    def __iter__(self):
        for batch in self.dataloader:
            if isinstance(batch, (list, tuple)):
                transferred = [
                    item.to(self.device, non_blocking=True) if isinstance(item, torch.Tensor) else item
                    for item in batch
                ]
                yield tuple(transferred) if isinstance(batch, tuple) else transferred
            elif isinstance(batch, dict):
                yield {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif isinstance(batch, torch.Tensor):
                yield batch.to(self.device, non_blocking=True)
            else:
                yield batch

    def __len__(self) -> int:
        return len(self.dataloader)


def create_fingerprint_dataloader(
    manifest_path: str,
    root_dir: Optional[str] = None,
    transform: Optional[FingerprintTransformPipeline] = None,
    label_column: Optional[str] = None,
    class_to_idx: Optional[Dict[str, int]] = None,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    return_metadata: bool = False,
    cache_transformed: bool = False,
) -> Union[DataLoader, DeviceDataLoader]:
    """
    Factory function for Fingerprint DataLoader with optional automatic device placement ('mps' or 'cpu').
    """
    dataset = FingerprintDataset(
        manifest_path=manifest_path,
        root_dir=root_dir,
        transform=transform,
        label_column=label_column,
        class_to_idx=class_to_idx,
        return_metadata=return_metadata,
        cache_transformed=cache_transformed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if device is not None:
        return DeviceDataLoader(loader, device=device)
    return loader
