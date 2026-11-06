"""
Automated Unit Tests for Stage 4: Image Preprocessing Pipeline & Data Harness.
Verifies tensor dimensions, value ranges, MPS compatibility, repeatability,
aspect padding, and raw dataset read-only integrity.
"""

import os
import sys
import hashlib
import pytest
import cv2
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PIL import Image
from src.data.transforms import (
    FingerprintTransformPipeline,
    pad_to_aspect_ratio,
    center_crop_to_square,
    enhance_clahe,
    enhance_bandpass_dog,
    enhance_percentile_stretch,
    enhance_recommended_hybrid,
    enhance_baseline_minmax,
    enhance_baseline_zscore,
)
from src.data.dataset import (
    FingerprintDataset,
    DeviceDataLoader,
    create_fingerprint_dataloader,
    NIST_CLASS_TO_IDX,
    IDX_TO_NIST_CLASS,
    SOCOFING_ALT_TO_IDX,
)

@pytest.fixture(scope="module")
def sample_nist_image_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0001_01.png")
    assert os.path.isfile(path), f"NIST sample not found: {path}"
    return path

@pytest.fixture(scope="module")
def sample_socofing_std_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/SOCOFing/Real/100__M_Left_index_finger.BMP")
    assert os.path.isfile(path), f"SOCOFing standard sample not found: {path}"
    return path

@pytest.fixture(scope="module")
def sample_socofing_highres_path():
    path = os.path.join(PROJECT_ROOT, "Fingerprint Data/SOCOFing/Real/226__M_Left_index_finger.BMP")
    assert os.path.isfile(path), f"SOCOFing highres sample not found: {path}"
    return path


# ---------------------------------------------------------------------
# 1. Output Dimensions & Channel Tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("target_res", [(256, 256), (224, 224), (160, 160)])
@pytest.mark.parametrize("channels", [1, 3])
def test_pipeline_output_dimensions(sample_nist_image_path, target_res, channels):
    img = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    pipeline = FingerprintTransformPipeline(
        target_size=target_res,
        channels=channels,
        enhancement="hybrid",
    )
    tensor = pipeline(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (channels, target_res[0], target_res[1])
    assert tensor.dtype == torch.float32

    if channels == 3:
        # Channels should be identical replications of the grayscale print
        assert torch.equal(tensor[0], tensor[1])
        assert torch.equal(tensor[1], tensor[2])


# ---------------------------------------------------------------------
# 2. Value Range & Normalization Modes
# ---------------------------------------------------------------------

def test_normalization_zero_one(sample_nist_image_path):
    img = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    pipeline = FingerprintTransformPipeline(normalization="zero_one")
    tensor = pipeline(img)
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0

def test_normalization_neg_one_one(sample_nist_image_path):
    img = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    pipeline = FingerprintTransformPipeline(normalization="neg_one_one")
    tensor = pipeline(img)
    assert tensor.min().item() >= -1.0
    assert tensor.max().item() <= 1.0

def test_normalization_standardize(sample_nist_image_path):
    img = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    pipeline = FingerprintTransformPipeline(normalization="standardize")
    tensor = pipeline(img)
    assert abs(tensor.mean().item()) < 1e-4
    assert abs(tensor.std().item() - 1.0) < 1e-2


# ---------------------------------------------------------------------
# 3. Determinism & Processing Repeatability
# ---------------------------------------------------------------------

def test_processing_repeatability(sample_nist_image_path):
    img = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    pipeline = FingerprintTransformPipeline(
        target_size=(256, 256),
        interpolation="lanczos",
        enhancement="hybrid",
        normalization="zero_one",
    )
    t1 = pipeline(img)
    t2 = pipeline(img)
    assert torch.equal(t1, t2), "Pipeline output is not bitwise deterministic across identical runs."


# ---------------------------------------------------------------------
# 4. Apple Silicon MPS Compatibility
# ---------------------------------------------------------------------

def test_mps_device_compatibility(sample_nist_image_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    img = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    pipeline = FingerprintTransformPipeline(target_size=(256, 256), channels=1)
    tensor = pipeline(img).unsqueeze(0).to(device)  # (1, 1, 256, 256)

    # Test forward operation on MPS
    conv = nn.Conv2d(1, 8, kernel_size=3, padding=1).to(device)
    out = conv(tensor)
    assert out.device.type == device.type
    assert out.shape == (1, 8, 256, 256)


# ---------------------------------------------------------------------
# 5. Aspect-Ratio Padding vs Cropping
# ---------------------------------------------------------------------

def test_aspect_ratio_padding_standard_socofing(sample_socofing_std_path):
    raw = cv2.imread(sample_socofing_std_path, cv2.IMREAD_GRAYSCALE)
    assert raw.shape == (103, 96)
    padded = pad_to_aspect_ratio(raw, pad_value=255)
    assert padded.shape == (103, 103)
    # Check that original 96 cols are centrally preserved
    diff = 103 - 96
    pad_left = diff // 2
    pad_right = diff - pad_left
    np.testing.assert_array_equal(padded[:, pad_left:pad_left + 96], raw)

def test_aspect_ratio_padding_highres_socofing(sample_socofing_highres_path):
    raw = cv2.imread(sample_socofing_highres_path, cv2.IMREAD_GRAYSCALE)
    assert raw.shape == (298, 241)
    padded = pad_to_aspect_ratio(raw, pad_value=255)
    assert padded.shape == (298, 298)
    diff = 298 - 241
    pad_left = diff // 2
    np.testing.assert_array_equal(padded[:, pad_left:pad_left + 241], raw)


# ---------------------------------------------------------------------
# 6. NIST SD04 Boundary Artifact Handling
# ---------------------------------------------------------------------

def test_nist_bottom_cropping_removes_white_border(sample_nist_image_path):
    raw = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    assert raw.shape == (512, 512)
    # Verify bottom 32 rows are 255.0
    assert np.all(raw[480:512, :] == 255)

    pipeline = FingerprintTransformPipeline(crop_mode="nist_bottom", aspect_mode="none", target_size=(480, 512), enhancement="none", normalization="none")
    out = pipeline(raw)
    assert out.shape == (1, 480, 512)


# ---------------------------------------------------------------------
# 7. Enhancement Algorithms Consistency
# ---------------------------------------------------------------------

@pytest.mark.parametrize("enhancement_mode", ["hybrid", "clahe_small", "clahe_std", "dog", "percentile", "minmax", "none"])
def test_enhancement_modes_execute(sample_nist_image_path, enhancement_mode):
    raw = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    pipeline = FingerprintTransformPipeline(
        target_size=(256, 256),
        enhancement=enhancement_mode,
        normalization="zero_one",
    )
    tensor = pipeline(raw)
    assert tensor.shape == (1, 256, 256)
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


# ---------------------------------------------------------------------
# 8. PyTorch Dataset & DataLoader Harness
# ---------------------------------------------------------------------

def test_nist_dataset_iteration():
    val_manifest = "manifests/nist_sd04_val.csv"
    dataset = FingerprintDataset(val_manifest, return_metadata=True)
    assert len(dataset) == 600

    img, target, meta = dataset[0]
    assert img.shape == (1, 256, 256)
    assert isinstance(target, int)
    assert 0 <= target <= 4
    assert meta["primary_class"] == IDX_TO_NIST_CLASS[target]
    assert "fingerprint_id" in meta
    assert "split" in meta and meta["split"] == "val"

def test_socofing_dataset_iteration():
    eval_manifest = "manifests/socofing_eval.csv"
    dataset = FingerprintDataset(eval_manifest, return_metadata=True)
    assert len(dataset) == 55270

    img, target, meta = dataset[0]
    assert img.shape == (1, 256, 256)
    assert isinstance(target, int)
    assert target in SOCOFING_ALT_TO_IDX.values()
    assert "subject_id" in meta

def test_dataloader_batching_and_device():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    loader = create_fingerprint_dataloader(
        "manifests/nist_sd04_val.csv",
        batch_size=16,
        device=device,
        return_metadata=False,
    )
    batch_imgs, batch_targets = next(iter(loader))
    assert batch_imgs.shape == (16, 1, 256, 256)
    assert batch_targets.shape == (16,)
    assert batch_imgs.device.type == torch.device(device).type
    assert batch_targets.device.type == torch.device(device).type


# ---------------------------------------------------------------------
# 9. Strict Read-Only Filesystem Integrity
# ---------------------------------------------------------------------

def test_raw_files_remain_unmodified(sample_nist_image_path):
    with open(sample_nist_image_path, "rb") as f:
        before_hash = hashlib.md5(f.read()).hexdigest()
    before_mtime = os.stat(sample_nist_image_path).st_mtime

    # Run through dataset and dataloader
    pipeline = FingerprintTransformPipeline()
    raw = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    _ = pipeline(raw)

    with open(sample_nist_image_path, "rb") as f:
        after_hash = hashlib.md5(f.read()).hexdigest()
    after_mtime = os.stat(sample_nist_image_path).st_mtime

    assert before_hash == after_hash, "Raw file MD5 hash changed after pipeline execution!"
    assert before_mtime == after_mtime, "Raw file mtime changed after pipeline execution!"


# ---------------------------------------------------------------------
# 10. Multi-Type Inputs & Input Robustness
# ---------------------------------------------------------------------

def test_pipeline_accepts_tensor_and_pil_inputs(sample_nist_image_path):
    raw = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    pipeline = FingerprintTransformPipeline(target_size=(256, 256), enhancement="hybrid")

    # 1. PIL Image input
    pil_img = Image.fromarray(raw)
    out_pil = pipeline(pil_img)
    assert isinstance(out_pil, torch.Tensor)
    assert out_pil.shape == (1, 256, 256)

    # 2. PyTorch Tensor input (1, H, W)
    t_img = torch.from_numpy(raw).unsqueeze(0).float()
    out_t = pipeline(t_img)
    assert isinstance(out_t, torch.Tensor)
    assert out_t.shape == (1, 256, 256)

def test_pipeline_accepts_float_numpy_inputs(sample_nist_image_path):
    raw = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    pipeline = FingerprintTransformPipeline(target_size=(256, 256), enhancement="hybrid")

    # Float [0.0, 1.0]
    f01 = raw.astype(np.float32) / 255.0
    out01 = pipeline(f01)
    assert isinstance(out01, torch.Tensor)
    assert out01.shape == (1, 256, 256)

    # Float [0.0, 255.0]
    f255 = raw.astype(np.float32)
    out255 = pipeline(f255)
    assert isinstance(out255, torch.Tensor)
    assert out255.shape == (1, 256, 256)

def test_pipeline_rejects_invalid_inputs():
    pipeline = FingerprintTransformPipeline()
    with pytest.raises(ValueError):
        pipeline(None)
    with pytest.raises(ValueError):
        pipeline(np.zeros((0, 0), dtype=np.uint8))


# ---------------------------------------------------------------------
# 11. Baseline Z-Score & MinMax Functions
# ---------------------------------------------------------------------

def test_enhancement_baseline_zscore_and_minmax_functions(sample_nist_image_path):
    raw = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)

    z = enhance_baseline_zscore(raw)
    assert isinstance(z, np.ndarray)
    assert z.shape == raw.shape
    assert z.dtype == np.uint8
    assert z.min() >= 0 and z.max() <= 255

    m = enhance_baseline_minmax(raw)
    assert isinstance(m, np.ndarray)
    assert m.shape == raw.shape
    assert m.dtype == np.uint8
    assert m.min() == 0 and m.max() == 255

    # Verify pipeline execution
    p_z = FingerprintTransformPipeline(enhancement="zscore")
    t_z = p_z(raw)
    assert t_z.shape == (1, 256, 256)


# ---------------------------------------------------------------------
# 12. DeviceDataLoader Attribute Delegation
# ---------------------------------------------------------------------

def test_device_dataloader_attribute_delegation():
    loader = create_fingerprint_dataloader(
        "manifests/nist_sd04_val.csv",
        batch_size=8,
        device="cpu",
        shuffle=False,
    )
    assert isinstance(loader, DeviceDataLoader)
    # Delegates DataLoader attributes transparently
    assert loader.batch_size == 8
    assert hasattr(loader, "dataset")
    assert len(loader.dataset) == 600
    assert hasattr(loader, "sampler")
    assert hasattr(loader, "num_workers")


# ---------------------------------------------------------------------
# 13. Default NIST SD04 Border Removal (No Fake White Bars)
# ---------------------------------------------------------------------

def test_default_pipeline_nist_clean_borders(sample_nist_image_path):
    raw = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    pipeline = FingerprintTransformPipeline()  # default pipeline
    out = pipeline(raw).squeeze(0).numpy()  # (256, 256)

    # Verify that the top and bottom rows are NOT solid 1.0 (artificial white borders)
    assert not np.allclose(out[0, :], 1.0), "Top row should not be artificial white padding!"
    assert not np.allclose(out[-1, :], 1.0), "Bottom row should not be artificial white padding!"
    # Verify standard deviation across borders reflects natural image content
    assert np.std(out[:8, :]) > 0.05
    assert np.std(out[-8:, :]) > 0.05


# ---------------------------------------------------------------------
# 14. Advanced Crop Modes & ImageNet Normalization
# ---------------------------------------------------------------------

def test_nist_square_crop_mode(sample_nist_image_path):
    raw = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    pipeline = FingerprintTransformPipeline(crop_mode="nist_square", aspect_mode="none", target_size=(256, 256))
    out = pipeline(raw)
    assert out.shape == (1, 256, 256)

def test_normalization_imagenet(sample_nist_image_path):
    raw = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    # 1 channel
    p1 = FingerprintTransformPipeline(channels=1, normalization="imagenet")
    t1 = p1(raw)
    assert t1.shape == (1, 256, 256)

    # 3 channels
    p3 = FingerprintTransformPipeline(channels=3, normalization="imagenet")
    t3 = p3(raw)
    assert t3.shape == (3, 256, 256)


# ---------------------------------------------------------------------
# 15. Custom Label Column Mapping
# ---------------------------------------------------------------------

def test_custom_label_column_socofing():
    # Use subject_id as label column
    dataset = FingerprintDataset(
        "manifests/socofing_eval.csv",
        label_column="subject_id",
        return_metadata=False,
    )
    img, target = dataset[0]
    assert isinstance(target, int)
    assert target >= 0  # Should be mapped to subject index or int ID, NOT -1


# ---------------------------------------------------------------------
# 16. Negative-One-to-One and Standardized Float Inputs
# ---------------------------------------------------------------------

def test_pipeline_accepts_negative_one_to_one_float_inputs(sample_nist_image_path):
    raw = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    # Image in [-1.0, 1.0] range
    f_neg1_1 = (raw.astype(np.float32) / 127.5) - 1.0
    pipeline = FingerprintTransformPipeline(target_size=(256, 256), enhancement="hybrid")
    out = pipeline(f_neg1_1)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 256, 256)
    # Must preserve contrast and not collapse into binary noise (<=2 unique values)
    assert len(torch.unique(out)) > 100

def test_pipeline_accepts_standardized_float_inputs(sample_nist_image_path):
    raw = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    f = raw.astype(np.float32)
    f_std = (f - f.mean()) / (f.std() + 1e-6)
    pipeline = FingerprintTransformPipeline(target_size=(256, 256), enhancement="hybrid")
    out = pipeline(f_std)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 256, 256)
    assert len(torch.unique(out)) > 100


# ---------------------------------------------------------------------
# 17. Integer OpenCV Interpolation Constants
# ---------------------------------------------------------------------

def test_pipeline_integer_interpolation_arguments(sample_nist_image_path):
    raw = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    for interp in [cv2.INTER_LANCZOS4, cv2.INTER_CUBIC, cv2.INTER_LINEAR]:
        pipeline = FingerprintTransformPipeline(interpolation=interp)
        out = pipeline(raw)
        assert out.shape == (1, 256, 256)


# ---------------------------------------------------------------------
# 18. PIL RGB and RGBA Input Modes
# ---------------------------------------------------------------------

def test_pipeline_pil_rgb_and_rgba_modes(sample_nist_image_path):
    raw = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    rgb = np.stack([raw, raw, raw], axis=-1)
    rgba = np.stack([raw, raw, raw, np.full_like(raw, 255)], axis=-1)

    pil_rgb = Image.fromarray(rgb, mode="RGB")
    pil_rgba = Image.fromarray(rgba, mode="RGBA")

    pipeline = FingerprintTransformPipeline(target_size=(256, 256))
    out_rgb = pipeline(pil_rgb)
    out_rgba = pipeline(pil_rgba)

    assert out_rgb.shape == (1, 256, 256)
    assert out_rgba.shape == (1, 256, 256)
    assert torch.allclose(out_rgb, out_rgba, atol=1e-3)


# ---------------------------------------------------------------------
# 19. In-Memory Caching and Preload in FingerprintDataset
# ---------------------------------------------------------------------

def test_dataset_in_memory_caching_and_preload():
    val_manifest = "manifests/nist_sd04_val.csv"
    dataset = FingerprintDataset(val_manifest, cache_transformed=True, return_metadata=True)
    assert dataset.cache_transformed is True
    assert len(dataset._cache) == 0

    # First access loads from disk and populates cache
    item0 = dataset[0]
    assert 0 in dataset._cache
    # Second access returns cached object directly
    item0_cached = dataset[0]
    assert item0[0] is item0_cached[0]

    # Clear cache
    dataset.clear_cache()
    assert len(dataset._cache) == 0


# ---------------------------------------------------------------------
# 20. DeviceDataLoader Multi-Element Tuple Preservation
# ---------------------------------------------------------------------

def test_device_dataloader_multi_element_tuple_preservation():
    # Simulate a dataloader that yields 4 elements: (img, target, meta, custom_extra)
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, idx):
            return torch.zeros(1, 10, 10), idx, {"meta": "val"}, torch.tensor([idx * 1.0])

    dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
    device_loader = DeviceDataLoader(dummy_loader, device="cpu")

    batch = next(iter(device_loader))
    assert len(batch) == 4, f"DeviceDataLoader dropped elements! Length: {len(batch)}"
    assert isinstance(batch[0], torch.Tensor)
    assert isinstance(batch[1], torch.Tensor)
    assert isinstance(batch[2], dict)
    assert isinstance(batch[3], torch.Tensor)


# ---------------------------------------------------------------------
# 21. Conservative Crop Mode
# ---------------------------------------------------------------------

def test_conservative_crop_mode(sample_nist_image_path):
    raw = cv2.imread(sample_nist_image_path, cv2.IMREAD_GRAYSCALE)
    pipeline = FingerprintTransformPipeline(crop_mode="conservative", target_size=(256, 256))
    out = pipeline(raw)
    assert out.shape == (1, 256, 256)
    assert not torch.isnan(out).any()


