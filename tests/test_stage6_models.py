"""
Unit and Integration Tests for Stage 6: Fingerprint Pattern Classification.
Verifies model backbones, tensor output shapes, loss convergence on dummy batches,
MPS acceleration, inference engine, anatomical mapping, and metrics computation.
"""

import os
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.backbones import (
    NIST_CLASSES,
    NIST_CLASS_NAMES,
    ANATOMICAL_CLASS_NAMES,
    NIST_CLASS_TO_IDX,
    IDX_TO_NIST_CLASS,
    CustomResNet18,
    FingerprintClassifier,
    create_fingerprint_classifier,
    load_fingerprint_classifier,
    get_default_device,
)
from src.models.predict import (
    predict_fingerprint,
    predict_batch,
    map_to_anatomical,
    infer_hand,
    FingerprintPrediction,
)
from src.features.transforms import OrientationAuxiliaryTransform
from src.training.losses import (
    CrossEntropyWithLabelSmoothing,
    AmbiguityAwareCrossEntropyLoss,
)
from src.training.metrics import (
    compute_classification_metrics,
    ClassificationMetrics,
)
from src.training.trainer import FingerprintTrainer


class TestStage6ModelArchitectures:
    """Test suite for neural network backbones."""

    @pytest.mark.parametrize("backbone", ["custom_resnet18", "resnet18", "resnet34", "efficientnet_b0"])
    def test_single_channel_forward_shape(self, backbone: str):
        """Verify 1-channel 256x256 input generates (B, 5) output."""
        model = create_fingerprint_classifier(backbone_name=backbone, in_channels=1, pretrained=False)
        model.eval()
        dummy_x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            out = model(dummy_x)
            feat = model.forward_features(dummy_x)
            emb = model.get_embedding(dummy_x, normalize=True)

        assert out.shape == (2, 5), f"{backbone} expected shape (2, 5), got {out.shape}"
        assert feat.shape[0] == 2
        assert emb.shape[0] == 2
        # Check L2 normalization
        norms = torch.norm(emb, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    @pytest.mark.parametrize("backbone", ["custom_resnet18", "resnet18", "efficientnet_b0"])
    def test_multi_channel_auxiliary_forward(self, backbone: str):
        """Verify 3-channel input (auxiliary orientation/coherence) forward pass."""
        model = create_fingerprint_classifier(backbone_name=backbone, in_channels=3, pretrained=False)
        model.eval()
        dummy_x = torch.randn(3, 3, 256, 256)
        with torch.no_grad():
            out = model(dummy_x)
        assert out.shape == (3, 5)

    def test_custom_resnet18_internal_layers(self):
        """Verify native CustomResNet18 residual block architecture."""
        custom_net = CustomResNet18(in_channels=1, num_classes=5, dropout=0.1)
        x = torch.randn(1, 1, 256, 256)
        feat = custom_net.forward_features(x)
        assert feat.shape == (1, 512)
        logits = custom_net(x)
        assert logits.shape == (1, 5)


class TestStage6AuxiliaryTransforms:
    """Test suite for auxiliary orientation feature injection."""

    def test_orientation_auxiliary_sincos(self):
        """Verify 3-channel [Image, Sin(2theta), Cos(2theta)] generation."""
        transform = OrientationAuxiliaryTransform(
            target_size=(256, 256),
            enhancement="hybrid",
            mode="sincos",
            block_size=16,
        )
        dummy_img = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
        tensor = transform(dummy_img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 256, 256)
        assert tensor.dtype == torch.float32
        assert float(tensor.min()) >= 0.0
        assert float(tensor.max()) <= 1.0

    def test_orientation_auxiliary_coherence(self):
        """Verify 3-channel [Image, Sin(2theta), Coherence] mode."""
        transform = OrientationAuxiliaryTransform(
            target_size=(256, 256),
            mode="sin_coherence",
            block_size=16,
        )
        dummy_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        tensor = transform(dummy_img)
        assert tensor.shape == (3, 256, 256)


class TestStage6LossesAndOptimization:
    """Test suite for loss functions and dummy batch convergence."""

    def test_cross_entropy_label_smoothing(self):
        loss_fn = CrossEntropyWithLabelSmoothing(label_smoothing=0.1)
        logits = torch.randn(4, 5, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.0
        loss.backward()
        assert logits.grad is not None

    def test_ambiguity_aware_loss(self):
        loss_fn = AmbiguityAwareCrossEntropyLoss(label_smoothing=0.05, secondary_weight=0.3)
        logits = torch.randn(4, 5, requires_grad=True)
        p_targets = torch.tensor([0, 1, 2, 3])
        s_targets = torch.tensor([-1, 0, -1, 4])
        is_amb = torch.tensor([False, True, False, True])

        loss = loss_fn(logits, p_targets, s_targets, is_amb)
        assert loss.item() > 0.0
        loss.backward()
        assert logits.grad is not None

    def test_dummy_batch_convergence(self):
        """Verify model converges monotonically on an overfitted synthetic batch."""
        device = get_default_device()
        model = create_fingerprint_classifier("custom_resnet18", in_channels=1, pretrained=False, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(8, 1, 256, 256, device=device)
        y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], device=device)

        model.train()
        initial_loss = None
        final_loss = None

        for step in range(25):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert final_loss < initial_loss * 0.5, f"Expected loss drop, initial: {initial_loss}, final: {final_loss}"


class TestStage6MPSExecution:
    """Test device placement and Apple Silicon GPU acceleration."""

    def test_mps_or_cpu_execution(self):
        device = get_default_device()
        assert device.type in ("mps", "cpu")

        model = create_fingerprint_classifier("resnet18", in_channels=1, pretrained=False, device=device)
        x = torch.randn(2, 1, 256, 256, device=device)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.device.type == device.type
        assert out.shape == (2, 5)


class TestStage6InferenceAndAnatomicalMapping:
    """Test suite for inference utility and Henry anatomical translation."""

    def test_hand_inference_from_digits(self):
        assert infer_hand(digit_code="01") == "Right"
        assert infer_hand(digit_code="05") == "Right"
        assert infer_hand(digit_code="06") == "Left"
        assert infer_hand(digit_code="10") == "Left"
        assert infer_hand(finger_num=3) == "Right"
        assert infer_hand(finger_num=8) == "Left"

    def test_anatomical_mapping_rules(self):
        # Plain Arch, Tented Arch, Whorl are hand-invariant
        assert map_to_anatomical("A", hand="Right") == "Plain Arch"
        assert map_to_anatomical("A", hand="Left") == "Plain Arch"
        assert map_to_anatomical("T", hand="Right") == "Tented Arch"
        assert map_to_anatomical("W", hand="Right") == "Whorl"
        assert map_to_anatomical("W", hand="Left") == "Whorl"

        # Slant vs Anatomical Hand rules
        # Right hand: R -> Ulnar, L -> Radial
        assert map_to_anatomical("R", hand="Right") == "Ulnar Loop"
        assert map_to_anatomical("L", hand="Right") == "Radial Loop"

        # Left hand: L -> Ulnar, R -> Radial
        assert map_to_anatomical("L", hand="Left") == "Ulnar Loop"
        assert map_to_anatomical("R", hand="Left") == "Radial Loop"

        # Using digit codes
        assert map_to_anatomical("R", digit_code="02") == "Ulnar Loop"  # Right Index
        assert map_to_anatomical("R", digit_code="07") == "Radial Loop" # Left Index

    def test_predict_fingerprint_formatting(self):
        model = create_fingerprint_classifier("resnet18", in_channels=1, pretrained=False)
        dummy_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        pred = predict_fingerprint(
            model=model,
            image=dummy_img,
            hand="Right",
            digit_code="01",
        )

        assert isinstance(pred, FingerprintPrediction)
        assert pred.predicted_class in NIST_CLASSES
        assert 0.0 <= pred.confidence <= 1.0
        assert len(pred.probabilities) == 5
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-4
        assert pred.anatomical_label in ["Plain Arch", "Tented Arch", "Whorl", "Ulnar Loop", "Radial Loop"]
        assert pred.hand == "Right"
        assert pred.finger_num == 1
        assert pred.finger_name == "Thumb"
        assert len(pred.top_k) == 5

    def test_predict_fingerprint_with_finger_num_only(self):
        """Verify finger_num uniquely resolves finger_name without digit_code."""
        model = create_fingerprint_classifier("resnet18", in_channels=1, pretrained=False)
        dummy_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        pred = predict_fingerprint(model=model, image=dummy_img, hand="Right", finger_num=2)
        assert pred.finger_num == 2
        assert pred.finger_name == "Index"
        assert pred.hand == "Right"

        pred_left = predict_fingerprint(model=model, image=dummy_img, finger_num=7)
        assert pred_left.finger_num == 7
        assert pred_left.finger_name == "Index"
        assert pred_left.hand == "Left"

    def test_predict_fingerprint_3channel_auto_transform(self):
        """Verify 3-channel model automatically configures orientation transform without error."""
        model = create_fingerprint_classifier("resnet18", in_channels=3, pretrained=False)
        dummy_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        # Call without passing transform explicitly - should not raise shape mismatch error
        pred = predict_fingerprint(model=model, image=dummy_img)
        assert isinstance(pred, FingerprintPrediction)
        assert pred.predicted_class in NIST_CLASSES

    def test_predict_fingerprint_rejects_batched_tensor(self):
        """Verify 4D tensor with B > 1 raises clear ValueError directing to predict_batch."""
        model = create_fingerprint_classifier("resnet18", in_channels=1, pretrained=False)
        batch_x = torch.randn(4, 1, 256, 256)
        with pytest.raises(ValueError, match="predict_batch"):
            predict_fingerprint(model=model, image=batch_x)

    def test_predict_batch_tensor_and_list(self):
        """Verify predict_batch handles both batched tensors and lists of raw inputs."""
        model = create_fingerprint_classifier("resnet18", in_channels=1, pretrained=False)
        batch_x = torch.randn(3, 1, 256, 256)
        results = predict_batch(model=model, images=batch_x)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, FingerprintPrediction)
            assert r.predicted_class in NIST_CLASSES

        # List of numpy images
        img_list = [np.random.randint(0, 255, (256, 256), dtype=np.uint8) for _ in range(2)]
        meta_list = [{"hand": "Right", "digit_code": "01"}, {"hand": "Left", "digit_code": "06"}]
        list_results = predict_batch(model=model, images=img_list, metadata_list=meta_list)
        assert len(list_results) == 2
        assert list_results[0].hand == "Right"
        assert list_results[0].finger_name == "Thumb"
        assert list_results[1].hand == "Left"
        assert list_results[1].finger_name == "Thumb"


class TestStage6MetricsAndTrainer:
    """Test suite for metrics evaluation and trainer workflow."""

    def test_metrics_p_and_pc_accuracy(self):
        y_true_p = [0, 1, 2, 3, 4, 1]
        y_true_s = [-1, 0, -1, -1, -1, 0]
        y_pred = [0, 0, 2, 3, 4, 1]
        is_amb = [False, True, False, False, False, True]

        metrics = compute_classification_metrics(
            y_pred=y_pred,
            y_true_primary=y_true_p,
            y_true_secondary=y_true_s,
            is_ambiguous=is_amb,
        )

        assert metrics.p_accuracy == 5 / 6
        assert metrics.pc_accuracy == 1.0
        assert metrics.clean_count == 4
        assert metrics.ambiguous_count == 2
        assert metrics.clean_p_accuracy == 1.0
        assert metrics.ambiguous_pc_accuracy == 1.0
        assert len(metrics.confusion_matrix) == 5

    def test_metrics_length_validation(self):
        """Verify compute_classification_metrics raises ValueError on mismatched lengths."""
        with pytest.raises(ValueError, match="Mismatched lengths"):
            compute_classification_metrics(y_pred=[0, 1], y_true_primary=[0])

        with pytest.raises(ValueError, match="Mismatched lengths"):
            compute_classification_metrics(y_pred=[0, 1], y_true_primary=[0, 1], y_true_secondary=[0])

        with pytest.raises(ValueError, match="Mismatched lengths"):
            compute_classification_metrics(y_pred=[0, 1], y_true_primary=[0, 1], is_ambiguous=[True])

    def test_trainer_checkpoint_workflow(self):
        x_tr = torch.randn(16, 1, 256, 256)
        y_tr = torch.randint(0, 5, (16,))
        tr_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=8)

        x_val = torch.randn(8, 1, 256, 256)
        y_val = torch.randint(0, 5, (8,))
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=8)

        model = create_fingerprint_classifier("custom_resnet18", in_channels=1, pretrained=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = FingerprintTrainer(
                model=model,
                train_loader=tr_loader,
                val_loader=val_loader,
                checkpoint_dir=tmpdir,
                total_epochs=2,
                patience=2,
            )
            history = trainer.fit(epochs=2, verbose=False)
            ckpt_path = os.path.join(tmpdir, "final_model.pt")
            assert os.path.isfile(ckpt_path)
            assert len(history["epoch"]) == 2

            # Test checkpoint reload via trainer
            new_model = create_fingerprint_classifier("custom_resnet18", in_channels=1, pretrained=False)
            trainer.model = new_model
            ckpt = trainer.load_checkpoint(ckpt_path)
            assert "model_state_dict" in ckpt

            # Test checkpoint reload via load_fingerprint_classifier helper
            loaded_classifier = load_fingerprint_classifier(ckpt_path)
            assert isinstance(loaded_classifier, FingerprintClassifier)
            assert loaded_classifier.in_channels == 1
            assert loaded_classifier.num_classes == 5
