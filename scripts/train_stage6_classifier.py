"""
Stage 6 Training Script: 5-Class NIST Pattern Classifier.
Trains ResNet-18 (or candidate backbones) on NIST SD04 dataset splits
using Apple Silicon GPU acceleration ('mps') and AdamW with Cosine Annealing.
"""

import os
import sys
import argparse
import time
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import FingerprintDataset
from src.data.transforms import FingerprintTransformPipeline
from src.features.transforms import OrientationAuxiliaryTransform
from src.models.backbones import (
    create_fingerprint_classifier,
    get_default_device,
    NIST_CLASSES,
    NIST_CLASS_TO_IDX,
)
from src.training.trainer import FingerprintTrainer
from src.training.losses import (
    CrossEntropyWithLabelSmoothing,
    AmbiguityAwareCrossEntropyLoss,
)


def plot_training_curves(history: dict, output_path: str, best_epoch: int = 0):
    """
    Generate publication-quality training and validation loss/accuracy curves.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    epochs = history["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Loss curves
    ax1.plot(epochs, history["train_loss"], "o-", color="#1f77b4", label="Train Loss", linewidth=2, markersize=4)
    ax1.plot(epochs, history["val_loss"], "s-", color="#ff7f0e", label="Val Loss", linewidth=2, markersize=4)
    if best_epoch > 0:
        ax1.axvline(best_epoch, color="#2ca02c", linestyle="--", alpha=0.8, label=f"Best Model (Ep {best_epoch})")
    ax1.set_title("Cross-Entropy Loss Dynamics", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Accuracy curves
    ax2.plot(epochs, [a * 100 for a in history["train_p_accuracy"]], "o-", color="#1f77b4", label="Train P-Accuracy", linewidth=2, markersize=4)
    ax2.plot(epochs, [a * 100 for a in history["val_p_accuracy"]], "s-", color="#ff7f0e", label="Val P-Accuracy (Strict)", linewidth=2, markersize=4)
    ax2.plot(epochs, [a * 100 for a in history["val_pc_accuracy"]], "^-", color="#2ca02c", label="Val PC-Accuracy (Permissive)", linewidth=2, markersize=4)
    if best_epoch > 0:
        ax2.axvline(best_epoch, color="#2ca02c", linestyle="--", alpha=0.8)
    ax2.set_title("Pattern Classification Accuracy Dynamics", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train NIST SD04 Fingerprint Classifier")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "efficientnet_b0", "custom_resnet18"])
    parser.add_argument("--channels", type=int, default=1, choices=[1, 3], help="1 (grayscale) or 3 (auxiliary orientation)")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Load ImageNet pretrained weights")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing factor")
    parser.add_argument("--use-ambiguity-loss", action="store_true", default=False, help="Use ambiguity-aware cross entropy")
    parser.add_argument("--output-dir", type=str, default="models", help="Model checkpoint directory")
    parser.add_argument("--visual-dir", type=str, default="investigation_output/stage6", help="Visual artifacts directory")
    args = parser.parse_args()

    device = get_default_device()
    print("=" * 75)
    print("STAGE 6: MAIN PATTERN CLASSIFICATION (5 NIST CLASSES)")
    print("=" * 75)
    print(f"Architecture: {args.backbone} | Channels: {args.channels} | Pretrained: {args.pretrained}")
    print(f"Hyperparameters: Batch Size={args.batch_size}, LR={args.lr}, Epochs={args.epochs}, Device={device}")
    print(f"Loss: {'AmbiguityAwareCrossEntropy' if args.use_ambiguity_loss else 'CrossEntropy'} (smoothing={args.label_smoothing})")

    # Build transforms
    if args.channels == 1:
        transform = FingerprintTransformPipeline(
            target_size=(256, 256),
            enhancement="hybrid",
            channels=1,
            normalization="zero_one",
        )
    else:
        transform = OrientationAuxiliaryTransform(
            target_size=(256, 256),
            enhancement="hybrid",
            mode="sincos",
            block_size=16,
        )

    # Build Datasets with preloading for fast training in RAM
    train_manifest = os.path.join(PROJECT_ROOT, "manifests", "nist_sd04_train.csv")
    val_manifest = os.path.join(PROJECT_ROOT, "manifests", "nist_sd04_val.csv")

    print("\nLoading and pre-caching datasets...")
    t0 = time.time()
    train_dataset = FingerprintDataset(
        manifest_path=train_manifest,
        root_dir=PROJECT_ROOT,
        transform=transform,
        label_column="primary_class",
        class_to_idx=NIST_CLASS_TO_IDX,
        return_metadata=True,
        cache_transformed=True,
    )
    train_dataset.preload()

    val_dataset = FingerprintDataset(
        manifest_path=val_manifest,
        root_dir=PROJECT_ROOT,
        transform=transform,
        label_column="primary_class",
        class_to_idx=NIST_CLASS_TO_IDX,
        return_metadata=True,
        cache_transformed=True,
    )
    val_dataset.preload()
    t1 = time.time()
    print(f"Dataset pre-caching complete in {t1 - t0:.2f}s ({len(train_dataset)} train, {len(val_dataset)} val).")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Instantiate model
    model = create_fingerprint_classifier(
        backbone_name=args.backbone,
        num_classes=5,
        in_channels=args.channels,
        pretrained=args.pretrained,
        dropout=0.2,
        device=device,
    )

    # Configure loss
    if args.use_ambiguity_loss:
        criterion = AmbiguityAwareCrossEntropyLoss(
            label_smoothing=args.label_smoothing,
            secondary_weight=0.3,
            num_classes=5,
        )
    else:
        criterion = CrossEntropyWithLabelSmoothing(
            label_smoothing=args.label_smoothing,
        )

    # Configure optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    # Trainer
    checkpoint_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    trainer = FingerprintTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        monitor_metric="val_p_accuracy",
        patience=args.patience,
        total_epochs=args.epochs,
    )

    history = trainer.fit(epochs=args.epochs, patience=args.patience, verbose=True)

    # Save training curves plot
    visual_dir = os.path.join(PROJECT_ROOT, args.visual_dir)
    os.makedirs(visual_dir, exist_ok=True)
    curve_path = os.path.join(visual_dir, "loss_accuracy_curves.png")
    plot_training_curves(history, curve_path, best_epoch=trainer.best_epoch)

    # Save training summary JSON
    summary_path = os.path.join(visual_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "backbone": args.backbone,
                "in_channels": args.channels,
                "pretrained": args.pretrained,
                "best_epoch": trainer.best_epoch,
                "best_val_p_accuracy": trainer.best_metric_val,
                "final_val_pc_accuracy": history["val_pc_accuracy"][trainer.best_epoch - 1] if trainer.best_epoch > 0 else 0.0,
                "epochs_trained": len(history["epoch"]),
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"Saved training summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()
