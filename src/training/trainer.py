"""
Training Engine for Fingerprint Pattern Classification.
Supports AdamW, Cosine Annealing, Early Stopping, Checkpoint Management,
and Device-aware execution ('mps' on Apple Silicon with CPU fallback).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.backbones import (
    NIST_CLASSES,
    NIST_CLASS_TO_IDX,
    IDX_TO_NIST_CLASS,
    get_default_device,
)
from src.training.losses import (
    CrossEntropyWithLabelSmoothing,
    AmbiguityAwareCrossEntropyLoss,
)
from src.training.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
)


class FingerprintTrainer:
    """
    Modular Trainer for Fingerprint Classifiers.
    Handles training, validation, early stopping, and checkpoint saving.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint_dir: str = "models",
        monitor_metric: str = "val_p_accuracy",  # 'val_p_accuracy' or 'val_loss'
        patience: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        total_epochs: int = 25,
        use_ambiguity_loss: bool = False,
    ):
        self.device = torch.device(device) if device is not None else get_default_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.monitor_metric = monitor_metric.lower()
        self.patience = patience
        self.total_epochs = total_epochs
        self.use_ambiguity_loss = use_ambiguity_loss

        # Default criterion
        if criterion is not None:
            self.criterion = criterion
        else:
            if self.use_ambiguity_loss:
                self.criterion = AmbiguityAwareCrossEntropyLoss(label_smoothing=0.05, secondary_weight=0.3)
            else:
                self.criterion = CrossEntropyWithLabelSmoothing(label_smoothing=0.05)

        # Default optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        # Default scheduler
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs,
                eta_min=1e-6,
            )

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_p_accuracy": [],
            "val_loss": [],
            "val_p_accuracy": [],
            "val_pc_accuracy": [],
            "lr": [],
        }

        # Early stopping state
        self.best_epoch = 0
        if self.monitor_metric in ("val_p_accuracy", "val_pc_accuracy"):
            self.best_metric_val = -float("inf")
            self.mode = "max"
        else:
            self.best_metric_val = float("inf")
            self.mode = "min"

    def _extract_batch(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract images, primary targets, and optional secondary/ambiguity targets.
        """
        if len(batch) == 3:
            imgs, targets, meta = batch
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            sec_targets = None
            is_amb = None
            if isinstance(meta, dict) and "secondary_class" in meta and "is_ambiguous" in meta:
                s_list = meta["secondary_class"]
                a_list = meta["is_ambiguous"]
                sec_indices = [NIST_CLASS_TO_IDX.get(str(s).strip(), -1) for s in s_list]
                amb_bools = [str(a).lower() == "true" for a in a_list]

                sec_targets = torch.tensor(sec_indices, dtype=torch.long, device=self.device)
                is_amb = torch.tensor(amb_bools, dtype=torch.bool, device=self.device)

            return imgs, targets, sec_targets, is_amb
        elif len(batch) == 2:
            imgs, targets = batch
            return imgs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True), None, None
        else:
            raise ValueError(f"Unexpected batch structure of length {len(batch)}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one complete training epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct_p = 0

        for batch in self.train_loader:
            imgs, targets, sec_targets, is_amb = self._extract_batch(batch)
            b_size = imgs.size(0)

            self.optimizer.zero_grad()
            logits = self.model(imgs)

            if isinstance(self.criterion, AmbiguityAwareCrossEntropyLoss) and sec_targets is not None:
                loss = self.criterion(logits, targets, sec_targets, is_amb)
            else:
                loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * b_size
            total_samples += b_size

            preds = torch.argmax(logits, dim=1)
            correct_p += (preds == targets).sum().item()

        avg_loss = total_loss / max(total_samples, 1)
        p_acc = correct_p / max(total_samples, 1)

        return {"train_loss": avg_loss, "train_p_accuracy": p_acc}

    def evaluate(
        self, dataloader: Optional[DataLoader] = None
    ) -> Tuple[Dict[str, float], ClassificationMetrics]:
        """
        Evaluate model on specified loader (or self.val_loader).
        Returns evaluation metrics dictionary and full ClassificationMetrics object.
        """
        loader = dataloader if dataloader is not None else self.val_loader
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        all_preds: List[int] = []
        all_true_p: List[int] = []
        all_true_s: List[int] = []
        all_is_amb: List[bool] = []

        with torch.no_grad():
            for batch in loader:
                imgs, targets, sec_targets, is_amb = self._extract_batch(batch)
                b_size = imgs.size(0)

                logits = self.model(imgs)

                if isinstance(self.criterion, AmbiguityAwareCrossEntropyLoss) and sec_targets is not None:
                    loss = self.criterion(logits, targets, sec_targets, is_amb)
                else:
                    loss = self.criterion(logits, targets)

                total_loss += loss.item() * b_size
                total_samples += b_size

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                t_p = targets.cpu().numpy()

                all_preds.extend(preds.tolist())
                all_true_p.extend(t_p.tolist())

                if sec_targets is not None:
                    all_true_s.extend(sec_targets.cpu().numpy().tolist())
                else:
                    all_true_s.extend([-1] * b_size)

                if is_amb is not None:
                    all_is_amb.extend(is_amb.cpu().numpy().tolist())
                else:
                    all_is_amb.extend([False] * b_size)

        avg_loss = total_loss / max(total_samples, 1)

        metrics = compute_classification_metrics(
            y_pred=all_preds,
            y_true_primary=all_true_p,
            y_true_secondary=all_true_s,
            is_ambiguous=all_is_amb,
            class_names=NIST_CLASSES,
        )

        eval_dict = {
            "val_loss": avg_loss,
            "val_p_accuracy": metrics.p_accuracy,
            "val_pc_accuracy": metrics.pc_accuracy,
            "macro_f1": metrics.macro_f1,
            "clean_p_accuracy": metrics.clean_p_accuracy,
            "ambiguous_pc_accuracy": metrics.ambiguous_pc_accuracy,
        }

        return eval_dict, metrics

    def fit(
        self,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Run the complete training pipeline with early stopping and checkpointing.
        """
        total_epochs = epochs or self.total_epochs
        patience = patience or self.patience
        no_improve_count = 0

        if verbose:
            print(f"Starting Training: {total_epochs} epochs | Device: {self.device}")
            print(f"Monitor: {self.monitor_metric} (mode={self.mode}, patience={patience})")
            print("-" * 75)

        start_time = time.time()

        for ep in range(1, total_epochs + 1):
            ep_start = time.time()
            train_res = self.train_epoch(ep)
            val_res, val_metrics = self.evaluate(self.val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Record history
            self.history["epoch"].append(ep)
            self.history["train_loss"].append(train_res["train_loss"])
            self.history["train_p_accuracy"].append(train_res["train_p_accuracy"])
            self.history["val_loss"].append(val_res["val_loss"])
            self.history["val_p_accuracy"].append(val_res["val_p_accuracy"])
            self.history["val_pc_accuracy"].append(val_res["val_pc_accuracy"])
            self.history["lr"].append(current_lr)

            # Check improvement
            monitored_val = val_res[self.monitor_metric]
            improved = False
            if self.mode == "max" and monitored_val > self.best_metric_val:
                improved = True
            elif self.mode == "min" and monitored_val < self.best_metric_val:
                improved = True

            ep_time = time.time() - ep_start

            if verbose:
                imp_mark = " [*BEST]" if improved else ""
                print(
                    f"Epoch {ep:02d}/{total_epochs:02d} [{ep_time:.1f}s] | "
                    f"Tr Loss: {train_res['train_loss']:.4f} Acc: {train_res['train_p_accuracy']*100:.1f}% | "
                    f"Val Loss: {val_res['val_loss']:.4f} P-Acc: {val_res['val_p_accuracy']*100:.1f}% "
                    f"PC-Acc: {val_res['val_pc_accuracy']*100:.1f}% | LR: {current_lr:.1e}{imp_mark}"
                )

            if improved:
                self.best_metric_val = monitored_val
                self.best_epoch = ep
                no_improve_count = 0
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, "best_model.pt"),
                    is_best=True,
                    extra_metadata={
                        "epoch": ep,
                        "best_metric": self.monitor_metric,
                        "best_metric_val": monitored_val,
                        "val_metrics": val_metrics.to_dict(),
                    },
                )
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    if verbose:
                        print(f"Early stopping triggered after {ep} epochs (no improvement for {patience} epochs).")
                    break

        # Save final model
        self.save_checkpoint(
            os.path.join(self.checkpoint_dir, "final_model.pt"),
            is_best=False,
            extra_metadata={
                "epoch": len(self.history["epoch"]),
                "final_train_loss": self.history["train_loss"][-1],
                "final_val_p_acc": self.history["val_p_accuracy"][-1],
            },
        )

        total_time = time.time() - start_time
        if verbose:
            print("-" * 75)
            print(f"Training Complete in {total_time:.1f}s. Best Epoch: {self.best_epoch} ({self.monitor_metric}={self.best_metric_val:.4f})")

        return self.history

    def save_checkpoint(
        self,
        filepath: str,
        is_best: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model weights, optimizer state, and training metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_epoch": self.best_epoch,
            "best_metric_val": self.best_metric_val,
            "history": self.history,
            "device": str(self.device),
            "is_best": is_best,
            "classes": NIST_CLASSES,
            "class_to_idx": NIST_CLASS_TO_IDX,
            "idx_to_class": IDX_TO_NIST_CLASS,
        }
        if hasattr(self.model, "backbone_name"):
            checkpoint["backbone_name"] = self.model.backbone_name
            checkpoint["in_channels"] = self.model.in_channels
            checkpoint["num_classes"] = self.model.num_classes

        if extra_metadata:
            checkpoint.update(extra_metadata)

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load weights from saved checkpoint into model."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
