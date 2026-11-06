"""
Loss Functions for Fingerprint Pattern Classification.
Supports standard Cross-Entropy, Cross-Entropy with Label Smoothing,
and Ambiguity-Aware Cross-Entropy for dual-class / cross-referenced samples.
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyWithLabelSmoothing(nn.Module):
    """
    Standard Cross-Entropy Loss with optional label smoothing.
    Accepts hard class indices (B,) or soft target distributions (B, C).
    """

    def __init__(self, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1):
            targets = targets.view(-1).long()
            return F.cross_entropy(
                logits,
                targets,
                label_smoothing=self.label_smoothing,
                reduction=self.reduction,
            )
        else:
            # Soft distribution targets (B, C)
            log_probs = F.log_softmax(logits, dim=-1)
            num_classes = logits.shape[-1]
            if self.label_smoothing > 0.0:
                smooth_targets = (1.0 - self.label_smoothing) * targets + (self.label_smoothing / num_classes)
            else:
                smooth_targets = targets

            loss = -(smooth_targets * log_probs).sum(dim=-1)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss


class AmbiguityAwareCrossEntropyLoss(nn.Module):
    """
    Ambiguity-Aware Cross-Entropy Loss.
    
    In NIST SD04, approximately 17% of fingerprints carry dual-class annotations
    where fingerprint experts cross-referenced an ambiguous secondary pattern.
    
    For clean samples:
        Target distribution is one-hot on the primary class.
    For ambiguous samples:
        Target distribution divides mass:
        P(primary) = 1.0 - secondary_weight
        P(secondary) = secondary_weight
        
    Optional label smoothing further regularizes boundary representations.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        secondary_weight: float = 0.3,
        num_classes: int = 5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.secondary_weight = float(secondary_weight)
        self.num_classes = int(num_classes)
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        primary_targets: torch.Tensor,
        secondary_targets: Optional[torch.Tensor] = None,
        is_ambiguous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw model predictions.
            primary_targets: (B,) integer primary class indices, or (B, C) soft targets.
            secondary_targets: Optional (B,) integer secondary indices (-1 if none).
            is_ambiguous: Optional (B,) bool or uint8 tensor indicating ambiguity.
        """
        # If already soft probability distribution
        if primary_targets.ndim == 2 and primary_targets.shape[1] == self.num_classes:
            loss_fn = CrossEntropyWithLabelSmoothing(
                label_smoothing=self.label_smoothing,
                reduction=self.reduction,
            )
            return loss_fn(logits, primary_targets)

        b_size = logits.shape[0]
        device = logits.device
        primary_targets = primary_targets.view(-1).long()

        if secondary_targets is None or is_ambiguous is None:
            # Fall back to standard label-smoothed cross entropy
            return F.cross_entropy(
                logits,
                primary_targets,
                label_smoothing=self.label_smoothing,
                reduction=self.reduction,
            )

        secondary_targets = secondary_targets.view(-1).long()
        is_ambiguous = is_ambiguous.view(-1).bool()

        # Construct soft target distribution entirely with vectorized PyTorch operations
        target_dist = torch.zeros((b_size, self.num_classes), device=device, dtype=torch.float32)

        valid_p = (primary_targets >= 0) & (primary_targets < self.num_classes)
        if valid_p.any():
            p_indices = torch.clamp(primary_targets, 0, self.num_classes - 1)
            target_dist.scatter_(1, p_indices.unsqueeze(1), 1.0)
            if not valid_p.all():
                target_dist[~valid_p] = 1.0 / self.num_classes
        else:
            target_dist.fill_(1.0 / self.num_classes)

        amb_mask = (
            is_ambiguous
            & valid_p
            & (secondary_targets >= 0)
            & (secondary_targets < self.num_classes)
            & (secondary_targets != primary_targets)
        )
        if amb_mask.any():
            amb_idx = torch.nonzero(amb_mask).squeeze(1)
            p_sel = primary_targets[amb_idx]
            s_sel = secondary_targets[amb_idx]
            target_dist[amb_idx, p_sel] = 1.0 - self.secondary_weight
            target_dist[amb_idx, s_sel] = self.secondary_weight

        # Apply label smoothing if configured
        if self.label_smoothing > 0.0:
            target_dist = (1.0 - self.label_smoothing) * target_dist + (self.label_smoothing / self.num_classes)

        log_probs = F.log_softmax(logits, dim=-1)
        sample_loss = -(target_dist * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return sample_loss.mean()
        elif self.reduction == "sum":
            return sample_loss.sum()
        return sample_loss
