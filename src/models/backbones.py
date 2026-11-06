"""
Fingerprint Pattern Classification Backbones.
Supports single-channel grayscale and multi-channel auxiliary inputs.
Candidate backbones: Custom ResNet-18, adapted ResNet-18/34, and EfficientNet-B0.
Fully compatible with Apple Silicon GPU ('mps') and CPU.
"""

from typing import Union, Optional, Tuple, Dict, Any
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

NIST_CLASS_NAMES = ["Plain Arch", "Tented Arch", "Whorl", "Left Loop", "Right Loop"]
ANATOMICAL_CLASS_NAMES = ["Plain Arch", "Tented Arch", "Whorl", "Ulnar Loop", "Radial Loop"]
NIST_CLASSES = ["A", "T", "W", "L", "R"]
NIST_CLASS_TO_IDX = {"A": 0, "T": 1, "W": 2, "L": 3, "R": 4}
IDX_TO_NIST_CLASS = {0: "A", 1: "T", 2: "W", 3: "L", 4: "R"}


def get_default_device() -> torch.device:
    """
    Returns the preferred acceleration device for Apple Silicon ('mps') with fallback to 'cpu'.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Custom ResNet-18 Implementation (Native PyTorch, no torchvision dependency)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CustomResNet18(nn.Module):
    """
    Custom 18-layer Residual Network optimized for 256x256 Fingerprint Classification.
    Accepts 1-channel (grayscale) or N-channel (auxiliary orientation/coherence) inputs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone_name = "custom_resnet18"
        self._in_planes = 64

        # Initial stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Stages: [2, 2, 2, 2] BasicBlocks
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # Global Pooling and Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block: type, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self._in_planes, planes, s))
            self._in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract post-pooling 512-dim feature embedding."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        feat = self.dropout(feat)
        return self.fc(feat)


# ---------------------------------------------------------------------------
# Adapted Torchvision Backbones (ResNet-18, ResNet-34, EfficientNet-B0)
# ---------------------------------------------------------------------------

def _adapt_first_conv(
    orig_conv: nn.Conv2d,
    in_channels: int,
    pretrained: bool,
) -> nn.Conv2d:
    """
    Adapt a 3-channel Conv2d layer to accept in_channels (e.g. 1 or 3 or 4) while preserving
    pretrained weight knowledge via channel-averaging.
    """
    if orig_conv.in_channels == in_channels:
        return orig_conv

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=orig_conv.out_channels,
        kernel_size=orig_conv.kernel_size,
        stride=orig_conv.stride,
        padding=orig_conv.padding,
        bias=(orig_conv.bias is not None),
    )

    if pretrained:
        with torch.no_grad():
            orig_weights = orig_conv.weight.data  # (out_ch, 3, kh, kw)
            if in_channels == 1:
                # Average across 3 RGB channels
                new_conv.weight.data = orig_weights.mean(dim=1, keepdim=True)
            elif in_channels > 3:
                # Copy RGB into first 3 channels, initialize remaining with channel mean
                new_conv.weight.data[:, :3] = orig_weights
                mean_ch = orig_weights.mean(dim=1, keepdim=True)
                for c in range(3, in_channels):
                    new_conv.weight.data[:, c:c+1] = mean_ch
            else:
                new_conv.weight.data[:, :in_channels] = orig_weights[:, :in_channels]

            if orig_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.data = orig_conv.bias.data

    return new_conv


class FingerprintClassifier(nn.Module):
    """
    Unified Fingerprint Classifier supporting multiple candidate backbones:
    - 'custom_resnet18': Native scratch implementation
    - 'resnet18': Torchvision ResNet-18 (pretrained or scratch) adapted for in_channels
    - 'resnet34': Torchvision ResNet-34 (pretrained or scratch) adapted for in_channels
    - 'efficientnet_b0': Torchvision EfficientNet-B0 adapted for in_channels
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 5,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.dropout_rate = dropout

        if self.backbone_name == "custom_resnet18":
            self.model = CustomResNet18(
                in_channels=in_channels,
                num_classes=num_classes,
                dropout=dropout,
            )
            self.feature_dim = 512

        elif self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            base.conv1 = _adapt_first_conv(base.conv1, in_channels, pretrained)
            in_feat = base.fc.in_features
            base.fc = nn.Sequential(
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_feat, num_classes),
            )
            self.model = base
            self.feature_dim = in_feat

        elif self.backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base = models.resnet34(weights=weights)
            base.conv1 = _adapt_first_conv(base.conv1, in_channels, pretrained)
            in_feat = base.fc.in_features
            base.fc = nn.Sequential(
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_feat, num_classes),
            )
            self.model = base
            self.feature_dim = in_feat

        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            base.features[0][0] = _adapt_first_conv(base.features[0][0], in_channels, pretrained)
            in_feat = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_feat, num_classes),
            )
            self.model = base
            self.feature_dim = in_feat

        else:
            raise ValueError(
                f"Unsupported backbone: '{backbone_name}'. "
                f"Supported: ['custom_resnet18', 'resnet18', 'resnet34', 'efficientnet_b0']"
            )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global feature vector before classifier head."""
        if self.backbone_name == "custom_resnet18":
            return self.model.forward_features(x)
        elif self.backbone_name in ("resnet18", "resnet34"):
            m = self.model
            x = m.conv1(x)
            x = m.bn1(x)
            x = m.relu(x)
            x = m.maxpool(x)
            x = m.layer1(x)
            x = m.layer2(x)
            x = m.layer3(x)
            x = m.layer4(x)
            x = m.avgpool(x)
            return torch.flatten(x, 1)
        elif self.backbone_name == "efficientnet_b0":
            m = self.model
            x = m.features(x)
            x = m.avgpool(x)
            return torch.flatten(x, 1)
        else:
            raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits (B, num_classes)."""
        return self.model(x)

    def get_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Extract feature embeddings (useful for metric learning and verification).
        """
        features = self.forward_features(x)
        if normalize:
            features = F.normalize(features, p=2, dim=1, eps=1e-20)
        return features


def create_fingerprint_classifier(
    backbone_name: str = "resnet18",
    num_classes: int = 5,
    in_channels: int = 1,
    pretrained: bool = True,
    dropout: float = 0.2,
    device: Optional[Union[str, torch.device]] = None,
) -> FingerprintClassifier:
    """
    Factory function for building and device-placing fingerprint classifiers.
    """
    model = FingerprintClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        dropout=dropout,
    )
    if device is not None:
        model = model.to(device)
    return model


def load_fingerprint_classifier(
    checkpoint_path: str,
    device: Optional[Union[str, torch.device]] = None,
) -> FingerprintClassifier:
    """
    Load a trained FingerprintClassifier from a saved checkpoint file.
    Automatically reconstructs the model architecture matching checkpoint metadata.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    target_device = torch.device(device) if device is not None else get_default_device()
    checkpoint = torch.load(checkpoint_path, map_location=target_device)

    backbone_name = checkpoint.get("backbone_name", "resnet18")
    in_channels = checkpoint.get("in_channels", 1)
    num_classes = checkpoint.get("num_classes", 5)

    model = create_fingerprint_classifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=False,
        device=target_device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
