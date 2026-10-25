"""The Step 5 baseline: ImageNet-pretrained ResNet-18 with a 4-class head."""
from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """ResNet-18 with its 1,000-class ImageNet head replaced by a num_classes head.

    Every layer stays trainable (fine-tuning); only the new head starts from random.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
