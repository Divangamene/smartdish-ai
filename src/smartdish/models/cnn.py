"""Arquiteturas CNN."""
import timm
import torch
from torch import nn


class SimpleCNN(nn.Module):
    """CNN simples de 4 blocos conv -> ponto de partida para experimentar."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(name: str = "simple_cnn", num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes)
    # Qualquer modelo do timm: resnet18, efficientnet_b0, mobilenetv3_small_100, ...
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
