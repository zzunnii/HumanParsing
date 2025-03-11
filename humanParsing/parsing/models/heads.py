import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ConvBNReLU(nn.Module):
    """Convolution-BatchNormalization-ReLU module."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BasicHead(nn.Module):
    """Basic segmentation head with simple convolution layers."""
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNReLU(in_channels, in_channels // 2),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels // 2, num_classes, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MultiscaleHead(nn.Module):
    """Multiscale segmentation head with parallel convolution branches."""
    def __init__(self, in_channels: int, num_classes: int, scales: tuple = (1, 2, 4, 8),
                 dropout: float = 0.1):
        super().__init__()
        self.scales = scales
        channels = in_channels // len(scales)
        self.branches = nn.ModuleList([
            nn.Sequential(
                ConvBNReLU(in_channels, channels),
                nn.Dropout2d(dropout),
            ) for _ in scales
        ])
        self.final_conv = nn.Conv2d(channels * len(scales), num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for branch, scale in zip(self.branches, self.scales):
            if scale != 1:
                size = (x.shape[2] // scale, x.shape[3] // scale)
                scaled = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            else:
                scaled = x
            output = branch(scaled)
            if scale != 1:
                output = F.interpolate(output, size=x.shape[2:], mode='bilinear',
                                       align_corners=False)
            outputs.append(output)
        out = torch.cat(outputs, dim=1)
        out = self.final_conv(out)
        return out


def build_segmentation_head(name: str, in_channels: int, num_classes: int,
                            dropout: float = 0.1, scales: Optional[tuple] = None) -> nn.Module:
    """Build segmentation head according to name."""
    if name == 'basic':
        return BasicHead(in_channels, num_classes, dropout)
    elif name == 'multiscale':
        if scales is None:
            scales = (1, 2, 4, 8)
        return MultiscaleHead(in_channels, num_classes, scales, dropout)
    else:
        raise ValueError(f"Unknown head type: {name}")
