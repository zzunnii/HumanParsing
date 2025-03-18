import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class PPM(nn.Module):
    def __init__(
            self,
            in_channels: int,  # 2048로 설정되어야 함
            out_channels: int,
            pool_sizes: Tuple[int, ...] = (1, 2, 3, 6)
    ):
        super().__init__()

        self.pool_sizes = pool_sizes
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PPM.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W)

        Returns:
            torch.Tensor: Concatenated feature maps from all pyramid levels
                Shape: (B, C * (len(pool_sizes) + 1), H, W)
        """
        size = x.size()[2:]
        outputs = [x]

        for stage in self.stages:
            feat = stage(x)
            upsampled = F.interpolate(
                feat, size=size, mode='bilinear', align_corners=False
            )
            outputs.append(upsampled)

        return torch.cat(outputs, dim=1)


class UPerDecoder(nn.Module):
    def __init__(
            self,
            in_channels: Dict[str, int],
            ppm_channels: int = 512,
            ppm_pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),  # 여기에 정의
            fpn_channels: int = 256,
            decoder_channels: int = 256,
            dropout: float = 0.1,
            num_classes: int = 20
    ):
        super().__init__()

        # PPM Module for highest level features
        self.ppm = PPM(
            in_channels['layer4'],  # 2048이 들어가야 함
            ppm_channels // len(ppm_pool_sizes),
            ppm_pool_sizes
        )

        # FPN fusion conv
        fpn_in_channels = in_channels['layer4'] + ppm_channels
        self.fpn_in = nn.Sequential(
            nn.Conv2d(fpn_in_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )

        # Lateral and output convs for each FPN level
        self.lateral_convs = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(channels, fpn_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True)
            )
            for name, channels in in_channels.items()
            if name != 'layer4'  # layer4 is handled by PPM
        })

        self.fpn_convs = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(fpn_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True)
            )
            for name in in_channels.keys()
        })

        # Final classifier
        self.dropout = nn.Dropout2d(p=dropout)
        self.classifier = nn.Sequential(
            nn.Conv2d(len(in_channels) * decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize the weights of convolution layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of UPerNet decoder.

        Args:
            features (Dict[str, torch.Tensor]): Dictionary of feature maps from backbone
                Keys are 'layer1' through 'layer4'

        Returns:
            torch.Tensor: Segmentation logits of shape (B, num_classes, H, W)
        """
        # PPM on highest level features
        ppm_out = self.ppm(features['layer4'])
        f = self.fpn_in(ppm_out)

        # Build FPN features
        fpn_features = {'layer4': self.fpn_convs['layer4'](f)}

        for name, conv in self.lateral_convs.items():
            # Get input feature
            feat = features[name]

            # Get target size
            if name == 'layer1':
                size = feat.shape[-2:]

            # Lateral connection
            lat = conv(feat)

            # Add top-down connection
            f = F.interpolate(f, size=lat.shape[-2:], mode='bilinear', align_corners=False)
            f = lat + f

            # Save processed feature
            fpn_features[name] = self.fpn_convs[name](f)

        # Concatenate all FPN levels
        out = []
        for name, feat in fpn_features.items():
            if name == 'layer1':
                size = feat.shape[-2:]
            else:
                feat = F.interpolate(feat, size=size, mode='bilinear', align_corners=False)
            out.append(feat)

        out = torch.cat(out, dim=1)
        out = self.dropout(out)
        out = self.classifier(out)

        return out