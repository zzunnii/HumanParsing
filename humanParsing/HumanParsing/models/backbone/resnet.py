import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List


class ResNetBackbone(nn.Module):
    """ResNet101 backbone with FPN for semantic segmentation.

    Extracts multi-scale features from ResNet101 and builds a Feature Pyramid Network.
    """

    def __init__(
            self,
            pretrained: bool = True,
            fpn_channels: int = 256,
            freeze_bn: bool = False,
            freeze_backbone: bool = False
    ):
        super().__init__()

        # Load pretrained ResNet101
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

        # Extract ResNet stages
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

        # Define output channels for each stage
        self.channels = {
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048
        }

        # FPN lateral connections and smoothing convs
        self.lateral_convs = nn.ModuleDict({
            name: nn.Conv2d(channels, fpn_channels, kernel_size=1)
            for name, channels in self.channels.items()
        })

        self.smooth_convs = nn.ModuleDict({
            name: nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
            for name in self.channels.keys()
        })

        if freeze_backbone:
            self._freeze_backbone()

        if freeze_bn:
            self._freeze_bn()

        self.fpn_channels = fpn_channels

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def _freeze_bn(self):
        """Freeze all batch normalization layers."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ResNet + FPN.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
            Dict[str, torch.Tensor]: Dictionary of multi-scale feature maps
                Keys are 'layer1' through 'layer4'
                Values are tensors of shape (B, fpn_channels, H/s, W/s)
                where s is the stride (4, 8, 16, 32)
        """
        x = self.conv1(x)
        c1 = self.layer1(x)  # (B, 256, H/4, W/4)
        c2 = self.layer2(c1)  # (B, 512, H/8, W/8)
        c3 = self.layer3(c2)  # (B, 1024, H/16, W/16)
        c4 = self.layer4(c3)  # (B, 2048, H/32, W/32)

        # 원본 피처들을 그대로 반환
        features = {
            'layer1': c1,
            'layer2': c2,
            'layer3': c3,
            'layer4': c4,
        }
        return features
    def load_state_dict(self, state_dict: Dict, strict: bool = True):
        """Custom load function that can handle both full and partial state dicts."""
        if strict:
            return super().load_state_dict(state_dict, strict)
        else:
            # Filter out unexpected keys
            model_state = self.state_dict()
            matched_state = {
                k: v for k, v in state_dict.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            return super().load_state_dict(matched_state, strict=False)

    def train(self, mode: bool = True):
        """Override train mode to keep BN frozen if necessary."""
        super().train(mode)
        if getattr(self, '_freeze_bn', False):
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    m.eval()