import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .backbone import ResNetBackbone
from .decoder import UPerDecoder
# from .heads import build_segmentation_head  <-- 더 이상 사용하지 않음


class ParsingModel(nn.Module):
    """Unified Human Parsing Model with ResNet backbone and UPerNet decoder."""

    def __init__(
            self,
            num_classes: int,
            backbone_pretrained: bool = True,
            fpn_channels: int = 256,
            ppm_channels: int = 512,
            decoder_channels: int = 512,
            head_type: str = 'multiscale',  # 사용하지 않음
            dropout: float = 0.2,
            freeze_backbone: bool = False,
            freeze_bn: bool = False
    ):
        """
        Initialize ParsingModel.
        """
        super().__init__()

        # Initialize backbone (원본 피처들을 반환하도록 수정된 버전이어야 함)
        self.backbone = ResNetBackbone(
            pretrained=backbone_pretrained,
            fpn_channels=fpn_channels,
            freeze_backbone=freeze_backbone,
            freeze_bn=freeze_bn
        )

        # Get backbone output channels (사전 정의된 채널 정보를 사용)
        backbone_channels = self.backbone.channels  # 예: {'layer1':256, 'layer2':512, 'layer3':1024, 'layer4':2048}

        # Initialize decoder (UPerDecoder 내부에 최종 classifier 포함)
        self.decoder = UPerDecoder(
            in_channels=backbone_channels,
            ppm_channels=ppm_channels,
            fpn_channels=fpn_channels,
            decoder_channels=decoder_channels,
            dropout=dropout,
            num_classes=num_classes  # UPerDecoder가 최종 분류기를 생성함
        )

        # 별도의 segmentation head는 제거합니다.
        self.num_classes = num_classes

    def forward(
            self,
            x: torch.Tensor,
            size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        if size is None:
            size = x.shape[-2:]

        # Extract features from backbone
        features = self.backbone(x)
        # Decode features -> UPerDecoder returns segmentation logits of shape (B, num_classes, H, W)
        logits = self.decoder(features)

        # Resize logits to target size if needed
        if logits.shape[-2:] != size:
            logits = F.interpolate(
                logits, size=size, mode='bilinear', align_corners=False
            )
        return logits

    def get_predictions(
            self,
            x: torch.Tensor,
            size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Get class predictions from input.
        """
        logits = self.forward(x, size)
        predictions = torch.argmax(logits, dim=1)
        return predictions

    def init_weights(self, pretrained: Optional[str] = None):
        """
        Initialize model weights.
        """
        if pretrained is not None:
            state_dict = torch.load(pretrained)
            self.load_state_dict(state_dict, strict=False)

    def train(self, mode: bool = True):
        """
        Set training mode for the model.
        """
        super().train(mode)
        if getattr(self.backbone, '_freeze_bn', False):
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    m.eval()

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device
