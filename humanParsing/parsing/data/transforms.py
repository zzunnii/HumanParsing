import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Tuple, Optional, List

# 전역 상수 정의
ENABLE_AUGMENTATION = True  # 데이터 증강 활성화


class ParsingTransform:
    def __init__(
            self,
            input_size: Tuple[int, int] = (320, 640),
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
            is_train: bool = True
    ):
        if is_train and ENABLE_AUGMENTATION:
            self.transform = A.Compose([
                # 중앙 기준 확대/축소 (scale_limit: -0.3 ~ 0.3으로 설정, 70%~130% 스케일)
                A.RandomScale(scale_limit=(0.1, 3.0), interpolation=cv2.INTER_LINEAR, p=0.7),
                # 중앙에서 크롭하여 원래 크기 유지
                A.CenterCrop(height=input_size[0], width=input_size[1], p=1.0),

                # 기존 증강 유지
                A.OneOf([
                    A.RandomScale(scale_limit=(1.0, 1.5), p=1.0),
                    A.Affine(translate_percent=0.2, scale=(0.5, 1.5), rotate=(-45, 45), p=1.0),
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0)
                ], p=0.7),

                A.PadIfNeeded(
                    min_height=input_size[0],
                    min_width=input_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),

                A.OneOf([
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.ToGray(p=1.0),
                    A.ToSepia(p=1.0),
                    A.ImageCompression(quality_lower=75, quality_upper=75, p=1.0),
                ], p=0.3),

                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                    A.RandomRotate90(p=1.0),
                    A.Transpose(p=1.0),
                ], p=0.5),

                A.OneOf([
                    A.OpticalDistortion(p=1.0),
                    A.GridDistortion(p=1.0),
                    A.ElasticTransform(p=1.0),
                ], p=0.3),

                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=50, p=1.0),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                ], p=0.8),

                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ], p=0.3),

                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                ], p=0.2),

                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3), p=0.3),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=150, src_color=(255, 255, 255), p=0.1),

                # 최종 크기 조정 (필요 시)
                A.Resize(height=input_size[0], width=input_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        else:
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=max(input_size)),
                A.PadIfNeeded(
                    min_height=input_size[0],
                    min_width=input_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})

    def __call__(
            self,
            image: np.ndarray,
            mask: Optional[np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        if mask is not None:
            transformed = self.transform(image=image, mask=mask)
            return {
                'image': transformed['image'],
                'mask': transformed['mask'].long()
            }
        else:
            transformed = self.transform(image=image)
            return {'image': transformed['image']}


def build_transforms(
        input_size: Tuple[int, int] = (320, 640),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        is_train: bool = True
) -> ParsingTransform:
    return ParsingTransform(
        input_size=input_size,
        mean=mean,
        std=std,
        is_train=is_train
    )