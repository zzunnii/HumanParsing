import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Tuple, Optional, List

# 전역 상수 정의
ENABLE_AUGMENTATION = True  # 데이터 증강 사용 여부를 제어하는 전역 변수


class ParsingTransform:
    def __init__(
            self,
            input_size: Tuple[int, int] = (320, 640),
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
            is_train: bool = True
    ):
        if is_train:
            self.transform = A.Compose([
                # 랜덤 크롭: RandomResizedCrop와 CenterCrop 중 하나를 확률적으로 적용
                A.OneOf([
                    A.RandomResizedCrop(
                        size=input_size,
                        scale=(0.8, 1.0),
                        ratio=(0.9, 1.1),
                        interpolation=cv2.INTER_NEAREST,
                        p=1.0
                    ),
                    A.CenterCrop(
                        height=input_size[0],
                        width=input_size[1],
                        p=1.0
                    )
                ], p=0.8),

                # 기하학적 변형: Affine 또는 ElasticTransform 중 하나 선택 (30% 확률)
                A.OneOf([
                    A.Affine(
                        translate_percent=0.1,
                        scale=(0.9, 1.1),
                        rotate=(-20, 20),
                        interpolation=cv2.INTER_NEAREST,
                        p=1.0
                    ),
                    A.ElasticTransform(
                        alpha=60,
                        sigma=60 * 0.05,
                        alpha_affine=10,
                        interpolation=cv2.INTER_NEAREST,
                        p=1.0
                    )
                ], p=0.3),

                # 필요한 패딩 추가 (항상 적용)
                A.PadIfNeeded(
                    min_height=input_size[0],
                    min_width=input_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),

                # 색상 변화: 여러 색상 변환 중 1개만 랜덤 선택 (50% 확률)
                A.SomeOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=20,
                        p=1.0
                    ),
                    A.CLAHE(
                        clip_limit=2.0,
                        tile_grid_size=(8, 8),
                        p=1.0
                    ),
                ], n=1, p=0.5),

                # 수평 플립 (50% 확률)
                A.HorizontalFlip(p=0.5),

                # 노이즈 추가: GaussNoise 또는 MultiplicativeNoise 중 하나 선택 (20% 확률)
                A.OneOf([
                    A.GaussNoise(
                        var_limit=(5.0, 20.0),
                        p=1.0
                    ),
                    A.MultiplicativeNoise(
                        multiplier=(0.95, 1.05),
                        p=1.0
                    ),
                ], p=0.2),

                # 블러 적용 (10% 확률)
                A.GaussianBlur(
                    blur_limit=(3, 5),
                    p=0.1
                ),

                # 랜덤 그림자 추가 (20% 확률)
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    p=0.2
                ),

                # 최종 크기 조정 (항상 적용)
                A.Resize(
                    height=input_size[0],
                    width=input_size[1],
                    interpolation=cv2.INTER_NEAREST
                ),

                # 정규화 및 텐서 변환 (항상 적용)
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
        """
        Apply transforms to image and mask.

        Args:
            image (np.ndarray): Input image
            mask (np.ndarray, optional): Segmentation mask

        Returns:
            dict: Transformed image and mask
        """
        if mask is not None:
            # Convert to torch tensor and ensure dtype is long
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
    """
    Build transforms for the dataset.

    Args:
        input_size (tuple): Model input size (height, width)
        mean (tuple): Normalization mean values
        std (tuple): Normalization std values
        is_train (bool): Whether to use training transforms

    Returns:
        ParsingTransform: Configured transform object
    """
    return ParsingTransform(
        input_size=input_size,
        mean=mean,
        std=std,
        is_train=is_train
    )