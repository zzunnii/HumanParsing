import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Tuple, Optional, List, Union
import random

# 전역 상수 정의
ENABLE_AUGMENTATION = True  # 데이터 증강 사용 여부를 제어하는 전역 변수


class BodyShapeTransform(A.ImageOnlyTransform):
    """인물 체형을 변경하는 커스텀 변환"""

    def __init__(self, width_scale_range=(0.85, 1.15), height_scale_range=(0.9, 1.1),
                 always_apply=False, p=0.5):
        super(BodyShapeTransform, self).__init__(always_apply, p)
        self.width_scale_range = width_scale_range
        self.height_scale_range = height_scale_range

    def apply(self, img, **params):
        height, width = img.shape[:2]

        # 가로 방향 스케일링 (체형 변화)
        width_scale = random.uniform(self.width_scale_range[0], self.width_scale_range[1])
        # 세로 방향 스케일링 (키 변화)
        height_scale = random.uniform(self.height_scale_range[0], self.height_scale_range[1])

        new_width = int(width * width_scale)
        new_height = int(height * height_scale)

        # 이미지 리사이징
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # 원본 크기로 패딩 또는 크롭
        result = np.zeros((height, width, 3), dtype=img.dtype)

        # 중앙 정렬
        h_offset = max(0, (height - new_height) // 2)
        w_offset = max(0, (width - new_width) // 2)

        # 원본보다 작은 경우
        if new_height <= height and new_width <= width:
            result[h_offset:h_offset + new_height, w_offset:w_offset + new_width] = img
        # 원본보다 큰 경우 (크롭)
        else:
            img_h_offset = max(0, (new_height - height) // 2)
            img_w_offset = max(0, (new_width - width) // 2)

            # 높이가 더 큰 경우
            if new_height > height:
                # 너비가 더 큰 경우
                if new_width > width:
                    result = img[img_h_offset:img_h_offset + height, img_w_offset:img_w_offset + width]
                # 너비는 작거나 같은 경우
                else:
                    result[:, w_offset:w_offset + new_width] = img[img_h_offset:img_h_offset + height, :]
            # 높이는 작거나 같지만 너비가 더 큰 경우
            else:
                result[h_offset:h_offset + new_height, :] = img[:, img_w_offset:img_w_offset + width]

        return result

    def get_transform_init_args_names(self):
        return ("width_scale_range", "height_scale_range")


class PersonExtraction(A.DualTransform):
    """마스크를 사용하여 인물을 추출하고 랜덤 배경으로 대체하는 변환"""

    def __init__(self, background_color_prob=0.7, background_noise_prob=0.3,
                 always_apply=False, p=0.5):
        super(PersonExtraction, self).__init__(always_apply, p)
        self.background_color_prob = background_color_prob
        self.background_noise_prob = background_noise_prob

        # 배경 대체용 색상 목록
        self.bg_colors = [
            (245, 245, 245),  # 흰색
            (240, 240, 240),  # 연한 회색
            (220, 220, 220),  # 회색
            (235, 245, 250),  # 연한 푸른색
            (245, 240, 230),  # 연한 베이지
            (235, 235, 245),  # 연한 보라색
            (230, 240, 235),  # 연한 녹색
        ]

    def apply(self, img, mask=None, **params):
        if mask is None:
            return img

        # 마스크에서 인물 영역 찾기 (마스크 값이 0이 아닌 영역)
        person_mask = mask > 0

        # 배경 영역 (마스크가 0인 곳)
        background_mask = ~person_mask

        # 결과 이미지 초기화
        result = img.copy()

        # 배경 처리 방식 선택
        r = random.random()

        if r < self.background_color_prob:
            # 배경을 단색으로 변경
            bg_color = random.choice(self.bg_colors)
            for c in range(3):  # RGB 각 채널에 대해
                result[:, :, c] = np.where(background_mask, bg_color[c], img[:, :, c])

        else:
            # 배경에 노이즈 추가
            noise = np.random.randint(220, 255, img.shape, dtype=np.uint8)
            # 노이즈에 텍스처 효과 추가
            noise = cv2.GaussianBlur(noise, (21, 21), 0)

            # 배경만 노이즈로 대체
            for c in range(3):
                result[:, :, c] = np.where(background_mask, noise[:, :, c], img[:, :, c])

        return result

    def apply_to_mask(self, mask, **params):
        # 마스크는 변경하지 않음
        return mask

    def get_transform_init_args_names(self):
        return ("background_color_prob", "background_noise_prob")


class ParsingTransform:
    def __init__(
            self,
            input_size: Tuple[int, int] = (320, 640),
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
            is_train: bool = True,
            enable_resume_mode: bool = False  # resume 모드 활성화 플래그 추가
    ):
        self.is_train = is_train
        self.enable_resume_mode = enable_resume_mode

        if is_train and ENABLE_AUGMENTATION:
            # 학습 모드이고 증강이 활성화된 경우

            # 기본 변환 (모든 경우 적용)
            basic_transforms = [
                A.LongestMaxSize(max_size=max(input_size)),
                A.PadIfNeeded(
                    min_height=input_size[0],
                    min_width=input_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ]

            # Resume 모드에서 추가되는 인물 특화 변환
            if enable_resume_mode:
                print("Resume 모드 활성화: 인물 특화 증강 적용")
                person_transforms = [
                    # 인물 체형 변형 (50% 확률)
                    BodyShapeTransform(
                        width_scale_range=(0.85, 1.15),  # 체형 변화
                        height_scale_range=(0.9, 1.1),  # 키 변화
                        p=0.5
                    ),

                    # 인물 추출 및 배경 대체 (70% 확률)
                    PersonExtraction(
                        background_color_prob=0.7,
                        background_noise_prob=0.3,
                        p=0.7
                    ),

                    # 인물 중심 이동, 회전, 크기 조정 (80% 확률)
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.2,
                        rotate_limit=15,
                        p=0.8
                    ),
                ]
            else:
                # 기존 변환 유지
                person_transforms = []

            # 기존 변환 (형태 변형) - 매개변수 수정
            shape_transforms = [
                A.OneOf([
                    A.Affine(
                        translate_percent=0.1,
                        scale=(0.8, 1.2),
                        rotate=(-15, 15),
                        p=1.0
                    ),
                    A.ElasticTransform(
                        alpha=80,
                        sigma=4,
                        p=1.0
                    )
                ], p=0.7 if not enable_resume_mode else 0.3),  # resume 모드에서는 확률 감소

                A.OneOf([
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=32,
                        max_width=32,
                        min_holes=1,
                        min_height=8,
                        min_width=8,
                        p=1.0
                    ),
                    A.GridDistortion(distort_limit=0.3, p=1.0),
                ], p=0.5 if not enable_resume_mode else 0.3),  # resume 모드에서는 확률 감소
            ]

            # 색상 변환 - 매개변수 수정
            color_transforms = [
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                ], p=0.8),

                A.OneOf([
                    A.GaussNoise(
                        var_limit=(10.0, 50.0),
                        mean=0,
                        p=1.0
                    ),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], p=0.3),
            ]

            # 최종 정규화 및 텐서 변환
            final_transforms = [
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ]

            # 모든 변환 결합
            all_transforms = basic_transforms + person_transforms + shape_transforms + color_transforms + final_transforms

            self.transform = A.Compose(
                all_transforms,
                additional_targets={'mask': 'mask'}
            )

        else:
            # 검증/테스트 모드에서는 크기 조정 후 정규화만 적용
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=max(input_size)),
                A.PadIfNeeded(
                    min_height=input_size[0],
                    min_width=input_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
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
        is_train: bool = True,
        enable_resume_mode: bool = False  # resume 모드 활성화 플래그 추가
) -> ParsingTransform:
    """
    Build transforms for the dataset.

    Args:
        input_size (tuple): Model input size (height, width)
        mean (tuple): Normalization mean values
        std (tuple): Normalization std values
        is_train (bool): Whether to use training transforms
        enable_resume_mode (bool): Whether to enable resume mode with advanced augmentations

    Returns:
        ParsingTransform: Configured transform object
    """
    return ParsingTransform(
        input_size=input_size,
        mean=mean,
        std=std,
        is_train=is_train,
        enable_resume_mode=enable_resume_mode
    )