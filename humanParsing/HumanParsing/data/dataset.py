import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
from torchvision import transforms as T

# 문자열 → 정수 형태의 매핑 (model과 item 각각)
class Mapping:
    MODEL_CATEGORIES = {
        "hair": 1,
        "face": 2,
        "neck": 3,
        "hat": 4,
        "outer_rsleeve": 5,
        "outer_lsleeve": 6,
        "outer_torso": 7,
        "inner_rsleeve": 8,
        "inner_lsleeve": 9,
        "inner_torso": 10,
        "pants_hip": 11,
        "pants_rsleeve": 12,
        "pants_lsleeve": 13,
        "skirt": 14,
        "right_arm": 15,
        "left_arm": 16,
        "right_shoe": 17,
        "left_shoe": 18,
        "right_leg": 19,
        "left_leg": 20
    }

class PreprocessedDataset(Dataset):
    def __init__(self, data_dir, transforms=None, split='train', mode='item'):
        """
        Args:
            data_dir: 전처리된 person 디렉토리들이 모여있는 상위 경로
            transforms: Albumentations 등 원하는 변환
            split: 'train', 'val', 'test'
            mode: 'model' 또는 'item' (어느 데이터를 사용할지 지정)
        """
        self.data_dir = data_dir
        self.transforms = transforms
        self.split = split
        self.mode = mode

        # person 디렉토리 목록
        self.person_dirs = sorted([
            os.path.join(data_dir, d)
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

    def __len__(self):
        return len(self.person_dirs)

    def __getitem__(self, idx):
        person_path = self.person_dirs[idx]

        # 1) 원본 이미지 불러오기
        image_path = os.path.join(person_path, "model_image.jpg")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape

        # 2) mask_info.json 불러오기
        mask_info_path = os.path.join(person_path, "mask_info.json")
        if not os.path.exists(mask_info_path):
            combined_mask = np.zeros((H, W), dtype=np.uint8)
        else:
            with open(mask_info_path, 'r', encoding='utf-8') as f:
                mask_info = json.load(f)

            combined_mask = np.zeros((H, W), dtype=np.uint8)

            # 3) 여러 이진 마스크를 합쳐서 다중 클래스 마스크 만들기
            for mask_entry in mask_info.get("model_masks", []):
                category = mask_entry.get("category")
                if self.mode == "model":
                    class_id = Mapping.MODEL_CATEGORIES.get(category, 0)

                mask_path = mask_entry.get("path", None)
                if mask_path and os.path.exists(mask_path):
                    binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    # 크기 조정
                    if binary_mask.shape != (H, W):
                        binary_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                    combined_mask[binary_mask > 127] = class_id

        # 4) transforms 적용
        sample = {'image': image, 'mask': combined_mask}
        if self.transforms is not None:
            sample = self.transforms(**sample)

        # 5) Tensor 변환 (필요 시)
        if not torch.is_tensor(sample['image']):
            sample['image'] = T.ToTensor()(sample['image'])
        if not torch.is_tensor(sample['mask']):
            sample['mask'] = torch.tensor(sample['mask'], dtype=torch.long)

        return sample


def build_dataset(data_dir: str, transforms: Optional[object] = None, split: str = 'train',
                  mode='model') -> PreprocessedDataset:
    """
    Build preprocessed dataset.
    """
    return PreprocessedDataset(data_dir=data_dir, transforms=transforms, split=split, mode=mode)