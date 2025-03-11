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
        "hair": 0,
        "face": 1,
        "neck": 2,
        "hat": 3,
        "outer_rsleeve": 4,
        "outer_lsleeve": 5,
        "outer_torso": 6,
        "inner_rsleeve": 7,
        "inner_lsleeve": 8,
        "inner_torso": 9,
        "pants_hip": 10,
        "pants_rsleeve": 11,
        "pants_lsleeve": 12,
        "skirt": 13,
        "right_arm": 14,
        "left_arm": 15,
        "right_shoe": 16,
        "left_shoe": 17,
        "right_leg": 18,
        "left_leg": 19
    }

    TOP_CATEGORIES = {
        # 모자 제외
        "rsleeve": 1,  # 오른쪽 소매
        "lsleeve": 2,  # 왼쪽 소매
        "torso": 3,  # 몸통 (보이는 부분)
        "top_hidden": 4  # 몸통 (히든/가려진 부분)
    }

    BOTTOM_CATEGORIES = {
        "hip": 1,  # 엉덩이
        "pants_rsleeve": 2,  # 바지 오른쪽
        "pants_lsleeve": 3,  # 바지 왼쪽
        "pants_hidden": 4,  # 바지(히든)
        "skirt": 5,  # 치마 (보이는 부분)
        "skirt_hidden": 6  # 치마 (히든/가려진 부분)
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
        image_path = os.path.join(person_path, "item_image.jpg")
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

            # 3) 모드에 따라 다른 마스크 처리 로직 적용
            if self.mode == "model":
                # 모델 모드: model_masks 사용
                for mask_entry in mask_info.get("model_masks", []):
                    category = mask_entry.get("category")
                    class_id = Mapping.MODEL_CATEGORIES.get(category, 0)

                    mask_path = mask_entry.get("path", None)
                    if mask_path and os.path.exists(mask_path):
                        binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if binary_mask.shape != (H, W):
                            binary_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                        combined_mask[binary_mask > 127] = class_id

            elif self.mode == "tops":
                # 상의 모드: 상의 관련 마스크만 처리
                top_product_types = ["rsleeve", "lsleeve", "torso", "top_hidden"]
                for mask_entry in mask_info.get("item_masks", []):
                    product_type = mask_entry.get("product_type")

                    # 상의 관련 제품인 경우만 처리
                    if product_type in top_product_types:
                        class_id = Mapping.TOP_CATEGORIES.get(product_type, 0)

                        mask_path = mask_entry.get("path", None)
                        if mask_path and os.path.exists(mask_path):
                            binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            if binary_mask.shape != (H, W):
                                binary_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                            combined_mask[binary_mask > 127] = class_id

            elif self.mode == "bottoms":
                # 하의 모드: 하의 관련 마스크만 처리
                bottom_product_types = ["hip", "pants_rsleeve", "pants_lsleeve", "pants_hidden", "skirt",
                                        "skirt_hidden"]
                for mask_entry in mask_info.get("item_masks", []):
                    product_type = mask_entry.get("product_type")

                    # 하의 관련 제품인 경우만 처리
                    if product_type in bottom_product_types:
                        class_id = Mapping.BOTTOM_CATEGORIES.get(product_type, 0)

                        mask_path = mask_entry.get("path", None)
                        if mask_path and os.path.exists(mask_path):
                            binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
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


def build_dataset(data_dir, transforms=None, split='train', mode='model', filter_empty=True):
    dataset = PreprocessedDataset(data_dir, transforms, split, mode)

    if filter_empty:
        # 배경만 있는 이미지 필터링
        filtered_indices = []
        for i in range(len(dataset)):
            sample = dataset[i]
            mask = sample['mask']
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            unique_classes = np.unique(mask)

            # 배경 외 클래스가 있는 이미지만 선택
            if len(unique_classes) > 1 or (len(unique_classes) == 1 and unique_classes[0] != 0):
                filtered_indices.append(i)

        from torch.utils.data import Subset
        return Subset(dataset, filtered_indices)

    return dataset


