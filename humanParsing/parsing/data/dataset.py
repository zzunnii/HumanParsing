import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import Dict, Optional
from torchvision import transforms as T

class Mapping:
    MODEL_CATEGORIES = {
        "hair": 1, "face": 2, "neck": 3, "hat": 4,
        "outer_rsleeve": 5, "outer_lsleeve": 6, "outer_torso": 7,
        "inner_rsleeve": 8, "inner_lsleeve": 9, "inner_torso": 10,
        "pants_hip": 11, "pants_rsleeve": 12, "pants_lsleeve": 13,
        "skirt": 14, "right_arm": 15, "left_arm": 16,
        "right_shoe": 17, "left_shoe": 18, "right_leg": 19, "left_leg": 20
    }
    ITEM_CATEGORIES = {
        "hat": 0, "hat_hidden": 1, "rsleeve": 2, "lsleeve": 3, "torso": 4,
        "top_hidden": 5, "hip": 6, "pants_rsleeve": 7, "pants_lsleeve": 8,
        "pants_hidden": 9, "skirt": 10, "skirt_hidden": 11, "shoe": 12, "shoe_hidden": 13
    }
    TOP_CATEGORIES = {"rsleeve": 1, "lsleeve": 2, "torso": 3, "top_hidden": 4}
    BOTTOM_CATEGORIES = {"hip": 1, "pants_rsleeve": 2, "pants_lsleeve": 3, "pants_hidden": 4, "skirt": 5, "skirt_hidden": 6}

class PreprocessedDataset(Dataset):
    def __init__(self, data_dir, transforms=None, split='train', mode='item', quick_test=False, subset_size=None):
        """
        Args:
            data_dir: 전처리된 person 디렉토리들이 모여있는 상위 경로
            transforms: 변환 함수
            split: 'train', 'val', 'test'
            mode: 'model', 'item', 'tops', 'bottoms'
            quick_test: 퀵테스트 모드 여부
            subset_size: 퀵테스트 시 사용할 데이터 크기 (기본값 None)
        """
        self.data_dir = data_dir
        self.transforms = transforms
        self.split = split
        self.mode = mode
        self.quick_test = quick_test
        self.subset_size = subset_size

        # person 디렉토리 목록
        all_person_dirs = sorted([
            os.path.join(data_dir, d)
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        # quick_test 모드라면 일부만 선택
        if self.quick_test and self.subset_size is not None:
            # 무작위로 섞은 후 subset_size만큼 선택
            np.random.shuffle(all_person_dirs)
            self.person_dirs = all_person_dirs[:self.subset_size]
        else:
            self.person_dirs = all_person_dirs

    def __len__(self):
        return len(self.person_dirs)

    def __getitem__(self, idx):
        person_path = self.person_dirs[idx]
        image_path = os.path.join(person_path, "model_image.jpg")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape

        mask_info_path = os.path.join(person_path, "mask_info.json")
        if not os.path.exists(mask_info_path):
            combined_mask = np.zeros((H, W), dtype=np.uint8)
        else:
            with open(mask_info_path, 'r', encoding='utf-8') as f:
                mask_info = json.load(f)
            combined_mask = np.zeros((H, W), dtype=np.uint8)

            if self.mode == "model":
                for mask_entry in mask_info.get("model_masks", []):
                    category = mask_entry.get("category")
                    class_id = Mapping.MODEL_CATEGORIES.get(category, 0)
                    mask_path = mask_entry.get("path")
                    if mask_path and os.path.exists(mask_path):
                        binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if binary_mask.shape != (H, W):
                            binary_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                        combined_mask[binary_mask > 127] = class_id
            elif self.mode == "item":
                for mask_entry in mask_info.get("item_masks", []):
                    product_type = mask_entry.get("product_type")
                    class_id = Mapping.ITEM_CATEGORIES.get(product_type, 0)
                    mask_path = mask_entry.get("path")
                    if mask_path and os.path.exists(mask_path):
                        binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if binary_mask.shape != (H, W):
                            binary_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                        combined_mask[binary_mask > 127] = class_id
            elif self.mode == "tops":
                top_product_types = ["rsleeve", "lsleeve", "torso", "top_hidden"]
                for mask_entry in mask_info.get("item_masks", []):
                    product_type = mask_entry.get("product_type")
                    if product_type in top_product_types:
                        class_id = Mapping.TOP_CATEGORIES.get(product_type, 0)
                        mask_path = mask_entry.get("path")
                        if mask_path and os.path.exists(mask_path):
                            binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            if binary_mask.shape != (H, W):
                                binary_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                            combined_mask[binary_mask > 127] = class_id
            elif self.mode == "bottoms":
                bottom_product_types = ["hip", "pants_rsleeve", "pants_lsleeve", "pants_hidden", "skirt", "skirt_hidden"]
                for mask_entry in mask_info.get("item_masks", []):
                    product_type = mask_entry.get("product_type")
                    if product_type in bottom_product_types:
                        class_id = Mapping.BOTTOM_CATEGORIES.get(product_type, 0)
                        mask_path = mask_entry.get("path")
                        if mask_path and os.path.exists(mask_path):
                            binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            if binary_mask.shape != (H, W):
                                binary_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                            combined_mask[binary_mask > 127] = class_id

        sample = {'image': image, 'mask': combined_mask}
        if self.transforms is not None:
            sample = self.transforms(**sample)

        if not torch.is_tensor(sample['image']):
            sample['image'] = T.ToTensor()(sample['image'])
        if not torch.is_tensor(sample['mask']):
            sample['mask'] = torch.tensor(sample['mask'], dtype=torch.long)

        return sample

def build_dataset(data_dir, transforms=None, split='train', mode='model', quick_test=False, subset_size=None, filter_empty=True):
    dataset = PreprocessedDataset(data_dir, transforms, split, mode, quick_test, subset_size)

    if filter_empty:
        filtered_indices = []
        for i in range(len(dataset)):
            sample = dataset[i]
            mask = sample['mask']
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            unique_classes = np.unique(mask)
            if len(unique_classes) > 1 or (len(unique_classes) == 1 and unique_classes[0] != 0):
                filtered_indices.append(i)
        return Subset(dataset, filtered_indices)

    return dataset

