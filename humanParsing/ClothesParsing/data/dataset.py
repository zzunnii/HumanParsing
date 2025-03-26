import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, List
from torchvision import transforms as T
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class Mapping:

    TOP_CATEGORIES = {
        "background": 0,
        "rsleeve": 1,
        "lsleeve": 2,
        "torso": 3,
        "top_hidden": 4
    }

    BOTTOM_CATEGORIES = {
        "background": 0,
        "hip": 1,
        "pants_rsleeve": 2,
        "pants_lsleeve": 3,
        "pants_hidden": 4,
        "skirt": 5,
        "skirt_hidden": 6
    }


class PreprocessedDataset(Dataset):
    def __init__(self, data_dir, transforms=None, split='train', mode='tops'):
        """
        Args:
            data_dir: 전처리된 person 디렉토리들이 모여있는 상위 경로
            transforms: 변환 함수
            split: 'train', 'val', 'test'
            mode: , 'tops', 'bottoms' 중 하나
        """
        self.data_dir = data_dir
        self.transforms = transforms
        self.split = split
        self.mode = mode
        self.load_times = {'dir_scan': 0, 'image_load': 0, 'mask_load': 0, 'transform': 0}

        # 디렉토리 스캔 시간 측정
        print(f"Scanning directory {data_dir}...")
        start_time = time.time()
        all_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.person_dirs = sorted([os.path.join(data_dir, d) for d in all_dirs])
        scan_time = time.time() - start_time
        self.load_times['dir_scan'] = scan_time
        print(f"Found {len(self.person_dirs)} directories in {scan_time:.2f} seconds")

        # 모드에 따른 카테고리 매핑 선택
        if mode == 'tops':
            self.category_mapping = Mapping.TOP_CATEGORIES
        elif mode == 'bottoms':
            self.category_mapping = Mapping.BOTTOM_CATEGORIES

    def _load_item(self, idx):
        """실제 아이템 로딩 구현"""
        person_path = self.person_dirs[idx]

        # 1) 이미지 로딩 시간 측정
        img_start = time.time()
        image_path = os.path.join(person_path, "item_image.jpg")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_time = time.time() - img_start
        self.load_times['image_load'] += img_time

        H, W, _ = image.shape

        # 2) 마스크 로딩 시간 측정
        mask_start = time.time()
        combined_mask = np.zeros((H, W), dtype=np.uint8)  # 배경 = 0
        mask_info_path = os.path.join(person_path, "mask_info.json")

        if os.path.exists(mask_info_path):
            with open(mask_info_path, 'r', encoding='utf-8') as f:
                mask_info = json.load(f)

            mask_entries = mask_info.get("item_masks", [])

            # 마스크 결합
            for mask_entry in mask_entries:
                category = mask_entry.get("product_type")

                # 현재 모드에 해당하는 카테고리만 처리
                class_id = self.category_mapping.get(category)
                if class_id is None:
                    continue  # 해당 없는 카테고리 스킵

                mask_path = mask_entry.get("path")
                if mask_path and os.path.exists(mask_path):
                    binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if binary_mask.shape != (H, W):
                        binary_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                    combined_mask[binary_mask > 127] = class_id

        mask_time = time.time() - mask_start
        self.load_times['mask_load'] += mask_time

        return {'image': image, 'mask': combined_mask}

    def __len__(self):
        return len(self.person_dirs)

    def __getitem__(self, idx):
        # 항상 데이터 로드 (캐시 사용 없음)
        sample = self._load_item(idx)

        # 변환 시간 측정
        transform_start = time.time()

        # transforms 적용
        if self.transforms is not None:
            sample = self.transforms(**sample)

        # Tensor 변환
        if not torch.is_tensor(sample['image']):
            sample['image'] = T.ToTensor()(sample['image'])
        if not torch.is_tensor(sample['mask']):
            sample['mask'] = torch.tensor(sample['mask'], dtype=torch.long)

        transform_time = time.time() - transform_start
        self.load_times['transform'] += transform_time

        return sample

    def print_timing_stats(self):
        """로딩 시간 통계 출력"""
        total = sum(self.load_times.values())
        print("\n--- Data Loading Timing Statistics ---")
        print(
            f"Directory scanning: {self.load_times['dir_scan']:.2f}s ({100 * self.load_times['dir_scan'] / total:.1f}%)")
        print(
            f"Image loading: {self.load_times['image_load']:.2f}s ({100 * self.load_times['image_load'] / total:.1f}%)")
        print(
            f"Mask processing: {self.load_times['mask_load']:.2f}s ({100 * self.load_times['mask_load'] / total:.1f}%)")
        print(f"Transforms: {self.load_times['transform']:.2f}s ({100 * self.load_times['transform'] / total:.1f}%)")
        print(f"Total timing: {total:.2f}s")
        print("---------------------------------------")


def build_dataset(data_dir, transforms=None, split='train', mode='tops', quick_test=False, subset_size=None,
                  filter_empty=False):
    """
    Build preprocessed dataset with detailed progress tracking.
    """
    print(f"Building {mode} dataset from {data_dir}...")
    start_time = time.time()

    # 원본 데이터셋 생성
    dataset = PreprocessedDataset(data_dir=data_dir, transforms=transforms, split=split, mode=mode)
    print(f"Base dataset created in {time.time() - start_time:.2f} seconds")

    # 빠른 테스트 모드 처리
    if quick_test and subset_size:
        from torch.utils.data import Subset
        print(f"Creating quick test subset with {subset_size} samples...")
        indices = np.random.choice(len(dataset), min(subset_size, len(dataset)), replace=False)
        dataset = Subset(dataset, indices)
        print(f"Subset created in {time.time() - start_time:.2f} seconds")
        return dataset

    # tops/bottoms 모드에서 필터링 필요 시
    from torch.utils.data import Subset
    import json

    print(f"Filtering required for {mode} mode...")
    filter_start = time.time()

    # 필터링 수행
    print(f"Filtering {len(dataset)} samples...")
    filtered_indices = []

    for i in tqdm(range(len(dataset)), desc=f"Filtering {mode} dataset"):
        try:
            sample = dataset[i]
            mask = sample['mask']
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()

            # 배경(0) 외 클래스가 있는지 확인
            unique_values = np.unique(mask)
            has_foreground = len(unique_values) > 1 or (len(unique_values) == 1 and unique_values[0] != 0)

            if has_foreground:
                filtered_indices.append(i)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")

    print(f"Filtering completed in {time.time() - filter_start:.2f}s. Found {len(filtered_indices)} valid samples.")

    # 타이밍 통계 출력
    if hasattr(dataset, 'print_timing_stats'):
        dataset.print_timing_stats()

    return Subset(dataset, filtered_indices)