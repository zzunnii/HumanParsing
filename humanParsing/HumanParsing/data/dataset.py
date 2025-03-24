import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, List, Tuple, Union
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

    # 역매핑 (ID → 이름)
    MODEL_CATEGORIES_INV = {v: k for k, v in MODEL_CATEGORIES.items()}


class PreprocessedDataset(Dataset):
    def __init__(self, data_dir, transforms=None, split='train', mode='model', use_cache=True):
        """
        Args:
            data_dir: 전처리된 person 디렉토리들이 모여있는 상위 경로
            transforms: Albumentations 등 원하는 변환
            split: 'train', 'val', 'test'
            mode: 'model' 또는 'item' (어느 데이터를 사용할지 지정)
            use_cache: 메모리 캐싱 사용 여부 (빠른 로딩을 위해)
        """
        self.data_dir = data_dir
        self.transforms = transforms
        self.split = split
        self.mode = mode
        self.use_cache = use_cache
        self.cache = {}  # 메모리 캐싱을 위한 딕셔너리

        # person 디렉토리 목록
        self.person_dirs = sorted([
            os.path.join(data_dir, d)
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        print(f"데이터셋 로드: {split} 분할에서 {len(self.person_dirs)}개의 샘플 발견")

    def __len__(self):
        return len(self.person_dirs)

    def get_person_info(self, idx):
        """인덱스로부터 해당 person 정보 가져오기"""
        person_path = self.person_dirs[idx]
        person_id = os.path.basename(person_path).split('_')[-1]  # 예: 'model_person_000001' -> '000001'

        # 메타데이터 로드 (있는 경우)
        metadata_path = os.path.join(person_path, "metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except:
                pass

        return {
            'person_id': person_id,
            'path': person_path,
            'metadata': metadata
        }

    def __getitem__(self, idx):
        # 캐싱 활성화되어 있고, 이미 처리된 항목이면 캐시에서 반환
        if self.use_cache and idx in self.cache:
            sample = self.cache[idx]

            # transforms가 적용되지 않은 원본 데이터만 캐싱
            if self.transforms is not None:
                transformed = self.transforms(**sample)
                if isinstance(transformed, dict):
                    sample = transformed
                else:
                    sample = {
                        'image': transformed['image'],
                        'mask': transformed['mask'].long() if 'mask' in transformed else None
                    }

            return sample

        person_path = self.person_dirs[idx]

        # 1) 원본 이미지 불러오기
        image_path = os.path.join(person_path, f"{self.mode}_image.jpg")

        # 파일이 없는 경우 대체 파일 시도
        if not os.path.exists(image_path) and self.mode == 'model':
            image_path = os.path.join(person_path, "model_image.jpg")

        image = cv2.imread(image_path)
        if image is None:
            # 유효하지 않은 이미지인 경우 대체 이미지 생성
            print(f"⚠️ 경고: 이미지를 불러올 수 없음: {image_path}")
            image = np.zeros((1280, 720, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape

        # 2) mask_info.json 불러오기
        mask_info_path = os.path.join(person_path, "mask_info.json")
        combined_mask = np.zeros((H, W), dtype=np.uint8)

        if os.path.exists(mask_info_path):
            try:
                with open(mask_info_path, 'r', encoding='utf-8') as f:
                    mask_info = json.load(f)

                # 3) 여러 이진 마스크를 합쳐서 다중 클래스 마스크 만들기
                mask_entries = mask_info.get(f"{self.mode}_masks", [])

                for mask_entry in mask_entries:
                    if self.mode == "model":
                        category = mask_entry.get("category")
                        class_id = Mapping.MODEL_CATEGORIES.get(category, 0)
                    else:
                        # item 모드일 경우 처리
                        product_type = mask_entry.get("product_type")
                        # 여기에 추가 로직 필요
                        class_id = 1  # 임시 값

                    mask_path = mask_entry.get("path", None)
                    if mask_path and os.path.exists(mask_path):
                        binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        # 크기 조정
                        if binary_mask.shape != (H, W):
                            binary_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                        combined_mask[binary_mask > 127] = class_id
            except Exception as e:
                print(f"⚠️ 경고: 마스크 처리 중 오류 발생: {e}")

        # 4) 기본 샘플 구성
        sample = {'image': image, 'mask': combined_mask}

        # 캐싱 활성화되어 있으면 원본 데이터 저장
        if self.use_cache:
            self.cache[idx] = sample.copy()

        # 5) transforms 적용
        if self.transforms is not None:
            transformed = self.transforms(**sample)
            if isinstance(transformed, dict):
                sample = transformed
            else:
                sample = {
                    'image': transformed['image'],
                    'mask': transformed['mask'].long() if 'mask' in transformed else None
                }

        # 6) Tensor 변환 (필요 시)
        if not torch.is_tensor(sample['image']):
            sample['image'] = T.ToTensor()(sample['image'])
        if not torch.is_tensor(sample['mask']):
            sample['mask'] = torch.tensor(sample['mask'], dtype=torch.long)

        return sample


def build_dataset(data_dir: str, transforms: Optional[object] = None, split: str = 'train',
                  mode='model', use_cache: bool = False) -> PreprocessedDataset:
    """
    Build preprocessed dataset.

    Args:
        data_dir: 데이터 디렉토리 경로
        transforms: 적용할 변환
        split: 데이터 분할 ('train', 'val', 'test')
        mode: 데이터 모드 ('model' 또는 'item')
        use_cache: 메모리 캐싱 사용 여부

    Returns:
        PreprocessedDataset: 구성된 데이터셋
    """
    return PreprocessedDataset(
        data_dir=data_dir,
        transforms=transforms,
        split=split,
        mode=mode,
        use_cache=use_cache
    )