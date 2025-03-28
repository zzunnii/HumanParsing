import os
import sys
# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Union
from .dataset import PreprocessedDataset
from humanParsing.HumanParsing.configs import TrainConfig

def worker_init_fn(worker_id: int) -> None:
    # 나머지 코드는 그대로 유지
    """Initialize worker with different random seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    import numpy as np
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable sized images
    """
    # Find maximum dimensions in the batch
    max_h = max([item['image'].shape[1] for item in batch])
    max_w = max([item['image'].shape[2] for item in batch])

    # Create new batch with resized tensors
    new_batch = []
    for item in batch:
        image = item['image']
        mask = item['mask']

        # Pad images to the max size
        if image.shape[1] != max_h or image.shape[2] != max_w:
            # Pad image
            padded_image = torch.zeros(3, max_h, max_w, device=image.device, dtype=image.dtype)
            padded_image[:, :image.shape[1], :image.shape[2]] = image

            # Pad mask
            padded_mask = torch.zeros(max_h, max_w, device=mask.device, dtype=mask.dtype)
            padded_mask[:mask.shape[0], :mask.shape[1]] = mask

            new_batch.append({'image': padded_image, 'mask': padded_mask})
        else:
            new_batch.append(item)

    # Stack the tensors
    return {
        'image': torch.stack([item['image'] for item in new_batch]),
        'mask': torch.stack([item['mask'] for item in new_batch])
    }

def build_dataloader(
        dataset: PreprocessedDataset,
        batch_size: int = None,
        num_workers: int = None,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
) -> DataLoader:
    """
    Build dataloader for the parsing dataset.

    Defaults for batch_size and num_workers are taken from TrainConfig.
    """
    if batch_size is None:
        batch_size = TrainConfig.BATCH_SIZE
    if num_workers is None:
        num_workers = TrainConfig.NUM_WORKERS

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn
    )

class DataPrefetcher:
    """Data prefetcher to speed up data loading."""
    def __init__(self, loader: DataLoader, device: torch.device):
        """
        Initialize prefetcher.

        Args:
            loader (DataLoader): DataLoader to prefetch from
            device (torch.device): Device to load data to
        """
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_data = None
        self._preload()

    def _preload(self):
        """Preload next batch of data."""
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return

        with torch.cuda.stream(self.stream):
            if isinstance(self.next_data, dict):
                for k, v in self.next_data.items():
                    if isinstance(v, torch.Tensor):
                        self.next_data[k] = v.to(device=self.device,
                                               non_blocking=True)
            else:
                self.next_data = [
                    v.to(self.device, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                    for v in self.next_data
                ]

    def next(self) -> Optional[Union[Dict, torch.Tensor]]:
        """
        Get next batch of data.

        Returns:
            Optional[Union[Dict, torch.Tensor]]: Next batch of data
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self._preload()
        return data

    def __iter__(self):
        return self

    def __next__(self):
        data = self.next()
        if data is None:
            raise StopIteration
        return data