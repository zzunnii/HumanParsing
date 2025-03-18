import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Union
from .dataset import PreprocessedDataset
# TrainConfig를 import 합니다.
def worker_init_fn(worker_id: int) -> None:
    """Initialize worker with different random seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    import numpy as np
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)

def build_dataloader(
        dataset: PreprocessedDataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
) -> DataLoader:
    """
    TrainConfig 대신 args에서 받은 값을 사용하여 dataloader를 구성
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,   # TrainConfig가 아닌 args에서 받은 값 사용
        num_workers=num_workers, # TrainConfig가 아닌 args에서 받은 값 사용
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