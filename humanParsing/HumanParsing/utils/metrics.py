import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch import Tensor


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, window_size: int = 20):
        """
        Initialize AverageMeter.

        Args:
            window_size (int): Size of moving average window
        """
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []

    def update(self, val: float, n: int = 1):
        """
        Update statistics.

        Args:
            val (float): Value to update with
            n (int): Number of items this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        self.history.append(val)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    @property
    def moving_avg(self) -> float:
        """Get moving average of recent values."""
        return sum(self.history) / len(self.history) if self.history else 0


class SegmentationMetric:
    """Compute segmentation metrics including IoU, accuracy, etc."""

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None, device: Optional[torch.device] = None):
        """
        Initialize metric calculator.

        Args:
            num_classes (int): Number of classes
            ignore_index (int, optional): Index to ignore in metrics
            device (torch.device, optional): Device to store confusion matrix
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()

    def reset(self):
        """Reset confusion matrix."""
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device=self.device
        )

    def _fast_hist(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute confusion matrix for one image.

        Args:
            pred (Tensor): Prediction tensor
            label (Tensor): Label tensor

        Returns:
            Tensor: Confusion matrix
        """
        # Move tensors to the same device as confusion matrix
        pred = pred.to(self.device)
        label = label.to(self.device)

        mask = (label >= 0) & (label < self.num_classes)
        if self.ignore_index is not None:
            mask = mask & (label != self.ignore_index)

        hist = torch.bincount(
            self.num_classes * label[mask].int() + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        return hist

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update confusion matrix.

        Args:
            pred (Tensor): Prediction tensor
            target (Tensor): Target tensor
        """
        for p, t in zip(pred, target):
            self.confusion_matrix += self._fast_hist(p.flatten(), t.flatten())

    def get_scores(self) -> Dict[str, float]:
        """
        Compute various segmentation metrics.

        Returns:
            dict: Dictionary containing various metrics
        """
        hist = self.confusion_matrix.float()

        # Move to CPU for final calculations
        hist = hist.cpu()

        # Compute IoU for each class
        iu = torch.diag(hist) / (
                hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist) + 1e-6
        )

        # Mean IoU
        mean_iu = torch.mean(iu).item()

        # Per-class IoU
        class_iu = {i: iu[i].item() for i in range(self.num_classes)}

        # Pixel accuracy
        acc = torch.sum(torch.diag(hist)) / (torch.sum(hist) + 1e-6)

        # Mean pixel accuracy
        acc_cls = torch.mean(
            torch.diag(hist) / (hist.sum(dim=1) + 1e-6)
        )

        # F1 score
        precision = torch.diag(hist) / (hist.sum(dim=1) + 1e-6)
        recall = torch.diag(hist) / (hist.sum(dim=0) + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        mean_f1 = torch.mean(f1).item()

        return {
            'pixel_acc': acc.item(),
            'mean_acc': acc_cls.item(),
            'mean_iu': mean_iu,
            'mean_f1': mean_f1,
            'class_iu': class_iu,  # ← 각 클래스별 IoU 딕셔너리
            'class_f1': {i: f1[i].item() for i in range(self.num_classes)}
        }

    def get_confusion_matrix(self) -> Tensor:
        """Get confusion matrix."""
        return self.confusion_matrix.clone()


def compute_metric_for_each_image(
        pred: Tensor,
        target: Tensor,
        num_classes: int,
        ignore_index: Optional[int] = None
) -> List[Dict[str, float]]:
    """
    Compute metrics for each image in batch.

    Args:
        pred (Tensor): Prediction tensor of shape (B, H, W)
        target (Tensor): Target tensor of shape (B, H, W)
        num_classes (int): Number of classes
        ignore_index (int, optional): Index to ignore in metrics

    Returns:
        list: List of metric dictionaries for each image
    """
    metrics = []
    for p, t in zip(pred, target):
        metric = SegmentationMetric(num_classes, ignore_index)
        metric.update(p.unsqueeze(0), t.unsqueeze(0))
        metrics.append(metric.get_scores())
    return metrics