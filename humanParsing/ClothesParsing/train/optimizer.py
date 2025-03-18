import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Union
import torch.nn as nn


def get_parameter_groups(
        model: nn.Module,
        weight_decay: float = 0.0,
        skip_list: Optional[List[str]] = None
) -> List[Dict]:
    """
    Get parameter groups for optimizer.
    Separates parameters that should have weight decay from those that shouldn't.

    Args:
        model (nn.Module): Model to get parameters from
        weight_decay (float): Weight decay value
        skip_list (list, optional): List of parameter names to skip

    Returns:
        list: List of parameter group dictionaries
    """
    if skip_list is None:
        skip_list = []

    # Skip bn and bias by default
    skip_list = ['bias', 'bn'] + skip_list

    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(skip_name in name for skip_name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]


def build_optimizer(
    model: nn.Module,
    name: str = 'adamw',
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    backbone_lr_factor: float = 0.1,
    **kwargs
) -> Optimizer:
    """
    Build optimizer for training.

    Args:
        model (nn.Module): Model to optimize
        name (str): Optimizer name ('sgd', 'adam', 'adamw')
        lr (float): Learning rate
        weight_decay (float): Weight decay
        backbone_lr_factor (float): Learning rate multiplier for backbone
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer: Configured optimizer
    """
    # 파라미터를 두 그룹으로 나눔
    backbone_params = []
    decoder_params = []

    # 모델의 모든 파라미터를 순회하면서 분류
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in param_name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    # 옵티마이저 설정
    if name == 'adamw':
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr * backbone_lr_factor},
            {'params': decoder_params, 'lr': lr}
        ], weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': lr * backbone_lr_factor},
            {'params': decoder_params, 'lr': lr}
        ])
    elif name == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': backbone_params, 'lr': lr * backbone_lr_factor},
            {'params': decoder_params, 'lr': lr}
        ], momentum=0.9)
    else:
        raise ValueError(f'Unsupported optimizer: {name}')

    return optimizer

class WarmupScheduler(_LRScheduler):
    """
    Warmup learning rate scheduler.
    Gradually increases learning rate from 0 to base_lr.
    """

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_epochs: int,
            after_scheduler: Optional[_LRScheduler] = None
    ):
        """
        Initialize scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer
            warmup_epochs (int): Number of warmup epochs
            after_scheduler (Scheduler, optional): Scheduler to use after warmup
        """
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Get learning rates for each parameter group."""
        if self.last_epoch >= self.warmup_epochs:
            if self.after_scheduler is not None:
                if not self.finished:
                    self.after_scheduler.base_lrs = self.base_lrs
                    self.finished = True
                return self.after_scheduler.get_lr()
            return self.base_lrs

        return [
            base_lr * (self.last_epoch + 1) / self.warmup_epochs
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        """Step scheduler."""
        if self.finished and self.after_scheduler is not None:
            self.after_scheduler.step(epoch)
        else:
            super().step(epoch)


def build_scheduler(
        optimizer: Optimizer,
        name: str = 'cosine',
        epochs: int = 100,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        steps_per_epoch: int = None,  # 추가된 매개변수
        **kwargs
) -> _LRScheduler:
    """
    Build learning rate scheduler.
    """
    name = name.lower()

    if name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=min_lr
        )
    elif name == 'step':
        step_size = kwargs.get('step_size', epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif name == 'poly':
        power = kwargs.get('power', 0.9)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=epochs - warmup_epochs,
            power=power
        )
    elif name == 'onecycle':  # 새로운 옵션 추가
        max_lr = kwargs.get('max_lr', optimizer.param_groups[0]['lr'] * 10)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=kwargs.get('pct_start', 0.3),
            anneal_strategy='cos',
            div_factor=kwargs.get('div_factor', 25.0),
            final_div_factor=kwargs.get('final_div_factor', 1000.0)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {name}")

    if warmup_epochs > 0 and name != 'onecycle':  # onecycle에는 이미 warmup이 포함되어 있으므로 추가 warmup 불필요
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            after_scheduler=scheduler
        )

    return scheduler