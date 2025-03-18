import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List, Tuple


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    Supports multi-class segmentation.
    """

    def __init__(
            self,
            smooth: float = 1.0,
            ignore_index: Optional[int] = None,
            weight: Optional[torch.Tensor] = None,
            mode: str = 'multiclass'  # 'binary' or 'multiclass'
    ):
        """
        Initialize DiceLoss.

        Args:
            smooth (float): Smoothing factor to avoid division by zero
            ignore_index (int, optional): Index to ignore in loss calculation
            weight (tensor, optional): Manual rescaling weight for each class
            mode (str): 'binary' for binary segmentation, 'multiclass' for multi-class segmentation
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight
        self.mode = mode

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            logits (tensor): Raw predictions (B, C, H, W)
            targets (tensor): Ground truth labels (B, H, W)

        Returns:
            tensor: Dice loss value
        """
        num_classes = logits.shape[1]

        # Create a mask of valid pixels
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            # Clone targets and replace ignored indices with a valid class (e.g., 0)
            targets = targets.clone()
            targets[~mask] = 0
        else:
            mask = torch.ones_like(targets, dtype=torch.bool)

        # Handle binary case separately for efficiency
        if self.mode == 'binary' and num_classes == 1:
            # For binary segmentation
            probs = torch.sigmoid(logits[:, 0]).float()
            targets_binary = (targets > 0).float()

            # Apply mask
            targets_binary = targets_binary * mask.float()
            probs = probs * mask.float()

            # Calculate Dice score
            dims = (0, 1, 2)  # Sum over batch, height, width
            intersection = torch.sum(probs * targets_binary, dims)
            cardinality = torch.sum(probs + targets_binary, dims)
            dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)

            return 1 - dice_score
        else:
            # For multi-class segmentation
            # Convert targets to one-hot encoding
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

            # Expand mask to have channel dimension and apply it
            mask = mask.unsqueeze(1).float()
            targets_one_hot = targets_one_hot * mask

            # Apply softmax to logits
            probs = F.softmax(logits, dim=1)

            # Calculate Dice score for each class
            dims = (0, 2, 3)  # Sum over batch, height, width
            intersection = torch.sum(probs * targets_one_hot, dims)
            cardinality = torch.sum(probs + targets_one_hot, dims)
            dice_scores = (2. * intersection + self.smooth) / (cardinality + self.smooth)

            # Apply class weights if specified
            if self.weight is not None:
                dice_scores = dice_scores * self.weight.to(dice_scores.device)

            # Return mean Dice loss over classes
            return 1 - dice_scores.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """

    def __init__(
            self,
            alpha: Optional[torch.Tensor] = None,
            gamma: float = 2.0,
            ignore_index: Optional[int] = None,
            reduction: str = 'mean'
    ):
        """
        Initialize FocalLoss.

        Args:
            alpha (tensor, optional): Class weights
            gamma (float): Focusing parameter
            ignore_index (int, optional): Index to ignore
            reduction (str): Reduction type - 'none', 'mean', 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal loss"""
        log_probs = F.log_softmax(logits, dim=1)  # [B, num_classes, H, W]
        probs = torch.exp(log_probs)  # [B, num_classes, H, W]

        # target 클래스의 확률만 선택
        pt = probs.gather(dim=1, index=targets.unsqueeze(1))  # [B, 1, H, W]
        focal_term = (1 - pt) ** self.gamma  # [B, 1, H, W]

        ce_loss = F.nll_loss(
            log_probs,
            targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction='none'
        )  # [B, H, W]

        loss = focal_term.squeeze(1) * ce_loss  # [B, H, W]

        if self.reduction == 'mean':
            if self.ignore_index is not None:
                valid_mask = targets != self.ignore_index
                loss = loss[valid_mask].mean() if valid_mask.sum() > 0 else loss.sum() * 0
            else:
                loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss for better segmentation edges.
    Computes weighted cross-entropy loss focused on boundary regions.
    """

    def __init__(
            self,
            theta0: float = 3,
            theta: float = 5,
            ignore_index: Optional[int] = None,
            weight: Optional[torch.Tensor] = None,
            reduce: bool = True
    ):
        """
        Initialize BoundaryLoss.

        Args:
            theta0 (float): Internal boundary width parameter
            theta (float): External boundary width parameter
            ignore_index (int, optional): Index to ignore
            weight (tensor, optional): Class weights
            reduce (bool): Whether to reduce loss to scalar
        """
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduce = reduce

    def _compute_boundaries(self, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Compute boundary regions for each class.

        Args:
            targets (tensor): Ground truth labels (B, H, W)
            num_classes (int): Number of classes

        Returns:
            tensor: Boundary weights (B, H, W)
        """
        # One-hot encoded targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Compute boundaries using image gradient
        boundaries = torch.zeros_like(targets, dtype=torch.float32)

        for c in range(num_classes):
            # Extract binary mask for current class
            mask = targets_one_hot[:, c]

            # Compute boundaries using Sobel filter approximation
            device = mask.device
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1,
                                                                                                                  3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1,
                                                                                                                  3, 3)

            # Pad before convolution
            mask_padded = F.pad(mask.unsqueeze(1), (1, 1, 1, 1), mode='replicate')

            # Apply Sobel filters
            grad_x = F.conv2d(mask_padded, sobel_x).squeeze(1)
            grad_y = F.conv2d(mask_padded, sobel_y).squeeze(1)

            # Compute gradient magnitude
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

            # Add to overall boundaries
            boundaries = torch.max(boundaries, grad_mag)

        return boundaries

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary-aware loss.

        Args:
            logits (tensor): Raw predictions (B, C, H, W)
            targets (tensor): Ground truth labels (B, H, W)

        Returns:
            tensor: Loss value
        """
        num_classes = logits.shape[1]

        # Compute boundary regions
        boundaries = self._compute_boundaries(targets, num_classes)

        # Create boundary weight map with exponential decay based on distance
        weight_map = torch.exp(torch.clamp(boundaries * self.theta0, 0, self.theta))

        # Ignore specific index if provided
        if self.ignore_index is not None:
            ignore_mask = (targets != self.ignore_index).float()
            weight_map = weight_map * ignore_mask

        # Compute weighted cross-entropy loss
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, targets,
                          weight=self.weight,
                          ignore_index=self.ignore_index if self.ignore_index is not None else -100,
                          reduction='none')

        # Apply boundary weights
        weighted_loss = loss * weight_map

        # Reduce if required
        if self.reduce:
            if self.ignore_index is not None:
                valid_mask = targets != self.ignore_index
                weighted_loss = weighted_loss[valid_mask].mean() if valid_mask.sum() > 0 else weighted_loss.sum() * 0
            else:
                weighted_loss = weighted_loss.mean()

        return weighted_loss


class LovaszLoss(nn.Module):
    """
    Lovasz-Softmax loss for multi-class semantic segmentation.
    Provides better optimization for mIoU.
    """

    def __init__(
            self,
            ignore_index: Optional[int] = None,
            per_class: bool = False
    ):
        """
        Initialize Lovasz loss.

        Args:
            ignore_index (int, optional): Index to ignore
            per_class (bool): Whether to compute per-class loss
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.per_class = per_class

    def _lovasz_grad(self, gt_sorted):
        """Compute gradient of the Lovasz extension w.r.t sorted errors"""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        iou = 1.0 - intersection / union

        # Extended to handle case where every prediction is correct
        if p > 1:
            iou[:-1] = iou[:-1] - iou[1:]
        return iou

    def _lovasz_softmax_flat(self, probs, labels, ignore=None):
        """Multi-class Lovasz-Softmax loss"""
        C = probs.shape[1]
        losses = []

        for c in range(C):
            if c == ignore:
                continue

            # Foreground for class c
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue

            class_pred = probs[:, c]
            errors = (fg - class_pred).abs()

            # Sort errors
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
            fg_sorted = fg[perm]

            # Compute Lovasz gradient and loss
            grad = self._lovasz_grad(fg_sorted)
            loss = torch.dot(errors_sorted, grad)
            losses.append(loss)

        if len(losses) > 0:
            return torch.stack(losses).mean() if self.per_class else sum(losses)
        else:
            return torch.tensor(0.0, device=probs.device, requires_grad=True)

    def forward(self, logits, targets):
        """Forward pass"""
        # Get probabilities
        probs = F.softmax(logits, dim=1)

        # Flatten
        B, C, H, W = probs.shape
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets = targets.view(-1)

        # Ignore specified indices
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            probs = probs[valid_mask]
            targets = targets[valid_mask]

        if probs.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return self._lovasz_softmax_flat(probs, targets, ignore=self.ignore_index)


class CombinedLoss(nn.Module):
    """
    Combination of multiple loss functions for human parsing.
    Default: CrossEntropy + Dice Loss
    """

    def __init__(
            self,
            num_classes: int,
            ce_weight: float = 1.0,
            dice_weight: float = 1.0,
            focal_weight: float = 0.0,
            boundary_weight: float = 0.0,
            lovasz_weight: float = 0.0,
            class_weights: Optional[Union[List[float], torch.Tensor]] = None,
            ignore_index: int = 255,
            focal_gamma: float = 2.0,
            dice_smooth: float = 1.0,
            boundary_theta0: float = 3.0,
            boundary_theta: float = 5.0
    ):
        """
        Initialize CombinedLoss.

        Args:
            num_classes (int): Number of classes
            ce_weight (float): Weight for CrossEntropy loss
            dice_weight (float): Weight for Dice loss
            focal_weight (float): Weight for Focal loss
            boundary_weight (float): Weight for Boundary loss
            lovasz_weight (float): Weight for Lovasz loss
            class_weights (list/tensor): Class weights
            ignore_index (int): Index to ignore
            focal_gamma (float): Focusing parameter for Focal loss
            dice_smooth (float): Smoothing factor for Dice loss
            boundary_theta0 (float): Internal boundary width parameter
            boundary_theta (float): External boundary width parameter
        """
        super().__init__()

        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights)
            class_weights = class_weights.float()

        self.cross_entropy = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        ) if ce_weight > 0 else None

        self.dice = DiceLoss(
            smooth=dice_smooth,
            ignore_index=ignore_index,
            weight=class_weights
        ) if dice_weight > 0 else None

        self.focal = FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            ignore_index=ignore_index
        ) if focal_weight > 0 else None

        self.boundary = BoundaryLoss(
            theta0=boundary_theta0,
            theta=boundary_theta,
            ignore_index=ignore_index,
            weight=class_weights
        ) if boundary_weight > 0 else None

        self.lovasz = LovaszLoss(
            ignore_index=ignore_index
        ) if lovasz_weight > 0 else None

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.lovasz_weight = lovasz_weight

        self.num_losses = sum(w > 0 for w in [ce_weight, dice_weight, focal_weight, boundary_weight, lovasz_weight])

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            logits (tensor): Raw predictions (B, C, H, W)
            targets (tensor): Ground truth labels (B, H, W)

        Returns:
            tensor: Combined loss value
        """
        loss = 0

        if self.cross_entropy is not None:
            ce_loss = self.ce_weight * self.cross_entropy(logits, targets)
            loss += ce_loss

        if self.dice is not None:
            dice_loss = self.dice_weight * self.dice(logits, targets)
            loss += dice_loss

        if self.focal is not None:
            focal_loss = self.focal_weight * self.focal(logits, targets)
            loss += focal_loss

        if self.boundary is not None:
            boundary_loss = self.boundary_weight * self.boundary(logits, targets)
            loss += boundary_loss

        if self.lovasz is not None:
            lovasz_loss = self.lovasz_weight * self.lovasz(logits, targets)
            loss += lovasz_loss

        return loss


def build_criterion(
        num_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.0,
        boundary_weight: float = 0.0,
        lovasz_weight: float = 0.0,
        class_weights: Optional[List[float]] = None,
        **kwargs
) -> nn.Module:
    """
    Build loss criterion.

    Args:
        num_classes (int): Number of classes
        ce_weight (float): Weight for CrossEntropy loss
        dice_weight (float): Weight for Dice loss
        focal_weight (float): Weight for Focal loss
        boundary_weight (float): Weight for Boundary loss
        lovasz_weight (float): Weight for Lovasz loss
        class_weights (list): Class weights
        **kwargs: Additional parameters

    Returns:
        nn.Module: Combined loss module
    """
    return CombinedLoss(
        num_classes=num_classes,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        focal_weight=focal_weight,
        boundary_weight=boundary_weight,
        lovasz_weight=lovasz_weight,
        class_weights=class_weights,
        **kwargs
    )