import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import cv2


class Visualizer:
    """Visualization tools for human parsing results."""

    def __init__(
            self,
            num_classes: int,
            class_names: Optional[List[str]] = None,
            class_colors: Optional[List[Tuple[int, int, int]]] = None
    ):
        """
        Initialize Visualizer.

        Args:
            num_classes (int): Number of classes
            class_names (list, optional): List of class names
            class_colors (list, optional): List of RGB colors for each class
        """
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]

        if class_colors is None:
            # Generate random colors
            rng = np.random.RandomState(42)
            colors = rng.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
            # Make first color white for background
            colors[0] = [255, 255, 255]
            self.class_colors = [tuple(map(int, color)) for color in colors]
        else:
            self.class_colors = class_colors

    def visualize_prediction(
            self,
            image: Union[torch.Tensor, np.ndarray],
            pred: Union[torch.Tensor, np.ndarray],
            target: Optional[Union[torch.Tensor, np.ndarray]] = None,
            alpha: float = 0.5
    ) -> np.ndarray:
        """
        Visualize prediction and optionally ground truth.
        """
        # Convert tensors to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor) and target is not None:
            target = target.cpu().numpy()

        # Handle different input formats
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)

        # Ensure image shape is correct (H, W, 3)
        if image.shape[-1] != 3:
            raise ValueError(f"Expected image with 3 channels, got shape {image.shape}")

        # Ensure image is uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Create color mask for predictions
        pred_mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(self.class_colors):
            pred_mask[pred == class_idx] = color

        # Ensure pred_mask has correct shape
        if pred_mask.shape[:2] != image.shape[:2]:
            pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Blend prediction with image
        blend = cv2.addWeighted(image, 1 - alpha, pred_mask, alpha, 0)

        if target is not None:
            # Create target color mask
            target_mask = np.zeros((*target.shape, 3), dtype=np.uint8)
            for class_idx, color in enumerate(self.class_colors):
                target_mask[target == class_idx] = color

            # Ensure target_mask has correct shape
            if target_mask.shape[:2] != image.shape[:2]:
                target_mask = cv2.resize(target_mask, (image.shape[1], image.shape[0]))

            # Stack images horizontally
            visualization = np.hstack([
                image,
                blend,
                cv2.addWeighted(image, 1 - alpha, target_mask, alpha, 0)
            ])
        else:
            visualization = np.hstack([image, blend])

        # Ensure final output is uint8 and correct format (H, W, 3)
        visualization = visualization.astype(np.uint8)
        if visualization.shape[-1] != 3:
            visualization = visualization.transpose(1, 2, 0)

        return visualization

    def create_legend(self) -> np.ndarray:
        """
        Create legend image showing class colors and names.

        Returns:
            np.ndarray: Legend image
        """
        # Calculate legend size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        padding = 10
        box_size = 20

        # Get text sizes
        text_sizes = [
            cv2.getTextSize(name, font, font_scale, thickness)[0]
            for name in self.class_names
        ]
        max_text_width = max(size[0] for size in text_sizes)
        text_height = max(size[1] for size in text_sizes)

        # Calculate image size
        legend_width = box_size + padding + max_text_width + padding
        legend_height = (box_size + padding) * self.num_classes

        # Create legend image
        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

        for i, (color, name) in enumerate(zip(self.class_colors, self.class_names)):
            # Draw color box
            y = i * (box_size + padding)
            cv2.rectangle(
                legend,
                (padding, y + padding),
                (padding + box_size, y + padding + box_size),
                color,
                -1
            )

            # Draw class name
            cv2.putText(
                legend,
                name,
                (padding * 2 + box_size, y + padding + box_size - padding // 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )

        return legend

    def plot_confusion_matrix(
            self,
            confusion_matrix: np.ndarray,
            normalize: bool = True,
            title: str = 'Confusion Matrix'
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            confusion_matrix (np.ndarray): Confusion matrix
            normalize (bool): Whether to normalize values
            title (str): Plot title

        Returns:
            plt.Figure: Matplotlib figure
        """
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / (
                    confusion_matrix.sum(axis=1, keepdims=True) + 1e-6
            )

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)

        # Add colorbar
        plt.colorbar(im)

        # Add labels
        tick_marks = np.arange(self.num_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.set_yticklabels(self.class_names)

        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = confusion_matrix.max() / 2.
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(
                    j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black"
                )

        plt.title(title)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()

        return fig

    def save_visualization(
            self,
            image: np.ndarray,
            save_path: str,
            create_parent: bool = True
    ):
        """
        Save visualization image.

        Args:
            image (np.ndarray): Image to save
            save_path (str): Path to save image
            create_parent (bool): Whether to create parent directories
        """
        if create_parent:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor to numpy array.

    Args:
        tensor (torch.Tensor): Input tensor

    Returns:
        np.ndarray: Output numpy array
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()