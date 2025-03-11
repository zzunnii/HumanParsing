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


def visualize_classes_separately(image, prediction, mode="tops", alpha=0.7, max_classes=20):
    """
    각 클래스별로 별도 시각화
    """
    # 모드별 클래스 이름 및 색상 정의
    class_configs = {
        "model": {
            "names": [
                "background", "hair", "face", "neck", "hat",
                "outer_rsleeve", "outer_lsleeve", "outer_torso",
                "inner_rsleeve", "inner_lsleeve", "inner_torso",
                "pants_hip", "pants_rsleeve", "pants_lsleeve",
                "skirt", "right_arm", "left_arm",
                "right_shoe", "left_shoe", "right_leg", "left_leg"
            ],
            "colors": [
                (0, 0, 0),  # background: 검정
                (139, 69, 19),  # hair: 갈색
                (255, 224, 189),  # face: 살구색
                (245, 222, 179),  # neck: 연한 살구색
                (128, 128, 128),  # hat: 회색
                (0, 255, 0),  # outer_rsleeve: 초록
                (0, 128, 255),  # outer_lsleeve: 하늘
                (0, 0, 255),  # outer_torso: 파랑
                (100, 255, 100),  # inner_rsleeve: 연초록
                (100, 200, 255),  # inner_lsleeve: 연청색
                (100, 100, 255),  # inner_torso: 연파랑
                (255, 255, 0),  # pants_hip: 노랑
                (255, 0, 255),  # pants_rsleeve: 핑크
                (200, 0, 255),  # pants_lsleeve: 자주
                (0, 255, 255),  # skirt: 청록
                (255, 100, 100),  # right_arm: 연분홍
                (255, 150, 150),  # left_arm: 살구핑크
                (255, 0, 0),  # right_shoe: 빨강
                (200, 0, 0),  # left_shoe: 진홍
                (255, 165, 0),  # right_leg: 주황
                (255, 140, 0)  # left_leg: 다크주황
            ]
        },
        "tops": {
            "names": ["background", "rsleeve", "lsleeve", "torsho", "top_hidden"],
            "colors": [
                (0, 0, 0),  # 배경: 검정
                (0, 255, 0),  # 오른소매: 초록
                (0, 128, 255),  # 왼소매: 하늘
                (0, 0, 255),  # 몸통: 파랑
                (128, 0, 255)  # 몸통(히든): 보라
            ]
        },
        "bottoms": {
            "names": ["background", "hip", "pants_rsleeve", "pants_lsleeve", "pants_hidden", "skirt", "skirt_hidden"],
            "colors": [
                (0, 0, 0),  # 배경: 검정
                (255, 255, 0),  # 엉덩이: 노랑
                (255, 0, 255),  # 바지_오른쪽: 핑크
                (200, 0, 255),  # 바지_왼쪽: 자주
                (150, 0, 150),  # 바지(히든): 보라
                (0, 255, 255),  # 치마: 청록
                (0, 200, 200)  # 치마(히든): 진청록
            ]
        }
    }

    # 현재 모드에 맞는 클래스 설정 가져오기
    if mode not in class_configs:
        mode = "model"  # 기본값

    class_names = class_configs[mode]["names"] if mode in class_configs else ["Background"] + [f"Class {i}" for i in
                                                                                               range(1, max_classes)]
    class_colors = class_configs[mode]["colors"] if mode in class_configs else [(0, 0, 0)] + [
        (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in
        range(1, max_classes)]

    # 원본 이미지 준비
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 3 and image.shape[0] == 3:  # CHW -> HWC
        image = image.transpose(1, 2, 0)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # 예측 결과 준비
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()

    # 클래스별 개별 이미지 생성
    class_images = []

    # 배경 이미지 (원본)
    background_img = image.copy()
    cv2.putText(background_img, "원본 이미지", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    class_images.append(background_img)

    # 전체 예측 결과 (모든 클래스 합친 것)
    full_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    for class_id in range(min(len(class_names), max_classes)):
        mask = prediction == class_id
        if np.any(mask):
            color = class_colors[class_id] if class_id < len(class_colors) else (128, 128, 128)
            full_mask[mask] = color

    full_prediction = image.copy()
    blend = cv2.addWeighted(full_prediction, 1 - alpha, full_mask, alpha, 0)
    cv2.putText(blend, "전체 예측", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    class_images.append(blend)

    # 각 클래스별 개별 시각화 (배경 제외)
    for class_id in range(1, min(len(class_names), max_classes)):
        mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        class_mask = prediction == class_id

        # 해당 클래스가 존재하는 경우에만 처리
        if np.any(class_mask):
            mask[class_mask] = class_colors[class_id] if class_id < len(class_colors) else (128, 128, 128)

            # 원본 이미지와 블렌딩
            class_img = image.copy()
            blend = cv2.addWeighted(class_img, 1 - alpha, mask, alpha, 0)

            # 클래스 이름 추가
            class_name = class_names[class_id] if class_id < len(class_names) else f"클래스 {class_id}"
            cv2.putText(blend, class_name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            class_images.append(blend)

    # 이미지 크기 통일
    if not class_images:
        return None

    max_h = max([img.shape[0] for img in class_images])
    max_w = max([img.shape[1] for img in class_images])

    resized_images = []
    for img in class_images:
        if img.shape[0] != max_h or img.shape[1] != max_w:
            # 패딩 추가
            padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            h, w = img.shape[:2]
            padded[:h, :w] = img
            resized_images.append(padded)
        else:
            resized_images.append(img)

    # 격자 레이아웃으로 배치 (3x3 또는 적절한 크기)
    max_per_row = 3
    rows = []

    # 각 행마다 같은 너비 가지도록 처리
    for i in range(0, len(resized_images), max_per_row):
        row_images = resized_images[i:min(i + max_per_row, len(resized_images))]
        if row_images:
            row = np.hstack(row_images)
            rows.append(row)

    # 모든 행의 너비를 통일
    if rows:
        max_row_width = max([row.shape[1] for row in rows])
        padded_rows = []

        for row in rows:
            if row.shape[1] < max_row_width:
                # 행의 너비가 최대 너비보다 작으면 패딩 추가
                padded_row = np.zeros((row.shape[0], max_row_width, 3), dtype=np.uint8)
                padded_row[:, :row.shape[1]] = row
                padded_rows.append(padded_row)
            else:
                padded_rows.append(row)

        return np.vstack(padded_rows)

    return None