import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import cv2


class Visualizer:
    def __init__(
            self,
            num_classes: int,
            mode: str = "",
            class_names: Optional[List[str]] = None,
            class_colors: Optional[List[Tuple[int, int, int]]] = None
    ):
        self.num_classes = num_classes
        self.mode = mode

        # 전체 클래스와 색상 정의
        self.ALL_CLASSES = {
            "background": (0, 0, 0),
            "rsleeve": (0, 255, 0),
            "lsleeve": (0, 128, 255),
            "torsho": (0, 0, 255),
            "top_hidden": (128, 0, 255),
            "hip": (255, 255, 0),
            "pants_rsleeve": (255, 0, 255),
            "pants_lsleeve": (200, 0, 255),
            "pants_hidden": (150, 0, 150),
            "skirt": (0, 255, 255),
            "skirt_hidden": (0, 200, 200),
            "hair": (139, 69, 19),
            "face": (255, 224, 189),
            "neck": (245, 222, 179),
            "hat": (128, 128, 128),
            "outer_rsleeve": (0, 255, 0),
            "outer_lsleeve": (0, 128, 255),
            "outer_torso": (0, 0, 255),
            "inner_rsleeve": (100, 255, 100),
            "inner_lsleeve": (100, 200, 255),
            "inner_torso": (100, 100, 255),
            "pants_hip": (255, 255, 0),
            "right_arm": (255, 100, 100),
            "left_arm": (255, 150, 150),
            "right_shoe": (255, 0, 0),
            "left_shoe": (200, 0, 0),
            "right_leg": (255, 165, 0),
            "left_leg": (255, 140, 0)
        }

        self.class_configs = {
            "tops": {
                "names": ["background", "rsleeve", "lsleeve", "torsho", "top_hidden"],
                "default_max_classes": 5
            },
            "bottoms": {
                "names": ["background", "hip", "pants_rsleeve", "pants_lsleeve", "pants_hidden", "skirt", "skirt_hidden"],
                "default_max_classes": 7
            },
            "model": {
                "names": [
                    "background", "hair", "face", "neck", "hat",
                    "outer_rsleeve", "outer_lsleeve", "outer_torso",
                    "inner_rsleeve", "inner_lsleeve", "inner_torso",
                    "pants_hip", "pants_rsleeve", "pants_lsleeve",
                    "skirt", "right_arm", "left_arm",
                    "right_shoe", "left_shoe", "right_leg", "left_leg"
                ],
                "default_max_classes": 21
            }
        }

        if mode not in self.class_configs:
            mode = "model"

        default_names = self.class_configs[mode]["names"]
        self.default_max_classes = self.class_configs[mode]["default_max_classes"]

        # 사용자 지정 값이 없으면 모드에 따른 기본값 사용
        if class_names is None:
            self.class_names = default_names
            if len(self.class_names) < num_classes:
                # 부족한 경우 기본 이름 추가
                self.class_names.extend([f"class_{i}" for i in range(len(self.class_names), num_classes)])
        else:
            self.class_names = class_names

        if class_colors is None:
            self.class_colors = [self.ALL_CLASSES.get(name, (128, 128, 128)) for name in default_names]
            if len(self.class_colors) < num_classes:
                print("색상부족")
                self.class_colors.extend([(128, 128, 128)] * (num_classes - len(self.class_colors)))
        else:
            self.class_colors = class_colors

        # 길이 체크 및 조정
        if len(self.class_names) < num_classes or len(self.class_colors) < num_classes:
            raise ValueError(f"Adjusted lists still provide fewer names/colors than num_classes ({num_classes})")
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
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor) and target is not None:
            target = target.cpu().numpy()

        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)

        if image.shape[-1] != 3:
            raise ValueError(f"Expected image with 3 channels, got shape {image.shape}")

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        pred_mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(self.class_colors):
            pred_mask[pred == class_idx] = color

        if pred_mask.shape[:2] != image.shape[:2]:
            pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        blend = cv2.addWeighted(image, 1 - alpha, pred_mask, alpha, 0)

        if target is not None:
            target_mask = np.zeros((*target.shape, 3), dtype=np.uint8)
            for class_idx, color in enumerate(self.class_colors):
                target_mask[target == class_idx] = color

            if target_mask.shape[:2] != image.shape[:2]:
                target_mask = cv2.resize(target_mask, (image.shape[1], image.shape[0]))

            visualization = np.hstack([
                image,
                blend,
                cv2.addWeighted(image, 1 - alpha, target_mask, alpha, 0)
            ])
        else:
            visualization = np.hstack([image, blend])

        visualization = visualization.astype(np.uint8)
        if visualization.shape[-1] != 3:
            visualization = visualization.transpose(1, 2, 0)

        return visualization

    def visualize_classes_separately(
            self,
            image: Union[torch.Tensor, np.ndarray],
            prediction: Union[torch.Tensor, np.ndarray],
            alpha: float = 0.7,
            max_classes: Optional[int] = None
    ) -> np.ndarray:
        """
        Visualize each class separately with mode-specific settings.
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()

        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # max_classes 설정
        max_classes = max_classes if max_classes is not None else self.default_max_classes
        selected_names = self.class_names[:max_classes]
        selected_colors = self.class_colors[:max_classes]

        # 클래스별 이미지 리스트
        class_images = []

        # 원본 이미지
        background_img = image.copy()
        cv2.putText(background_img, "Original Image", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        class_images.append(background_img)

        # 전체 예측 결과
        full_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        for class_id, color in enumerate(selected_colors):
            mask = prediction == class_id
            if np.any(mask):
                full_mask[mask] = color

        full_prediction = image.copy()
        blend = cv2.addWeighted(full_prediction, 1 - alpha, full_mask, alpha, 0)
        cv2.putText(blend, "Full Prediction", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        class_images.append(blend)

        # 각 클래스별 개별 시각화 (배경 제외)
        for class_id in range(1, min(len(selected_names), max_classes)):
            mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
            class_mask = prediction == class_id

            if np.any(class_mask):
                mask[class_mask] = selected_colors[class_id]
                class_img = image.copy()
                blend = cv2.addWeighted(class_img, 1 - alpha, mask, alpha, 0)
                class_name = selected_names[class_id]
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
                padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                h, w = img.shape[:2]
                padded[:h, :w] = img
                resized_images.append(padded)
            else:
                resized_images.append(img)

        # 격자 레이아웃으로 배치
        max_per_row = 3
        rows = []

        for i in range(0, len(resized_images), max_per_row):
            row_images = resized_images[i:min(i + max_per_row, len(resized_images))]
            if row_images:
                row = np.hstack(row_images)
                rows.append(row)

        if rows:
            max_row_width = max([row.shape[1] for row in rows])
            padded_rows = []

            for row in rows:
                if row.shape[1] < max_row_width:
                    padded_row = np.zeros((row.shape[0], max_row_width, 3), dtype=np.uint8)
                    padded_row[:, :row.shape[1]] = row
                    padded_rows.append(padded_row)
                else:
                    padded_rows.append(row)

            return np.vstack(padded_rows)

        return None

    def create_legend(self) -> np.ndarray:
        """Create legend image showing class colors and names."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        padding = 10
        box_size = 20

        text_sizes = [
            cv2.getTextSize(name, font, font_scale, thickness)[0]
            for name in self.class_names
        ]
        max_text_width = max(size[0] for size in text_sizes)
        text_height = max(size[1] for size in text_sizes)

        legend_width = box_size + padding + max_text_width + padding
        legend_height = (box_size + padding) * len(self.class_names)

        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

        for i, (color, name) in enumerate(zip(self.class_colors, self.class_names)):
            y = i * (box_size + padding)
            cv2.rectangle(
                legend,
                (padding, y + padding),
                (padding + box_size, y + padding + box_size),
                color,
                -1
            )
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
        """Plot confusion matrix."""
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / (
                    confusion_matrix.sum(axis=1, keepdims=True) + 1e-6
            )

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im)

        tick_marks = np.arange(self.num_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.set_yticklabels(self.class_names)

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
        """Save visualization image."""
        if create_parent:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()
