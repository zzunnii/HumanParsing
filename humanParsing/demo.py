import os
import cv2
import torch
import argparse
import numpy as np
from typing import Union, List, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

from parsing.models import ParsingModel  # 사용자의 모델 클래스 가정


def visualize_classes_separately(image, prediction, mode="tops", alpha=0.7, max_classes=None):
    """각 클래스별로 별도 시각화"""
    ALL_CLASSES = {
        "background": (0, 0, 0),        # 검정
        "rsleeve": (0, 255, 0),         # 초록
        "lsleeve": (0, 128, 255),       # 하늘
        "torsho": (0, 0, 255),          # 파랑
        "top_hidden": (255, 255, 255),    # 흰
        "hip": (255, 255, 0),           # 노랑
        "pants_rsleeve": (255, 0, 255), # 핑크
        "pants_lsleeve": (200, 0, 255), # 자주
        "pants_hidden": (255, 255, 255),  # 흰색
        "skirt": (0, 255, 255),         # 청록
        "skirt_hidden": (0, 200, 200),  # 진청록
        "hair": (139, 69, 19),          # 갈색
        "face": (255, 224, 189),        # 살구색
        "neck": (245, 222, 179),        # 연한 살구색
        "hat": (128, 128, 128),         # 회색
        "outer_rsleeve": (0, 255, 0),   # 초록 (rsleeve와 동일)
        "outer_lsleeve": (0, 128, 255), # 하늘 (lsleeve와 동일)
        "outer_torso": (0, 0, 255),     # 파랑 (torsho와 동일)
        "inner_rsleeve": (100, 255, 100),# 연초록
        "inner_lsleeve": (100, 200, 255),# 연청색
        "inner_torso": (100, 100, 255), # 연파랑
        "pants_hip": (255, 255, 0),     # 노랑 (hip과 동일)
        "right_arm": (255, 100, 100),   # 연분홍
        "left_arm": (255, 150, 150),    # 살구핑크
        "right_shoe": (255, 0, 0),      # 빨강
        "left_shoe": (200, 0, 0),       # 진홍
        "right_leg": (255, 165, 0),     # 주황
        "left_leg": (255, 140, 0)       # 다크주황
    }

    class_configs = {
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

    selected_classes = class_configs[mode]["names"]
    default_max = class_configs[mode]["default_max_classes"]
    max_classes = max_classes if max_classes is not None else default_max
    selected_classes = selected_classes[:max_classes]
    selected_colors = [ALL_CLASSES[class_name] for class_name in selected_classes]

    output = image.copy()
    overlay = np.zeros_like(image)

    for idx, color in enumerate(selected_colors):
        if idx < prediction.max() + 1:
            mask = (prediction == idx).astype(np.uint8)
            overlay[mask > 0] = color

    output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0.0)
    return output


class Demo:
    """학습 시 검증 변환과 정확히 일치하는 데모 인터페이스"""

    def __init__(
            self,
            checkpoint_path: str,
            num_classes: int = 20,
            device: torch.device = None,
            input_size: Tuple[int, int] = (320, 640),
            mode: str = "tops"
    ):
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.mode = mode

        self.model = ParsingModel(
            num_classes=num_classes,
            backbone_pretrained=False
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=max(input_size)),
            A.PadIfNeeded(
                min_height=input_size[0],
                min_width=input_size[1],
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def preprocess_image(self, image: Union[str, np.ndarray]) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image from {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_h, original_w = image.shape[:2]
        original_image = image.copy()
        transformed = self.transform(image=image)

        return transformed['image'].unsqueeze(0), original_image, (original_h, original_w)

    def calculate_scale_and_padding(self, original_size: Tuple[int, int]) -> Tuple[float, Tuple[int, int, int, int]]:
        original_h, original_w = original_size
        max_size = max(self.input_size)
        scale = min(max_size / max(original_h, original_w), 1.0)
        scaled_h, scaled_w = int(original_h * scale), int(original_w * scale)
        pad_h = max(0, self.input_size[0] - scaled_h)
        pad_w = max(0, self.input_size[1] - scaled_w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return scale, (pad_top, pad_bottom, pad_left, pad_right)

    @torch.no_grad()
    def predict(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_h, original_w = image_rgb.shape[:2]
        x, original_image, original_size = self.preprocess_image(image_rgb)
        scale, padding = self.calculate_scale_and_padding(original_size)

        x = x.to(self.device)
        outputs = self.model(x)
        pred = torch.argmax(outputs, dim=1)[0].cpu().numpy()

        pad_top, pad_bottom, pad_left, pad_right = padding
        scaled_h = int(original_h * scale)
        scaled_w = int(original_w * scale)

        if pad_top > 0 or pad_left > 0:
            h_start, h_end = pad_top, int(pad_top + scaled_h)
            w_start, w_end = pad_left, int(pad_left + scaled_w)
            pred_unpadded = pred[h_start:h_end, w_start:w_end]
        else:
            pred_unpadded = pred

        pred_resized = cv2.resize(
            pred_unpadded,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        return pred_resized, original_image

    def run_and_save(self, image_path, save_dir, alpha=0.7, max_classes=None):
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(image_path))[0]

        pred_mask, original_image = self.predict(image_path)
        overlay_image = visualize_classes_separately(
            image=original_image,
            prediction=pred_mask,
            mode=self.mode,
            alpha=alpha,
            max_classes=max_classes
        )

        overlay_path = os.path.join(save_dir, f"{file_name}_result.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

        print(f"Saved overlay result to {overlay_path} with size {overlay_image.shape[:2]}")


def parse_args():
    parser = argparse.ArgumentParser(description='Demo for Human Parsing Model')
    parser.add_argument('--checkpoint', type=str, default=r".\humanParsing\output\bottoms_20250310_215044\model_best.pth",
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str,
                        default=r".\parsingData\viton_dataset\cloth\bottom_014537.jpg",
                        help='Path to input image')
    parser.add_argument('--output-dir', type=str, default='demo_results',
                        help='Directory to save results')
    parser.add_argument('--num-classes', type=int, default=7,
                        help='Number of classes')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Transparency for overlay visualization (0-1)')
    parser.add_argument('--input-size', type=str, default='320,640',
                        help='Input size in format "height,width", default: 320,640')
    parser.add_argument('--mode', type=str, default='bottoms', choices=['tops', 'bottoms', 'model'],
                        help='Visualization mode: tops, bottoms, or model (default: bottoms)')
    parser.add_argument('--max-classes', type=int, default=None,
                        help='Maximum number of classes to visualize (overrides mode default if set)')
    return parser.parse_args()


def main():
    args = parse_args()
    input_size = tuple(map(int, args.input_size.split(',')))

    demo = Demo(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        input_size=input_size,
        mode=args.mode
    )
    demo.run_and_save(
        image_path=args.image,
        save_dir=args.output_dir,
        alpha=args.alpha,
        max_classes=args.max_classes
    )
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()