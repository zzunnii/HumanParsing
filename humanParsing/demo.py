import os
import cv2
import torch
import argparse
import numpy as np
from typing import Union, List, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

from parsing.models import ParsingModel
from parsing.utils import Visualizer


class Demo:
    """학습 시 검증 변환과 정확히 일치하는 데모 인터페이스"""

    def __init__(
            self,
            checkpoint_path: str,
            num_classes: int = 20,
            device: torch.device = None,
            input_size: Tuple[int, int] = (320, 640)  # 학습 시와 동일한 입력 크기
    ):
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size

        # 모델 초기화
        self.model = ParsingModel(
            num_classes=num_classes,
            backbone_pretrained=False
        )

        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 정확히 학습 시 검증 변환과 동일한 변환 설정
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

        # Visualizer 초기화
        self.visualizer = self.initialize_custom_visualizer(num_classes)

    def initialize_custom_visualizer(self, num_classes):
        # TOP_CATEGORIES와 일치하는 색상 매핑 사용
        mode = 'botoms'
        if mode == 'model' :
            custom_colors = [

            ]
        elif mode == 'tops' :
            custom_colors = [
                (0, 0, 0),  # 배경: 검정
                (0, 255, 0),  # 오른소매: 초록
                (0, 128, 255),  # 왼소매: 하늘
                (0, 0, 255),  # 몸통: 파랑
                (128, 0, 255)  # 몸통(히든): 보라
            ]
        elif mode == 'botoms' :
            custom_colors = [
                (0, 0, 0),  # 배경: 검정
                (255, 255, 0),  # 엉덩이: 노랑
                (255, 0, 255),  # 바지_오른쪽: 핑크
                (200, 0, 255),  # 바지_왼쪽: 자주
                (150, 0, 150),  # 바지(히든): 보라
                (0, 255, 255),  # 치마: 청록
                (0, 200, 200)  # 치마(히든): 진청록
            ]
        return Visualizer(num_classes=num_classes, class_colors=custom_colors)

    def preprocess_image(self, image: Union[str, np.ndarray]) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        """학습과 동일한 방식으로 이미지 전처리"""
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image from {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 원본 크기 저장
        original_h, original_w = image.shape[:2]

        # 크기 조정 전 원본 이미지 복사
        original_image = image.copy()

        # 변환 적용
        transformed = self.transform(image=image)

        return transformed['image'].unsqueeze(0), original_image, (original_h, original_w)

    def calculate_scale_and_padding(self, original_size: Tuple[int, int]) -> Tuple[float, Tuple[int, int, int, int]]:
        """원본 이미지 크기를 기반으로 스케일과 패딩 계산"""
        original_h, original_w = original_size

        # LongestMaxSize 로직과 동일하게 스케일 계산
        max_size = max(self.input_size)
        scale = min(max_size / max(original_h, original_w), 1.0)

        # 스케일링 후 크기 계산
        scaled_h, scaled_w = int(original_h * scale), int(original_w * scale)

        # 패딩 계산
        pad_h = max(0, self.input_size[0] - scaled_h)
        pad_w = max(0, self.input_size[1] - scaled_w)

        # 좌우상하 패딩 계산
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return scale, (pad_top, pad_bottom, pad_left, pad_right)

    @torch.no_grad()
    def predict(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """이미지 예측 및 원본 크기로 결과 반환"""
        # 원본 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 원본 크기 저장
        original_h, original_w = image_rgb.shape[:2]

        # 전처리 및 스케일/패딩 정보 계산
        x, original_image, original_size = self.preprocess_image(image_rgb)
        scale, padding = self.calculate_scale_and_padding(original_size)

        # 모델 추론
        x = x.to(self.device)
        outputs = self.model(x)

        # 디버깅 출력
        print(f"Output shape: {outputs.shape}")
        logits = outputs[0].cpu()
        probs = torch.softmax(logits, dim=0).numpy()
        print(f"Probability range: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
        print(f"Class distribution: {np.bincount(probs.argmax(axis=0).flatten())}")

        # 클래스 예측
        pred = torch.argmax(outputs, dim=1)[0].cpu().numpy()

        # 패딩 제거 및 원본 크기로 복원
        pad_top, pad_bottom, pad_left, pad_right = padding
        scaled_h = int(original_h * scale)
        scaled_w = int(original_w * scale)

        # 패딩 제거
        if pad_top > 0 or pad_left > 0:
            h_start, h_end = pad_top, int(pad_top + scaled_h)
            w_start, w_end = pad_left, int(pad_left + scaled_w)
            pred_unpadded = pred[h_start:h_end, w_start:w_end]
        else:
            pred_unpadded = pred

        # 원본 크기로 리사이징
        pred_resized = cv2.resize(
            pred_unpadded,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        return pred_resized, original_image

    def create_overlay(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """마스크와 이미지 오버레이 생성"""
        # 마스크를 컬러로 변환
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(self.visualizer.class_colors):
            colored_mask[mask == class_idx] = color

        # 원본 이미지와 마스크 합성
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        return overlay

    def run_and_save(self, image_path, save_dir, alpha=0.7):
        """예측 실행 및 결과 저장"""
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(image_path))[0]

        # 예측 실행
        pred_mask, original_image = self.predict(image_path)

        # 예측 클래스 출력
        unique_classes = np.unique(pred_mask)
        print(f"Predicted classes: {unique_classes.tolist()}")

        # 오버레이 이미지 생성
        overlay_image = self.create_overlay(original_image, pred_mask, alpha)

        # 순수 마스크 생성 (배경은 검정색으로)
        pure_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(self.visualizer.class_colors):
            if class_idx == 0:  # 배경은 검정색으로
                continue
            pure_mask[pred_mask == class_idx] = color

        # 결과 저장
        overlay_path = os.path.join(save_dir, f"{file_name}_result.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

        # 순수 마스크 저장
        mask_path = os.path.join(save_dir, f"{file_name}_mask.png")
        cv2.imwrite(mask_path, cv2.cvtColor(pure_mask, cv2.COLOR_RGB2BGR))

        print(f"Saved overlay result to {overlay_path} with size {overlay_image.shape[:2]}")
        print(f"Saved colored mask to {mask_path} with size {pure_mask.shape[:2]}")


def parse_args():
    parser = argparse.ArgumentParser(description='Demo for Human Parsing Model')
    parser.add_argument('--checkpoint', type=str, default=r"C:\Users\tjdwn\GitHub\tryon\humanParsing\output\bottoms_20250310_215044\model_best.pth",
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str,
                        default=r"C:\Users\tjdwn\OneDrive\Desktop\botoms.png",
                        help='Path to input image')
    parser.add_argument('--output-dir', type=str, default='demo_results',
                        help='Directory to save results')
    parser.add_argument('--num-classes', type=int, default=7,
                        help='Number of classes')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Transparency for overlay visualization (0-1)')
    parser.add_argument('--input-size', type=str, default='320,640',
                        help='Input size in format "height,width", default: 320,640')
    return parser.parse_args()


def main():
    args = parse_args()

    # 입력 크기 파싱
    input_size = tuple(map(int, args.input_size.split(',')))

    demo = Demo(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        input_size=input_size
    )
    demo.run_and_save(
        image_path=args.image,
        save_dir=args.output_dir,
        alpha=args.alpha
    )
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()