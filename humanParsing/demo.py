import os
import cv2
import torch
import argparse
import numpy as np
from typing import Union,  Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

from HumanParsing.models import ParsingModel

from HumanParsing.analysis.birefnet import (
    setup_model as setup_birefnet_model,
    load_dataset_stats,
    process_image_for_segmentation
)



def visualize_classes_separately(image, prediction, mode="tops", alpha=0.7, max_classes=None):
    """각 클래스별로 별도 시각화"""
    ALL_CLASSES = {
        "background": (0, 0, 0),  # 검정
        "rsleeve": (0, 200, 0),  # 진한 초록
        "lsleeve": (0, 128, 255),  # 하늘파랑
        "torsho": (0, 0, 255),  # 파랑
        "top_hidden": (200, 200, 200),  # 회색
        "hip": (255, 255, 0),  # 노랑
        "pants_rsleeve": (255, 0, 255),  # 핑크/마젠타
        "pants_lsleeve": (180, 0, 255),  # 보라
        "pants_hidden": (150, 150, 150),  # 진한 회색
        "skirt": (0, 255, 255),  # 시안/청록
        "skirt_hidden": (0, 180, 180),  # 진한 청록
        "hair": (139, 69, 19),  # 갈색
        "face": (255, 200, 150),  # 더 선명한 살구색
        "neck": (230, 170, 120),  # 명확한 목 색상
        "hat": (70, 70, 70),  # 진한 회색
        "outer_rsleeve": (50, 180, 50),  # 이끼 초록
        "outer_lsleeve": (50, 100, 220),  # 진한 파랑
        "outer_torso": (20, 20, 180),  # 남색
        "inner_rsleeve": (100, 255, 100),  # 연한 초록
        "inner_lsleeve": (100, 200, 255),  # 연한 하늘
        "inner_torso": (100, 100, 255),  # 연한 파랑
        "pants_hip": (220, 220, 0),  # 진한 황색
        "right_arm": (255, 100, 100),  # 연한 빨강
        "left_arm": (255, 150, 150),  # 연한 분홍
        "right_shoe": (255, 0, 0),  # 빨강
        "left_shoe": (180, 0, 0),  # 암적색
        "right_leg": (255, 165, 0),  # 주황
        "left_leg": (220, 120, 0)  # 갈색 주황
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
    if image.shape[2] == 4:
        image_rgb = image[:, :, :3].copy()
    else:
        image_rgb = image.copy()
    selected_classes = class_configs[mode]["names"]
    default_max = class_configs[mode]["default_max_classes"]
    max_classes = max_classes if max_classes is not None else default_max
    selected_classes = selected_classes[:max_classes]
    selected_colors = [ALL_CLASSES[class_name] for class_name in selected_classes]

    output = image.copy()
    overlay = np.zeros_like(image_rgb)

    for idx, color in enumerate(selected_colors):
        if idx < prediction.max() + 1:
            mask = (prediction == idx).astype(np.uint8)
            overlay[mask > 0] = color

    output = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0.0)
    return output, overlay

def save_mask_only(prediction, save_path, mode="tops", max_classes=None):
    """마스크만 저장하는 함수"""
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

    default_max = class_configs[mode]["default_max_classes"]
    max_classes = max_classes if max_classes is not None else default_max

    # 마스크 시각화를 위한 이미지 생성 (각 클래스별로 고유 색상)
    mask_image = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

    for idx in range(1, max_classes):  # 0은 배경이므로 1부터 시작
        if idx < prediction.max() + 1:
            mask = (prediction == idx).astype(np.uint8)
            # 각 클래스별로 다른 색상 할당 (간단한 예시)
            color = ((idx * 37) % 256, (idx * 73) % 256, (idx * 113) % 256)
            mask_image[mask > 0] = color

    cv2.imwrite(save_path, mask_image)
    return mask_image


def apply_thin_edge_blur(image, mask, kernel_size=3, sigma=0.5):
    """마스크 경계선에 매우 얇은 블러 적용"""
    # 마스크가 3채널인 경우 1채널로 변환
    if len(mask.shape) == 3:
        mask_gray = mask[:, :, 0]
    else:
        mask_gray = mask

    # 더 작은 커널로 침식 적용 (가능한 얇은 경계선 생성)
    kernel = np.ones((3, 3), np.uint8)
    mask_erode = cv2.erode(mask_gray, kernel, iterations=1)
    edges = mask_gray - mask_erode

    # 매우 약한 가우시안 블러
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # 경계선에만 블러 적용
    edges_3d = np.repeat(edges[:, :, np.newaxis], 3, axis=2)
    result = np.where(edges_3d > 0, blurred, image)

    return result


def remove_misclassification(pred_mask, min_area=50):
    """작은 오분류 영역 제거"""
    cleaned_mask = np.zeros_like(pred_mask)

    # 각 클래스별로 처리
    for class_id in range(1, pred_mask.max() + 1):  # 0은 배경
        # 현재 클래스만 추출
        class_mask = (pred_mask == class_id).astype(np.uint8)

        # 작은 노이즈 제거 (opening 연산)
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)

        # 연결 컴포넌트 레이블링으로 각 영역 분석
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)

        # 각 연결 영역에 대해 처리
        for i in range(1, num_labels):  # 0은 배경
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:  # 너무 작은 영역은 제거
                labels[labels == i] = 0

        # 현재 클래스에 대한 마스크 업데이트 (작은 영역 제거됨)
        cleaned_component = (labels > 0).astype(np.uint8)
        cleaned_mask[cleaned_component > 0] = class_id

    return cleaned_mask

def apply_exact_mask_from_parsing(original_image, person_mask, pred_mask):
    """원래 이미지에서 파싱 마스크를 이용해 정확하게 경계 적용"""
    # person_mask는 배경 제거된 사람 마스크
    # pred_mask는 세그멘테이션 마스크

    # 배경은 투명하게 처리
    result = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)

    # 사람 영역 복사 (RGB 채널)
    result[:, :, :3] = original_image

    # 인물 마스크와 파싱 마스크를 결합한 최종 알파 채널
    # 배경(0)이 아닌 부분만 마스크로 설정
    final_mask = np.zeros_like(person_mask)
    final_mask[pred_mask > 0] = 255  # 배경이 아닌 모든 클래스에 대해 마스크 설정

    # 알파 채널에 마스크 적용
    result[:, :, 3] = final_mask

    return result


class Demo:
    """학습 시 검증 변환과 정확히 일치하는 데모 인터페이스"""

    def __init__(
            self,
            checkpoint_path: str,
            num_classes: int = 21,
            device: torch.device = None,
            input_size: Tuple[int, int] = (320, 640),
            mode: str = "tops",
            birefnet_token: str = None,
            stats_file: str = "./analysis_results/statistics_summary.json"
    ):
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.mode = mode

        # 파싱 모델 로드
        self.model = ParsingModel(
            num_classes=num_classes,
            backbone_pretrained=False
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 모드에 따라 적절한 모듈 및 함수 임포트
        if self.mode == "model":
            from HumanParsing.analysis.birefnet import setup_model, load_dataset_stats, process_image_for_segmentation
        else:  # tops 또는 bottoms
            from ClothesParsing.analysis.birefnet import setup_model, load_dataset_stats, process_image_for_segmentation

        # BiRefNet 모델 및 데이터셋 통계 로드
        self.birefnet_model = setup_model(token=birefnet_token)
        self.dataset_stats = load_dataset_stats(stats_file)
        self.process_func = process_image_for_segmentation

        # 비율을 유지하면서 크기 조정
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
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """이미지 배열을 입력받아 세그멘테이션 예측"""
        image_rgb = image.copy()
        if image_rgb.shape[2] == 4:  # RGBA인 경우 RGB로 변환
            image_rgb = image_rgb[:, :, :3]

        original_h, original_w = image_rgb.shape[:2]
        x, original_image, original_size = self.preprocess_image(image_rgb)
        scale, padding = self.calculate_scale_and_padding(original_size)

        x = x.to(self.device)
        outputs = self.model(x)
        pred = torch.argmax(outputs, dim=1)[0].cpu().numpy()

        # 패딩 제거 및 원본 크기로 복원
        pad_top, pad_bottom, pad_left, pad_right = padding
        scaled_h = int(original_h * scale)
        scaled_w = int(original_w * scale)

        if pad_top > 0 or pad_left > 0:
            h_start, h_end = pad_top, int(pad_top + scaled_h)
            w_start, w_end = pad_left, int(pad_left + scaled_w)
            pred_unpadded = pred[h_start:h_end, w_start:w_end]
        else:
            pred_unpadded = pred

        # 원래 이미지 크기로 복원
        pred_resized = cv2.resize(
            pred_unpadded,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        return pred_resized, original_image

    def process_and_segment(self, image_path, canvas_size=(720, 1280)):
        """배경 제거 -> 세그멘테이션 -> 앤티앨리어싱 처리 원스톱 파이프라인"""
        # 1. 원본 이미지 로드
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # 2. 배경 제거 및 캔버스 배치 (임포트된 모드별 함수 사용)
        processed_image, person_mask = self.process_func(
            self.birefnet_model,
            original_image,
            self.dataset_stats,
            final_canvas=canvas_size
        )

        # 3. 세그멘테이션 예측
        pred_mask, _ = self.predict(processed_image)

        # 3.5. 오분류 제거 (작은 영역 필터링)
        clean_pred_mask = remove_misclassification(pred_mask, min_area=20)  # 더 작은 영역 필터링

        # 4. 매우 얇은 엣지 블러 적용 (sigma 값을 더 낮게 설정)
        antialiased_image = apply_thin_edge_blur(processed_image, person_mask, kernel_size=3, sigma=0.5)

        # 5. 정확한 마스크 기반 이미지 생성
        final_segmented_image = apply_exact_mask_from_parsing(antialiased_image, person_mask, clean_pred_mask)

        # 6. 결과 생성 (RGBA 이미지에서 RGB만 추출하여 시각화)
        overlay_image, pure_mask = visualize_classes_separately(
            image=final_segmented_image[:, :, :3],  # RGB 채널만 사용
            prediction=clean_pred_mask,  # 정제된 마스크 사용
            mode=self.mode,
            alpha=0.7
        )

        return final_segmented_image, clean_pred_mask, overlay_image, pure_mask

    def run_and_save(self, image_path, save_dir, alpha=0.7, max_classes=None):
        """배경 제거 + 세그멘테이션 + 앤티앨리어싱 처리 결과 저장"""
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(image_path))[0]

        processed_image, pred_mask, overlay_image, pure_mask = self.process_and_segment(image_path)

        # 결과 저장
        processed_path = os.path.join(save_dir, f"{file_name}_processed.png")
        overlay_path = os.path.join(save_dir, f"{file_name}_result.png")
        mask_path = os.path.join(save_dir, f"{file_name}_mask.png")
        pure_mask_path = os.path.join(save_dir, f"{file_name}_pure_mask.png")

        # 처리된 이미지 저장
        cv2.imwrite(processed_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

        # 오버레이 이미지 저장
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

        # 순수 마스크 저장
        cv2.imwrite(pure_mask_path, cv2.cvtColor(pure_mask, cv2.COLOR_RGB2BGR))

        # 클래스별 마스크 저장
        save_mask_only(pred_mask, mask_path, mode=self.mode, max_classes=max_classes)

        print(f"Saved processed image to {processed_path}")
        print(f"Saved overlay result to {overlay_path}")
        print(f"Saved pure mask to {pure_mask_path}")
        print(f"Saved class mask to {mask_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Demo for Human Parsing with Background Removal')
    parser.add_argument('--image', type=str,
                        required=True, help='Path to image')
    parser.add_argument('--output-dir', type=str, default='demo_results',
                        help='Directory to save results')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='Transparency for overlay visualization (0-1)')
    parser.add_argument('--input-size', type=str, default='320,640',
                        help='Input size for parsing model in format "height,width"')
    parser.add_argument('--mode', type=str, default='model', choices=['tops', 'bottoms', 'model'],
                        help='Visualization mode: tops, bottoms, or model')
    parser.add_argument('--token', type=str, default=None,
                        help='Hugging Face token if needed')
    parser.add_argument('--canvas-size', type=str, default='720,1280',
                        help='Final canvas size in format "width,height"')
    return parser.parse_args()


def main():
    args = parse_args()
    input_size = tuple(map(int, args.input_size.split(',')))
    canvas_size = tuple(map(int, args.canvas_size.split(',')))

    # 모드에 따라 적절한 값을 변수로 설정
    if args.mode == 'model':
        num_classes = 21
        stats_file = r".\HumanParsing\analysis\analysis_results\statistics_summary.json"
        checkpoint = r".\HumanParsing\model_outputs\checkpoint_epoch18.pth"
    elif args.mode == 'tops':
        num_classes = 5
        stats_file = r".\ClothesParsing\analysis\analysis_results\tops\tops_statistics_summary.json"
        checkpoint = r".\ClothesParsing\output\tops_20250326_122103\checkpoint_epoch5.pth"
    else:  # bottoms
        num_classes = 7
        stats_file = r".\ClothesParsing\analysis\analysis_results\bottoms\bottoms_statistics_summary.json"
        checkpoint = r".\ClothesParsing\output\bottoms_20250326_123456\checkpoint_epoch5.pth"

    # Demo 객체 생성 시 변수 값 사용
    demo = Demo(
        checkpoint_path=checkpoint,
        num_classes=num_classes,
        input_size=input_size,
        mode=args.mode,
        birefnet_token=args.token,
        stats_file=stats_file
    )

    demo.run_and_save(
        image_path=args.image,
        save_dir=args.output_dir,
        alpha=args.alpha
    )
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()