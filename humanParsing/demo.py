import os
import cv2
import torch
import argparse
import numpy as np
from typing import Union, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

from HumanParsing.models import ParsingModel

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
        "tops": {"names": ["background", "rsleeve", "lsleeve", "torsho", "top_hidden"], "default_max_classes": 5},
        "bottoms": {"names": ["background", "hip", "pants_rsleeve", "pants_lsleeve", "pants_hidden", "skirt", "skirt_hidden"], "default_max_classes": 7},
        "model": {"names": ["background", "hair", "face", "neck", "hat", "outer_rsleeve", "outer_lsleeve", "outer_torso", "inner_rsleeve", "inner_lsleeve", "inner_torso", "pants_hip", "pants_rsleeve", "pants_lsleeve", "skirt", "right_arm", "left_arm", "right_shoe", "left_shoe", "right_leg", "left_leg"], "default_max_classes": 21}
    }

    default_max = class_configs[mode]["default_max_classes"]
    max_classes = max_classes if max_classes is not None else default_max

    mask_image = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for idx in range(1, max_classes):
        if idx < prediction.max() + 1:
            mask = (prediction == idx).astype(np.uint8)
            color = ((idx * 37) % 256, (idx * 73) % 256, (idx * 113) % 256)
            mask_image[mask > 0] = color

    cv2.imwrite(save_path, mask_image)
    return mask_image

def apply_edge_processing(image, mask, kernel_size=5, sigma=1.0):
    """개선된 경계 처리 함수"""
    if len(mask.shape) == 3:
        mask_gray = mask[:, :, 0]
    else:
        mask_gray = mask

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_gray, kernel, iterations=1)
    eroded = cv2.erode(mask_gray, kernel, iterations=1)
    edges = dilated - eroded

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    edges_3d = np.repeat(edges[:, :, np.newaxis], 3, axis=2)
    weight_map = edges_3d.astype(np.float32) / 255.0

    result = image.copy()
    for i in range(3):
        result[:, :, i] = image[:, :, i] * (1 - weight_map[:, :, i]) + blurred[:, :, i] * weight_map[:, :, i]
    return result

def remove_misclassification(pred_mask, min_area=50):
    """작은 오분류 영역 제거"""
    cleaned_mask = np.zeros_like(pred_mask)
    for class_id in range(1, pred_mask.max() + 1):
        class_mask = (pred_mask == class_id).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned_mask[labels == i] = class_id
    return cleaned_mask

def apply_exact_mask_from_parsing(original_image, person_mask, pred_mask):
    """원래 이미지에서 파 управления싱 마스크를 이용해 정확하게 경계 적용"""
    result = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)
    result[:, :, :3] = original_image
    final_mask = np.zeros_like(person_mask)
    final_mask[pred_mask > 0] = 255
    result[:, :, 3] = final_mask
    return result

def class_boundaries(pred_mask):
    """클래스 간 경계 개선"""
    edges = np.zeros_like(pred_mask, dtype=np.uint8)
    for class_id in range(1, pred_mask.max() + 1):
        class_mask = (pred_mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edges, contours, -1, 1, thickness=2)

    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    blending_region = dilated_edges > 0

    improved_mask = pred_mask.copy()
    y_indices, x_indices = np.where(blending_region)
    for y, x in zip(y_indices, x_indices):
        neighborhood = pred_mask[max(0, y - 1):min(pred_mask.shape[0], y + 2),
                       max(0, x - 1):min(pred_mask.shape[1], x + 2)]
        if neighborhood.size > 0:
            values, counts = np.unique(neighborhood, return_counts=True)
            if len(values) > 1 or (len(values) == 1 and values[0] != 0):
                most_common = values[np.argmax(counts)]
                improved_mask[y, x] = most_common
    return improved_mask

class Demo:
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

        self.model = ParsingModel(num_classes=num_classes, backbone_pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.mode == "model":
            from HumanParsing.analysis.birefnet import setup_model, load_dataset_stats, process_image_for_segmentation
        else:
            from ClothesParsing.analysis.birefnet import setup_model, load_dataset_stats, process_image_for_segmentation

        self.birefnet_model = setup_model(token=birefnet_token)
        self.dataset_stats = load_dataset_stats(stats_file)
        self.process_func = process_image_for_segmentation

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=max(input_size)),
            A.PadIfNeeded(min_height=input_size[0], min_width=input_size[1], border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
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
        if image_rgb.shape[2] == 4:
            image_rgb = image_rgb[:, :, :3]

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

        pred_resized = cv2.resize(pred_unpadded, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        return pred_resized, original_image

    def get_bounding_box(self, mask):
        """마스크에서 바운딩 박스를 계산"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return 0, 0, 0, 0
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return x1, y1, x2 - x1, y2 - y1

    def process_and_segment(self, image_path, canvas_size=(720, 1280)):
        """배경 제거 -> 세그멘테이션 -> 앤티앨리어싱 처리 원스톱 파이프라인"""
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # 변환 정보도 받아옴
        processed_image, person_mask, original_img, original_mask, transform_info = self.process_func(
            self.birefnet_model,
            original_image,
            self.dataset_stats,
            final_canvas=canvas_size
        )

        if transform_info is None:
            # 사람/물체가 감지되지 않은 경우
            return None, None, None, None, None

        # 세그멘테이션 예측
        pred_mask, _ = self.predict(processed_image)
        clean_pred_mask = remove_misclassification(pred_mask, min_area=20)
        boundaies_mask = class_boundaries(clean_pred_mask)

        # 안티앨리어싱 및 결과 생성
        antialiased_image = apply_edge_processing(processed_image, person_mask, kernel_size=5, sigma=1.5)
        final_segmented_image = apply_exact_mask_from_parsing(antialiased_image, person_mask, boundaies_mask)
        overlay_image, pure_mask = visualize_classes_separately(
            image=final_segmented_image[:, :, :3],
            prediction=boundaies_mask,
            mode=self.mode,
            alpha=0.7
        )

        # 세그멘테이션 마스크를 원본 좌표계로 역변환
        # 캔버스 위치 정보
        x, y, pw, ph = transform_info['canvas_placement']
        # 캔버스에서 객체 영역 추출
        canvas_mask = np.zeros_like(boundaies_mask)
        canvas_mask[y:y + ph, x:x + pw] = boundaies_mask[y:y + ph, x:x + pw]

        # 원본 이미지의 바운딩 박스와 크기
        orig_x, orig_y, orig_x2, orig_y2 = transform_info['original_bbox']
        orig_w, orig_h = transform_info['original_size']

        # 원본 크기에 맞는 마스크 생성
        orig_mask = np.zeros((orig_h, orig_w), dtype=boundaies_mask.dtype)

        # 원본 이미지의 객체 영역에 맞춰 마스크 리사이징 및 위치 조정
        obj_w, obj_h = orig_x2 - orig_x + 1, orig_y2 - orig_y + 1
        if pw > 0 and ph > 0:
            # 마스크를 캔버스 위치에서 추출해 원본 객체 크기로 리사이징
            obj_mask = cv2.resize(canvas_mask[y:y + ph, x:x + pw], (obj_w, obj_h), interpolation=cv2.INTER_NEAREST)
            # 원본 위치에 배치
            orig_mask[orig_y:orig_y2 + 1, orig_x:orig_x2 + 1] = obj_mask

        # 원본 이미지에 마스크 적용
        overlay_original, _ = visualize_classes_separately(
            image=original_img,
            prediction=orig_mask,
            mode=self.mode,
            alpha=0.7
        )

        return final_segmented_image, boundaies_mask, overlay_image, pure_mask, overlay_original

    def run_and_save(self, image_path, save_dir, alpha=0.7, max_classes=None):
        """배경 제거 + 세그멘테이션 + 앤티앨리어싱 처리 결과 저장"""
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(image_path))[0]

        processed_image, pred_mask, overlay_image, pure_mask, overlay_original = self.process_and_segment(image_path)

        processed_path = os.path.join(save_dir, f"{file_name}_processed.png")
        overlay_path = os.path.join(save_dir, f"{file_name}_result.png")
        mask_path = os.path.join(save_dir, f"{file_name}_mask.png")
        pure_mask_path = os.path.join(save_dir, f"{file_name}_pure_mask.png")
        original_overlay_path = os.path.join(save_dir, f"{file_name}_original_overlay.png")

        cv2.imwrite(processed_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(pure_mask_path, cv2.cvtColor(pure_mask, cv2.COLOR_RGB2BGR))
        save_mask_only(pred_mask, mask_path, mode=self.mode, max_classes=max_classes)
        cv2.imwrite(original_overlay_path, cv2.cvtColor(overlay_original, cv2.COLOR_RGB2BGR))

        print(f"Saved processed image to {processed_path}")
        print(f"Saved overlay result to {overlay_path}")
        print(f"Saved pure mask to {pure_mask_path}")
        print(f"Saved class mask to {mask_path}")
        print(f"Saved original overlay to {original_overlay_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Demo for Human Parsing with Background Removal')
    parser.add_argument('--image', type=str,
                        default=r"C:\Users\tjdwn\OneDrive\Desktop\test.jpg",
                        help='Path to input image')
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