import argparse
import numpy as np
import cv2
import os

def parse_arguments():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="Visualize segmentation classes separately")
    parser.add_argument(
        "--mode",
        type=str,
        default="tops",
        choices=["tops", "bottoms", "model"],
        help="Visualization mode: tops, bottoms, or model (default: tops)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--prediction",
        type=str,
        required=True,
        help="Path to prediction mask (numpy array)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Transparency level for overlay (default: 0.7)"
    )
    parser.add_argument(
        "--max_classes",
        type=int,
        default=None,
        help="Maximum number of classes to visualize (overrides mode default if set)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save output images (default: outputs)"
    )
    return parser.parse_args()

def visualize_classes_separately(image, prediction, mode="tops", alpha=0.7, max_classes=None):
    """
    각 클래스별로 별도 시각화
    """
    # 전체 클래스와 색상 정의
    ALL_CLASSES = {
        "background": (0, 0, 0),        # 검정
        "rsleeve": (0, 255, 0),         # 초록
        "lsleeve": (0, 128, 255),       # 하늘
        "torsho": (0, 0, 255),          # 파랑
        "top_hidden": (128, 0, 255),    # 보라
        "hip": (255, 255, 0),           # 노랑
        "pants_rsleeve": (255, 0, 255), # 핑크
        "pants_lsleeve": (200, 0, 255), # 자주
        "pants_hidden": (150, 0, 150),  # 연보라
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

    # 모드별 사용 클래스 및 기본 max_classes
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
            "default_max_classes": 20
        }
    }

    # 모드에 따른 설정 적용
    selected_classes = class_configs[mode]["names"]
    default_max = class_configs[mode]["default_max_classes"]
    max_classes = max_classes if max_classes is not None else default_max  # 명령줄 인자가 우선
    selected_classes = selected_classes[:max_classes]  # max_classes에 맞게 자름
    selected_colors = [ALL_CLASSES[class_name] for class_name in selected_classes]

    # 이미지와 예측 마스크 처리
    if isinstance(image, str):
        image = cv2.imread(image)
    if isinstance(prediction, str):
        prediction = np.load(prediction)

    # 결과 이미지 초기화
    output = image.copy()
    overlay = np.zeros_like(image)

    # 예측 마스크를 클래스별로 색상 적용
    for idx, color in enumerate(selected_colors):
        if idx < prediction.max() + 1:  # 예측값 범위 내에서만 처리
            mask = (prediction == idx).astype(np.uint8)
            overlay[mask > 0] = color

    # 원본 이미지와 오버레이 결합
    output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0.0)

    return output

def main():
    # 인자 파싱
    args = parse_arguments()

    # 이미지와 예측 데이터 로드
    image = cv2.imread(args.image)
    prediction = np.load(args.prediction)

    if image is None or prediction is None:
        print("Error: Could not load image or prediction data")
        return

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 시각화 실행
    result = visualize_classes_separately(
        image=image,
        prediction=prediction,
        mode=args.mode,
        alpha=args.alpha,
        max_classes=args.max_classes
    )

    # 결과 저장 경로 설정
    output_path = os.path.join(args.output_dir, f"output_{args.mode}.png")

    # 결과 표시 및 저장
    cv2.imshow(f"Visualization ({args.mode})", result)
    cv2.imwrite(output_path, result)
    print(f"Result saved to: {output_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()