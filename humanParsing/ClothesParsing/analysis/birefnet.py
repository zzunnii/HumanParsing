import os
import cv2
import json
import numpy as np
import torch
import argparse
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

def load_dataset_stats(stats_file):
    """statistics_summary.json에서 학습셋 통계 로드"""
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Stats file not found: {stats_file}")
    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    print(f"[INFO] Loaded dataset statistics from: {stats_file}")
    return stats

def setup_model(token=None):
    """Hugging Face에서 BiRefNet_HR 모델 로드"""
    print("[INFO] Loading BiRefNet_HR model...")
    model = AutoModelForImageSegmentation.from_pretrained(
        'ZhengPeng7/BiRefNet_HR',
        trust_remote_code=True,
        token=token
    )
    if torch.cuda.is_available():
        model = model.to('cuda').half()
        model.eval()
        print("[INFO] Model loaded on GPU (FP16).")
    else:
        model.eval()
        print("[INFO] Model loaded on CPU.")
    return model

def load_image_with_exif(np_image):
    """NumPy 배열을 PIL 이미지로 변환 후 EXIF 방향 보정, RGB로 반환"""
    pil_image = Image.fromarray(np_image)
    pil_image = ImageOps.exif_transpose(pil_image)  # EXIF 회전정보 처리
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    return np.array(pil_image)

def remove_background(model, np_image, threshold=0.5, image_size=(1024, 1024)):
    """BiRefNet_HR 모델로 배경 제거 -> RGBA 반환"""
    pil_image = Image.fromarray(np_image)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    inp = transform(pil_image).unsqueeze(0)
    if torch.cuda.is_available():
        inp = inp.to('cuda').half()

    with torch.no_grad():
        pred = model(inp)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = torch.sigmoid(pred).cpu().squeeze().numpy()

    orig_h, orig_w = np_image.shape[:2]
    mask = cv2.resize((pred * 255).astype(np.uint8), (orig_w, orig_h))
    bin_mask = (mask > threshold * 255).astype(np.uint8) * 255

    rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    rgba[:, :, :3] = np_image
    rgba[:, :, 3] = bin_mask
    return rgba

def extract_person_bbox(rgba_image):
    """알파>0 영역의 최소 bounding box로 크롭"""
    alpha = rgba_image[:, :, 3]
    rows = np.any(alpha, axis=1)
    cols = np.any(alpha, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None, None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    cropped = rgba_image[y_min:y_max + 1, x_min:x_max + 1]
    return cropped, (x_min, y_min, x_max, y_max)

def scale_person_by_dataset_stats(person_rgba, canvas_size, dataset_stats, extra_scale=1.2):
    """학습셋 평균 크기로 스케일 조정"""
    canvas_w, canvas_h = canvas_size
    ph, pw = person_rgba.shape[:2]
    tgt_h_ratio = dataset_stats['height']['mean']
    tgt_w_ratio = dataset_stats['width']['mean']
    target_h = int(tgt_h_ratio * canvas_h * extra_scale)
    target_w = int(tgt_w_ratio * canvas_w * extra_scale)
    resized = cv2.resize(person_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized

def place_by_dataset_center(canvas_size, person_rgba, dataset_stats, bg_color=(255, 255, 255)):
    """학습셋 평균 중심에 배치"""
    cw, ch = canvas_size
    ph, pw = person_rgba.shape[:2]
    result = np.full((ch, cw, 3), bg_color, dtype=np.uint8)
    cx_mean = dataset_stats['center_x']['mean']
    cy_mean = dataset_stats['center_y']['mean']
    center_x = int(cx_mean * cw)
    center_y = int(cy_mean * ch)
    x = center_x - pw // 2
    y = center_y - ph // 2
    if x < 0: x = 0
    if y < 0: y = 0
    if x + pw > cw: x = cw - pw
    if y + ph > ch: y = ch - ph
    alpha = person_rgba[:, :, 3:4].astype(float) / 255.0
    for c in range(3):
        result[y:y + ph, x:x + pw, c] = (
            result[y:y + ph, x:x + pw, c] * (1 - alpha[:, :, 0]) +
            person_rgba[:, :, c] * alpha[:, :, 0]
        )
    return result


def process_image_for_segmentation(model, np_image, dataset_stats, final_canvas=(720, 1280)):
    """배경 제거 및 크기 조정 후 캔버스에 배치 - 파싱모델 입력용"""
    print("[INFO] Processing image with BiRefNet...")
    original_img = load_image_with_exif(np_image)
    rgba = remove_background(model, original_img, threshold=0.5)
    person_rgba, bbox = extract_person_bbox(rgba)

    if person_rgba is None:
        print("[WARN] No person found, returning blank canvas.")
        blank_image = np.full((final_canvas[1], final_canvas[0], 3), 255, dtype=np.uint8)
        blank_mask = np.zeros((final_canvas[1], final_canvas[0]), dtype=np.uint8)
        return blank_image, blank_mask

    scaled_rgba = scale_person_by_dataset_stats(person_rgba, final_canvas, dataset_stats)

    # 최종 이미지 생성
    final = place_by_dataset_center(final_canvas, scaled_rgba, dataset_stats)

    # 최종 마스크 생성
    final_mask = np.zeros((final_canvas[1], final_canvas[0]), dtype=np.uint8)
    cw, ch = final_canvas
    ph, pw = scaled_rgba.shape[:2]
    cx_mean = dataset_stats['center_x']['mean']
    cy_mean = dataset_stats['center_y']['mean']
    center_x = int(cx_mean * cw)
    center_y = int(cy_mean * ch)
    x = center_x - pw // 2
    y = center_y - ph // 2

    # 경계 검사
    if x < 0: x = 0
    if y < 0: y = 0
    if x + pw > cw: x = cw - pw
    if y + ph > ch: y = ch - ph

    # 마스크 복사
    final_mask[y:y + ph, x:x + pw] = scaled_rgba[:, :, 3]

    return final, final_mask