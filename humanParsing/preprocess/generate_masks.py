import os
import json
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_person_masks(person_info: Dict, config: object) -> Dict:
    """개별 person의 마스크 생성"""
    # person_info에 type이 포함되어 있으므로, 해당 타입별 person 디렉토리를 구성합니다.
    type_ = person_info.get("type", "model")
    split = person_info["split"]
    person_id = person_info["person_id"]
    # 예: config.PROCESSED_DATA["persons_dir"]["model"]["train"] / f"model_person_{person_id}"
    person_dir = Path(config.PROCESSED_DATA[type_][split]) / f"{type_}_person_{person_id}"

    try:
        with open(person_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        results = {
            "person_id": person_id,
            "split": split,
            "type": type_,
            "model_masks": [],
            "item_masks": [],
            "status": "success"
        }

        # Model 마스크 생성 (모델 데이터가 있을 경우)
        model_parse_path = person_dir / "model_parse.json"
        if model_parse_path.exists():
            model_masks = process_model_data(
                person_dir / "model_parse.json",
                person_dir / "masks/model",
                config
            )
            results["model_masks"] = model_masks

        # Item 마스크 생성 (아이템 데이터가 있을 경우)
        item_parse_path = person_dir / "item_parse.json"
        if item_parse_path.exists():
            item_masks = process_item_data(
                person_dir / "item_parse.json",
                person_dir / "masks/item",
                config
            )
            results["item_masks"] = item_masks

        mask_info = {
            "model_masks": results["model_masks"],
            "item_masks": results["item_masks"],
            "process_time": None  # TODO: 필요한 경우 처리 시간 추가
        }
        with open(person_dir / "mask_info.json", "w", encoding="utf-8") as f:
            json.dump(mask_info, f, indent=2)

        return results

    except Exception as e:
        return {
            "person_id": person_id,
            "split": split,
            "type": type_,
            "status": "error",
            "error": str(e)
        }

def process_model_data(parse_path: Path, output_dir: Path, config: object) -> List[Dict]:
    """Model parse data 처리"""
    with open(parse_path, "r", encoding="utf-8") as f:
        parse_data = json.load(f)

    masks_info = []
    img_h, img_w = config.IMAGE_SIZE["height"], config.IMAGE_SIZE["width"]

    category_polygons = defaultdict(list)
    for key, region in parse_data.items():
        if not key.startswith("region"):
            continue
        category_name = region.get("category_name")
        if not category_name or "segmentation" not in region:
            continue
        try:
            for seg in region["segmentation"]:
                points = np.array(seg, dtype=np.int32).reshape((-1, 2))
                category_polygons[category_name].append(points)
        except Exception as e:
            print(f"Error processing segmentation for {category_name}: {e}")
            continue

    for category_name, polygons in category_polygons.items():
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        try:
            cv2.fillPoly(mask, polygons, color=255)
            if np.sum(mask) > 0:
                output_dir.mkdir(parents=True, exist_ok=True)
                mask_path = output_dir / f"{category_name}.png"
                cv2.imwrite(str(mask_path), mask)
                masks_info.append({
                    "category": category_name,
                    "path": str(mask_path),
                    "pixel_count": int(np.sum(mask) / 255)
                })
        except Exception as e:
            print(f"Error creating mask for {category_name}: {e}")
            continue

    return masks_info

def process_item_data(parse_path: Path, output_dir: Path, config: object) -> List[Dict]:
    """Item parse data 처리"""
    with open(parse_path, "r", encoding="utf-8") as f:
        parse_data = json.load(f)

    masks_info = []
    img_h, img_w = config.IMAGE_SIZE["height"], config.IMAGE_SIZE["width"]

    category_polygons = defaultdict(list)
    for key, region in parse_data.items():
        if not key.startswith("region"):
            continue
        product_type = region.get("product_type")
        if not product_type or "segmentation" not in region:
            continue
        try:
            for seg in region["segmentation"]:
                points = np.array(seg, dtype=np.int32).reshape((-1, 2))
                category_polygons[product_type].append(points)
        except Exception as e:
            print(f"Error processing segmentation for {product_type}: {e}")
            continue

    for product_type, polygons in category_polygons.items():
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        try:
            cv2.fillPoly(mask, polygons, color=255)
            if np.sum(mask) > 0:
                output_dir.mkdir(parents=True, exist_ok=True)
                mask_path = output_dir / f"{product_type}.png"
                cv2.imwrite(str(mask_path), mask)
                masks_info.append({
                    "product_type": product_type,
                    "path": str(mask_path),
                    "pixel_count": int(np.sum(mask) / 255)
                })
        except Exception as e:
            print(f"Error creating mask for {product_type}: {e}")
            continue

    return masks_info

def generate_masks(config: object) -> Dict:
    """전체 데이터셋의 마스크 생성 (인덱스 파일 대신 person 디렉토리를 스캔)"""
    results = {"model": {"train": [], "val": [], "test": []},
               "item": {"train": [], "val": [], "test": []}}

    for type_ in ["model", "item"]:
        for split in ["train", "val", "test"]:
            person_root = config.PROCESSED_DATA[type_][split]
            if not os.path.exists(person_root):
                print(f"⚠️ Warning: Person directory not found: {person_root}")
                continue

            person_dirs = [d for d in os.listdir(person_root) if os.path.isdir(os.path.join(person_root, d))]
            print(f"Found {len(person_dirs)} {type_} persons in {split} set")

            tasks = []
            for d in person_dirs:
                # 디렉토리 이름은 예: "model_person_000001" 또는 "item_person_000001"
                parts = d.split('_')
                person_id = parts[-1]  # 단순 추출 (네이밍 규칙을 유지한다고 가정)
                person_info = {"person_id": person_id, "split": split, "type": type_}
                tasks.append(person_info)

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(process_person_masks, person_info, config) for person_info in tasks]
                for future in tqdm(futures, desc=f"Generating {type_} {split} masks"):
                    results[type_][split].append(future.result())

    try:
        output_path = Path(config.PROCESSED_DATA_ROOT) / "mask_generation_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"❌ Error saving results: {e}")

    return results

if __name__ == "__main__":
    from config import Config
    generate_masks(Config)
