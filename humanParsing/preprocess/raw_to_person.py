import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor

def load_and_validate_json(file_path: str) -> Optional[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

###############################
# 모델 데이터 처리 (별도 작업)
###############################
def collect_model_files(raw_split: str, config: object) -> List[Dict]:
    """
    raw_split 폴더(예: "train" 또는 "val")에서 모델 데이터를 수집합니다.
    JSON 파일명에서 .json을 제거하고 .jpg로 치환하여 이미지 파일을 찾습니다.
    """
    model_files = []
    parse_dir = config.RAW_DATA[raw_split]["model"]["parse"]
    image_dir = config.RAW_DATA[raw_split]["model"]["image"]

    print(f"\n수집 중: {raw_split} 모델 데이터")
    parse_files = [f for f in os.listdir(parse_dir) if f.endswith('.json')]
    print(f"발견된 모델 Parse 파일: {len(parse_files)}")

    for parse_file in tqdm(sorted(parse_files), desc="모델 파일 검증"):
        parse_path = os.path.join(parse_dir, parse_file)
        data = load_and_validate_json(parse_path)
        if not data:
            continue

        base_name = os.path.splitext(parse_file)[0]
        image_name = base_name + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            continue

        model_files.append({
            "parse_file": parse_path,
            "image_file": image_path,
            "data": data,
            "base_name": base_name
        })
    print(f"유효한 모델 파일: {len(model_files)}")
    return model_files

def process_model_file(task: Dict) -> Dict:
    """개별 모델 파일을 person 구조로 변환합니다."""
    try:
        person_id = task["person_id"]
        model_info = task["model_info"]
        target_dir = task["target_dir"]
        split = task["split"]

        # 모델 데이터용 person 디렉토리 생성 (예: rawdata/model/train/model_person_000001)
        person_dir = os.path.join(target_dir, f"model_person_{person_id}")
        os.makedirs(person_dir, exist_ok=True)

        shutil.copy2(model_info["image_file"], os.path.join(person_dir, "model_image.jpg"))
        with open(os.path.join(person_dir, "model_parse.json"), 'w', encoding='utf-8') as f:
            json.dump(model_info["data"], f, indent=2)

        metadata = {
            "person_id": person_id,
            "split": split,
            "model_file_name": os.path.basename(model_info["image_file"])
        }
        with open(os.path.join(person_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        return {"person_id": person_id, "split": split, "status": "success"}
    except Exception as e:
        return {"person_id": task.get("person_id", "unknown"),
                "split": task.get("split", ""),
                "status": "error", "error": str(e)}

def convert_model_structure(config: object) -> Dict:
    """
    모델 데이터를 raw 데이터("train"와 "val")에서 모두 수집한 후,
    전체를 섞어 8:1:1로 분할하여 person 구조로 변환합니다.
    """
    all_persons = []
    person_id_counter = 0
    for raw_split in ["train", "val"]:
        model_files = collect_model_files(raw_split, config)
        for model_info in model_files:
            all_persons.append({
                "person_id": f"{person_id_counter:06d}",
                "model_info": model_info
            })
            person_id_counter += 1

    random.shuffle(all_persons)
    N = len(all_persons)
    train_count = int(N * 0.8)
    val_count = int(N * 0.1)
    test_count = N - train_count - val_count

    train_persons = all_persons[:train_count]
    val_persons = all_persons[train_count:train_count+val_count]
    test_persons = all_persons[train_count+val_count:]

    results = {"train": [], "val": [], "test": []}
    target_dir_train = config.PROCESSED_DATA["model"]["train"]
    target_dir_val   = config.PROCESSED_DATA["model"]["val"]
    target_dir_test  = config.PROCESSED_DATA["model"]["test"]
    os.makedirs(target_dir_train, exist_ok=True)
    os.makedirs(target_dir_val, exist_ok=True)
    os.makedirs(target_dir_test, exist_ok=True)

    tasks = []
    for person in train_persons:
        tasks.append({
            "person_id": person["person_id"],
            "model_info": person["model_info"],
            "target_dir": target_dir_train,
            "split": "train"
        })
    for person in val_persons:
        tasks.append({
            "person_id": person["person_id"],
            "model_info": person["model_info"],
            "target_dir": target_dir_val,
            "split": "val"
        })
    for person in test_persons:
        tasks.append({
            "person_id": person["person_id"],
            "model_info": person["model_info"],
            "target_dir": target_dir_test,
            "split": "test"
        })

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_model_file, task) for task in tasks]
        for future in tqdm(futures, desc="모델 데이터 처리"):
            res = future.result()
            results[res["split"]].append(res)
    return results

###############################
# 아이템 데이터 처리 (별도 작업)
###############################
def collect_item_files(raw_split: str, config: object) -> List[Dict]:
    """
    아이템 데이터를 raw 데이터("train"와 "val")에서 모두 수집합니다.
    아이템은 JSON의 file_name 필드를 사용하여 이미지 파일을 찾습니다.
    """
    item_files = []
    parse_dir = config.RAW_DATA[raw_split]["item"]["parse"]
    image_dir = config.RAW_DATA[raw_split]["item"]["image"]

    print(f"\n수집 중: {raw_split} 아이템 데이터")
    parse_files = [f for f in os.listdir(parse_dir) if f.endswith('.json')]
    print(f"발견된 아이템 Parse 파일: {len(parse_files)}")

    for parse_file in tqdm(sorted(parse_files), desc="아이템 파일 검증"):
        parse_path = os.path.join(parse_dir, parse_file)
        data = load_and_validate_json(parse_path)
        if not data:
            continue

        file_name = data.get("file_name")
        if not file_name:
            file_name = os.path.splitext(parse_file)[0] + ".jpg"
        image_path = os.path.join(image_dir, file_name)
        if not os.path.exists(image_path):
            continue

        item_files.append({
            "parse_file": parse_path,
            "image_file": image_path,
            "data": data,
            "base_name": os.path.splitext(parse_file)[0]
        })
    print(f"유효한 아이템 파일: {len(item_files)}")
    return item_files

def process_item_file(task: Dict) -> Dict:
    """개별 아이템 파일을 person 구조로 변환합니다."""
    try:
        person_id = task["person_id"]
        item_info = task["item_info"]
        target_dir = task["target_dir"]
        split = task["split"]

        person_dir = os.path.join(target_dir, f"item_person_{person_id}")
        os.makedirs(person_dir, exist_ok=True)

        shutil.copy2(item_info["image_file"], os.path.join(person_dir, "item_image.jpg"))
        with open(os.path.join(person_dir, "item_parse.json"), 'w', encoding='utf-8') as f:
            json.dump(item_info["data"], f, indent=2)

        metadata = {
            "person_id": person_id,
            "split": split,
            "item_file_name": os.path.basename(item_info["image_file"])
        }
        with open(os.path.join(person_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        return {"person_id": person_id, "split": split, "status": "success"}
    except Exception as e:
        return {"person_id": task.get("person_id", "unknown"),
                "split": task.get("split", ""),
                "status": "error", "error": str(e)}

def convert_item_structure(config: object) -> Dict:
    """
    아이템 데이터를 raw 데이터("train"와 "val")에서 모두 수집한 후,
    전체를 섞어 8:1:1로 분할하여 person 구조로 변환합니다.
    """
    all_persons = []
    person_id_counter = 0
    for raw_split in ["train", "val"]:
        item_files = collect_item_files(raw_split, config)
        for item_info in item_files:
            all_persons.append({
                "person_id": f"{person_id_counter:06d}",
                "item_info": item_info
            })
            person_id_counter += 1

    random.shuffle(all_persons)
    N = len(all_persons)
    train_count = int(N * 0.8)
    val_count = int(N * 0.1)
    test_count = N - train_count - val_count

    train_persons = all_persons[:train_count]
    val_persons = all_persons[train_count:train_count+val_count]
    test_persons = all_persons[train_count+val_count:]

    results = {"train": [], "val": [], "test": []}
    target_dir_train = config.PROCESSED_DATA["item"]["train"]
    target_dir_val   = config.PROCESSED_DATA["item"]["val"]
    target_dir_test  = config.PROCESSED_DATA["item"]["test"]
    os.makedirs(target_dir_train, exist_ok=True)
    os.makedirs(target_dir_val, exist_ok=True)
    os.makedirs(target_dir_test, exist_ok=True)

    tasks = []
    for person in train_persons:
        tasks.append({
            "person_id": person["person_id"],
            "item_info": person["item_info"],
            "target_dir": target_dir_train,
            "split": "train"
        })
    for person in val_persons:
        tasks.append({
            "person_id": person["person_id"],
            "item_info": person["item_info"],
            "target_dir": target_dir_val,
            "split": "val"
        })
    for person in test_persons:
        tasks.append({
            "person_id": person["person_id"],
            "item_info": person["item_info"],
            "target_dir": target_dir_test,
            "split": "test"
        })

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_item_file, task) for task in tasks]
        for future in tqdm(futures, desc="아이템 데이터 처리"):
            res = future.result()
            results[res["split"]].append(res)
    return results

def convert_to_person_structure(config: object) -> Dict:
    """
    모델과 아이템 데이터를 각각 개별 작업으로 처리하여
    별도의 person 구조로 변환하고, 8:1:1로 분할합니다.
    """
    model_results = convert_model_structure(config)
    item_results = convert_item_structure(config)
    return {"model": model_results, "item": item_results}

if __name__ == "__main__":
    from config import Config
    convert_to_person_structure(Config)
