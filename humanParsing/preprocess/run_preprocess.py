import os
import time
from datetime import datetime
from pathlib import Path
import shutil

from config import Config
from raw_to_person import convert_to_person_structure
from generate_masks import generate_masks


def create_directory_structure(config: Config):
    """필요한 디렉토리 구조 생성"""
    directories = [
        # Person 디렉토리 (모델/아이템 각각의 결과가 저장될 경로)
        config.PROCESSED_DATA["model"]["train"],
        config.PROCESSED_DATA["model"]["val"],
        config.PROCESSED_DATA["model"]["test"],
        config.PROCESSED_DATA["item"]["train"],
        config.PROCESSED_DATA["item"]["val"],
        config.PROCESSED_DATA["item"]["test"],
        # 인덱스 디렉토리 (각 split 별)
        os.path.dirname(config.PROCESSED_DATA["indexes"]["train"]),
        os.path.dirname(config.PROCESSED_DATA["indexes"]["val"]),
        os.path.dirname(config.PROCESSED_DATA["indexes"]["test"])
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"디렉토리 생성 완료: {directory}")


def backup_existing_data(config: Config):
    """기존 전처리 데이터 백업"""
    if os.path.exists(config.PROCESSED_DATA_ROOT):
        backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{config.PROCESSED_DATA_ROOT}_backup_{backup_time}"
        shutil.move(config.PROCESSED_DATA_ROOT, backup_dir)
        print(f"기존 데이터 백업 완료: {backup_dir}")


def main():
    """전처리 파이프라인 실행"""
    start_time = time.time()
    print(f"\n전처리 파이프라인 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        # 1. 기존 데이터 백업
        print("\n1기존 데이터 백업 중...")
        backup_existing_data(Config)

        # 2. 디렉토리 구조 생성
        print("\n디렉토리 구조 생성 중...")
        create_directory_structure(Config)

        # 3. Raw Data를 Person 구조로 변환 (모델과 아이템을 개별 작업으로 처리)
        print("\nRaw Data를 Person 구조로 변환 중...")
        person_results = convert_to_person_structure(Config)
        # person_results: { "model": { "train": [...], "val": [...], "test": [...] },
        #                   "item": { "train": [...], "val": [...], "test": [...] } }

        # 4. 마스크 생성 (모델/아이템 별로 처리)
        print("\n마스크 생성 중...")
        mask_results = generate_masks(Config)
        # mask_results 또한 { "model": { "train": [...], ... }, "item": { "train": [...], ... } } 형태로 반환

        # 처리 시간 계산
        end_time = time.time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)

        print("\n전처리 파이프라인 완료!")
        print(f"총 처리 시간: {hours}시간 {minutes}분 {seconds}초")
        print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 결과 요약
        print("\n처리 결과 요약:")

        # 모델 데이터 결과 요약
        print("\n[MODEL 데이터]")
        for split in ["train", "val", "test"]:
            model_person = person_results["model"].get(split, [])
            model_mask = mask_results.get("model", {}).get(split, [])
            total = len(model_person)
            mask_total = len(model_mask)
            success = sum(1 for r in model_person if r.get("status") == "success")
            mask_success = sum(1 for r in model_mask if r.get("status") == "success")
            percentage = (success / total * 100) if total > 0 else 0.0
            mask_percentage = (mask_success / mask_total * 100) if mask_total > 0 else 0.0
            print(f"\n{split.upper()} 세트 (MODEL):")
            print(f"  - Person 변환: {success}/{total} ({percentage:.2f}%)")
            print(f"  - 마스크 생성: {mask_success}/{mask_total} ({mask_percentage:.2f}%)")

        # 아이템 데이터 결과 요약
        print("\n[ITEM 데이터]")
        for split in ["train", "val", "test"]:
            item_person = person_results["item"].get(split, [])
            item_mask = mask_results.get("item", {}).get(split, [])
            total = len(item_person)
            mask_total = len(item_mask)
            success = sum(1 for r in item_person if r.get("status") == "success")
            mask_success = sum(1 for r in item_mask if r.get("status") == "success")
            percentage = (success / total * 100) if total > 0 else 0.0
            mask_percentage = (mask_success / mask_total * 100) if mask_total > 0 else 0.0
            print(f"\n{split.upper()} 세트 (ITEM):")
            print(f"  - Person 변환: {success}/{total} ({percentage:.2f}%)")
            print(f"  - 마스크 생성: {mask_success}/{mask_total} ({mask_percentage:.2f}%)")

        print(f"\n처리된 데이터 저장 위치: {Config.PROCESSED_DATA_ROOT}")

    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
