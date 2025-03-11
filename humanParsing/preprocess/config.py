class Config:
    # 원본 데이터 경로
    RAW_DATA = {
        "train": {
            "model": {
                "parse": r"C:\Users\tjdwn\OneDrive\Desktop\parsingData\data\1.Training\label\Model-Parse",
                "image": r"C:\Users\tjdwn\OneDrive\Desktop\parsingData\data\1.Training\original\Model-Image"
            },
            "item": {
                "parse": r"C:\Users\tjdwn\OneDrive\Desktop\parsingData\data\1.Training\label\Item-Parse",
                "image": r"C:\Users\tjdwn\OneDrive\Desktop\parsingData\data\1.Training\original\Item-Image"
            }
        },
        "val": {
            "model": {
                "parse": r"C:\Users\tjdwn\OneDrive\Desktop\parsingData\data\2.Validation\label\Model-Parse",
                "image": r"C:\Users\tjdwn\OneDrive\Desktop\parsingData\data\2.Validation\original\Model-Image"
            },
            "item": {
                "parse": r"C:\Users\tjdwn\OneDrive\Desktop\parsingData\data\2.Validation\label\Item-Parse",
                "image": r"C:\Users\tjdwn\OneDrive\Desktop\parsingData\data\2.Validation\original\Item-Image"
            }
        }
    }

    # 전처리된 데이터 저장 경로
    PROCESSED_DATA_ROOT = r"C:\Users\tjdwn\OneDrive\Desktop\parsingData\preprocessed"

    # 전처리된 데이터 구조
    PROCESSED_DATA = {
        "model": {
            "train": f"{PROCESSED_DATA_ROOT}/model/train",
            "val": f"{PROCESSED_DATA_ROOT}/model/val",
            "test": f"{PROCESSED_DATA_ROOT}/model/test"
        },
        "item": {
            "train": f"{PROCESSED_DATA_ROOT}/item/train",
            "val": f"{PROCESSED_DATA_ROOT}/item/val",
            "test": f"{PROCESSED_DATA_ROOT}/item/test"
        },
        "indexes": {
            "train": f"{PROCESSED_DATA_ROOT}/indexes/train_index.json",
            "val": f"{PROCESSED_DATA_ROOT}/indexes/val_index.json",
            "test": f"{PROCESSED_DATA_ROOT}/indexes/test_index.json"
        }
    }

    # 카테고리 정보
    MODEL_CATEGORIES = {
        0: "hair",
        1: "face",
        2: "neck",
        3: "hat",
        4: "outer_rsleeve",
        5: "outer_lsleeve",
        6: "outer_torso",
        7: "inner_rsleeve",
        8: "inner_lsleeve",
        9: "inner_torso",
        10: "pants_hip",
        11: "pants_rsleeve",
        12: "pants_lsleeve",
        13: "skirt",
        14: "right_arm",
        15: "left_arm",
        16: "right_shoe",
        17: "left_shoe",
        18: "right_leg",
        19: "left_leg"
    }

    ITEM_CATEGORIES = {
        "main_categories": {
            0: "cap_and_hat",
            1: "outerwear",
            2: "tops",
            3: "bottoms",
            4: "shoes"
        },
        "product_types": {
            0: "hat",
            1: "hat_hidden",
            2: "rsleeve",
            3: "lsleeve",
            4: "torso",
            5: "top_hidden",
            6: "hip",
            7: "pants_rsleeve",
            8: "pants_lsleeve",
            9: "pants_hidden",
            10: "skirt",
            11: "skirt_hidden",
            12: "shoe",
            13: "shoe_hidden"
        }
    }

    # 데이터 분할 비율
    SPLIT_RATIO = {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1
    }

    # 이미지 크기
    IMAGE_SIZE = {
        "height": 1280,
        "width": 720
    }

    @staticmethod
    def get_person_path(split: str, person_id: str) -> str:
        """person 디렉토리 경로 반환"""
        return f"{Config.PROCESSED_DATA['persons_dir'][split]}/person_{person_id}"