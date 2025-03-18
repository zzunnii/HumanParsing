class TrainConfig:
    # 학습 기본 설정
    EPOCHS = 100
    BATCH_SIZE = 16
    NUM_WORKERS = 2

    # 옵티마이저 설정
    OPTIMIZER = "AdamW"
    BASE_LR = 1e-4
    BACKBONE_LR = 1e-5
    WEIGHT_DECAY = 0.01

    # Gradient Accumulation 설정
    GRADIENT_ACCUMULATION_STEPS = 8  # 새로 추가

    # 스케줄러 설정
    SCHEDULER = "CosineAnnealingLR"
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 5

    # 손실 함수 가중치
    CROSS_ENTROPY_WEIGHT = 1.0
    DICE_WEIGHT = 1.0

    # 학습 최적화
    MIXED_PRECISION = True
    GRADIENT_CLIPPING = 1.0

    # 체크포인트 설정
    SAVE_FREQ = 5  # 에폭 단위
    SAVE_BEST = True
    EARLY_STOPPING_PATIENCE = 5

    # 로깅 설정
    LOG_FREQ = 100  # 배치 단위
    VAL_FREQ = 1  # 에폭 단위

    # 경로 설정
    CHECKPOINT_DIR = "logs/checkpoints"
    TENSORBOARD_DIR = "logs/tensorboard"

    # 데이터 증강
    AUGMENTATION = {
        'random_flip': True,
        'random_crop': True,
        'random_rotate': True,
        'random_brightness': True,
        'random_contrast': True,
        'random_saturation': True
    }