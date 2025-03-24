class ModelConfig:
    # 백본 설정
    BACKBONE = "resnet-50"
    BACKBONE_PRETRAINED = True

    # 입력 이미지 설정
    INPUT_SIZE = (720, 1280)

    # 모델 파싱 클래스
    MODEL_CLASSES = 21  # 신체/의류 클래스

    # UPerNet 설정
    UPERNET_CHANNELS = 512
    PPM_SIZES = (1, 2, 3, 6)
    PPM_CHANNELS = 512
    FPN_CHANNELS = 256

    # 디코더 설정
    DECODER_CHANNELS = 256
    DECODER_DROPOUT = 0.1