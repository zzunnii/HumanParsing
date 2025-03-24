# HumanParsing
ResNet101_Weights.IMAGENET1K_V1을 파인튜닝하여 인체 및 의류 영역을 정밀하게 분할하는 Segmentation 모델. 의류 Try-On, AR, VR 등의 응용을 위해 개발됨.
현재 14에포크에서 가장 우수한 성능을 보이지만 새로운 증강을 도입하여 학습 진행중

# Class
## Model(20), Tops(5), Bottoms(7)

# Performance
### model
#### Test mIoU - 0.83
![모델 테스트 마스크](/assets/model_test2_mask.png)
![모델 테스트 처리됨](/assets/model_test2_processed.png)
![모델 테스트 순수 마스크](/assets/model_test2_pure_mask.png)
![모델 테스트 결과](/assets/model_test2_result.png)

# modified 2025.03.12
