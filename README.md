# HumanParsing
ResNet101_Weights.IMAGENET1K_V1을 파인튜닝하여 인체 및 의류 영역을 정밀하게 분할하는 Segmentation 모델. 의류 Try-On, AR, VR 등의 응용을 위해 개발됨.
현재 14에포크에서 가장 우수한 성능을 보이지만 새로운 증강을 도입하여 학습 진행중

배경제거와 후처리 작업에는 ZhengPeng7/BiRefNet_HR 모델을 사용함.

# Class
## Model(20), Tops(5), Bottoms(7)
# Performance
### model
#### Test mIoU - 0.83
<table>
 <tr>
   <td><img src="/assets/model_test2.jpg" alt="모델 원본 이미지" width="100%"></td>
   <td><img src="/assets/model_test2_processed.png" alt="모델 배경제거" width="100%"></td>
 </tr>
 <tr>
   <td><img src="/assets/model_test2_mask.png" alt="모델 테스트 마스크" width="100%"></td>
   <td><img src="/assets/model_test2_pure_mask.png" alt="모델 테스트 순수 마스크" width="100%"></td>
 </tr>
 <tr>
   <td colspan="2"><img src="/assets/model_test2_result.png" alt="모델 테스트 결과" width="50%"></td>
 </tr>
</table>
# modified 2025.03.24
