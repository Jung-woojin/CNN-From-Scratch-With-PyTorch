# CNN-From-Scratch-With-PyTorch

PyTorch로 주요 CNN 아키텍처를 직접 구현하고, 하나의 공통 실행기(`model_test.py`)로 테스트/비교/벤치마크할 수 있게 정리한 저장소입니다.

## 구현된 14개 모델

1. AlexNet - First deep CNN with ReLU, Dropout
2. VGG19 - Thin 3x3 convolutions, 19 layers
3. ResNet50 - Bottleneck Blocks, Skip Connections, Residual Learning
4. Inception (v1) - Multi-Branch, Concat
5. MobileNetV2 - Depthwise Separable Conv, Inverted Bottlenecks
6. Xception - Depthwise Separable Conv, Bottleneck style
7. ResNeXt - Group Convolution, Residual Learning
8. RepVGG - Re-parameterization, Train/Inference structure split
9. GoogLeNet - Multi-Branch, 1x1 reduction, Auxiliary classifiers
10. EfficientNet (B0) - Compound Scaling, MBConv, SE blocks
11. SqueezeNet 1.0 - Fire Module
12. DenseNet-121 style - Dense Connection
13. NASNet-A Mobile style - Cell-based architecture
14. MobileNetV3 Large - h-swish activation, SE blocks

## 파일 구성

- `Alexnet.py`
- `vgg19.py`
- `resnet50.py`
- `inception.py`
- `mobilenet.py`
- `Xception.py`
- `ResNext.py`
- `RepVGG.py`
- `GoogLeNet.py`
- `EfficientNet.py`
- `SqueezeNet.py`
- `DenseNet.py`
- `NASNet.py`
- `MobileNetV3.py`
- `model_test.py` (14개 모델 공통 테스트/비교/벤치마크)

## 설치

```bash
pip install torch torchvision pillow
```

선택:

```bash
pip install torchinfo
```

## model_test.py 사용법

### 단일 모델 실행

```bash
python model_test.py --model efficientnet --image test.jpg --top_k 5
```

### 전체 모델 비교

```bash
python model_test.py --image test.jpg --compare --top_k 3
```

### 전체 모델 벤치마크

```bash
python model_test.py --image test.jpg --benchmark --iterations 20
```

### 디바이스 지정

```bash
python model_test.py --model mobilenetv3 --image test.jpg --device cuda
```

## 사용 가능한 모델 키

```text
alexnet, vgg19, resnet50, inception, mobilenetv2, xception, resnext,
repvgg, googlenet, efficientnet, squeezenet, densenet, nasnet, mobilenetv3
```

## CLI 옵션

- `--model`, `-m`: 실행할 단일 모델 키
- `--image`, `-i`: 입력 이미지 경로 (필수)
- `--top_k`, `-k`: 상위 예측 개수
- `--device`, `-d`: `auto | cpu | cuda`
- `--compare`, `-c`: 14개 모델 비교 실행
- `--benchmark`, `-b`: 14개 모델 벤치마크 실행
- `--iterations`, `-n`: 벤치마크 반복 횟수
- `--num_classes`: 모델 출력 클래스 수 (기본값 `1000`)

## References

1. AlexNet (2012)
2. VGG (2015)
3. ResNet (2015)
4. GoogLeNet / Inception v1 (2015)
5. MobileNetV2 (2018)
6. Xception (2017)
7. ResNeXt (2016)
8. RepVGG (2021)
9. EfficientNet (2019)
10. SqueezeNet (2016)
11. DenseNet (2017)
12. NASNet (2018)
13. MobileNetV3 (2019)
