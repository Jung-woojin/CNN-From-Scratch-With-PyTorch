# CNN-From-Scratch-With-PyTorch
## PyTorch 로 CNN 직접 구현

PyTorch 로 CNN 아키텍처를 직접 설계·재구현하며 커스터마이징까지 연습하는 저장소

## What I did
1. **AlexNet** - First deep CNN with ReLU, Dropout
2. **VGG19** - Thin 3x3 convolutions, 19 layers
3. **ResNet50** - Bottleneck Blocks, Skip Connections, Residual Learning
4. **Inception (v1)** - Multi-Branch, Concat, GoogLeNet architecture
5. **MobileNetV2** - Depthwise Separable Conv, Inverted (Residual) Bottlenecks
6. **Xception** - Depthwise Separable Conv, Multi-Branch, Bottlenecks
7. **ResNeXt** - Group Convolution, Projection, Residual Learning
8. **RepVGG** - Reparameterization, Train/Inference structure separation
9. **GoogLeNet** - Multi-Branch, 1x1 dimensionality reduction, Auxiliary classifiers
10. **EfficientNet (B0)** - Compound Scaling, MBConv, SE blocks, SOTA accuracy
11. **SqueezeNet 1.0** - Fire Module, AlexNet-level accuracy with 50x fewer parameters
12. **DenseNet-121** - Dense Connection, Every layer connected to all subsequent layers
13. **NASNet-A Mobile** - Neural Architecture Search, Reusable building blocks
14. **MobileNetV3 Large** - h-swish activation, SE blocks, Mobile optimized

## To Do List
1. ✅ Inception v1 구현 완료
2. ✅ Compound Scaling 구현 완료 (EfficientNet)
3. ✅ 초경량 CNN 구현 완료 (SqueezeNet)
4. ✅ Dense Connection 구현 완료 (DenseNet)
5. ✅ NAS 구현 완료 (NASNet)
6. ✅ Mobile 최적화 구현 완료 (MobileNetV3)
7. 데이터 로더와 훈련 파라미터 설정하는 코드에 대한 학습 필요
8. 훈련 결과 확인 및 분석을 위한 코드 학습 필요
9. 모델 구조 구현 코드 공부 (자속)
10. 반복 횟수, 스트라이드 별 프로젝션을 하나의 구조 안에서 사용할 수 있도록 인자 사용 능력 키우기

## CNN 아키텍처 비교

| 아키텍처 | 연도 | 주요 특징 | 파라미터 |
|----------|------|-----------|----------|
| AlexNet | 2012 | ReLU, Dropout, Overlapping Pooling | ~60M |
| VGG19 | 2014 | 얇은 3x3 컨볼루션 | ~143M |
| ResNet50 | 2015 | Bottleneck, Skip Connections | ~25.6M |
| Inception | 2014 | Multi-Branch, 1x1 conv | ~5.8M |
| MobileNetV2 | 2018 | Depthwise Separable, Inverted Residual | ~3.5M |
| Xception | 2017 | Depthwise Separable, Extends Inception | ~23M |
| ResNeXt | 2016 | Group Convolution, Cardinality | ~30M |
| RepVGG | 2021 | Reparameterization, Training-time structure | ~21M |
| GoogLeNet | 2014 | Auxiliary Classifiers, Multi-Scale | ~5.8M |
| EfficientNet | 2019 | Compound Scaling, MBConv | ~5.3M |
| SqueezeNet | 2016 | Fire Module, 1x1 dominant | ~1.2M |
| DenseNet | 2017 | Dense Connection, Feature Reuse | ~8M |
| NASNet | 2018 | Neural Architecture Search | ~5.4M |
| MobileNetV3 | 2019 | h-swish, SE blocks | ~5.4M |

## 핵심 기술

### MBConv (Mobile Inverted Bottleneck Convolution)
- **Expansion**: 1x1 Conv for channel expansion
- **Depthwise**: 3x3 Depthwise convolution for spatial filtering
- **Projection**: 1x1 Conv for channel reduction
- **SE (Squeeze-and-Excitation)**: Channel-wise attention mechanism

### SqueezeNet (Fire Module)
- **Squeeze**: 1x1 conv to reduce channels
- **Expand**: Parallel 1x1 and 3x3 convolutions
- **Concat**: Combine results
- **Result**: AlexNet accuracy with 50x fewer parameters

### DenseNet (Dense Connection)
- **Feature Reuse**: Each layer receives feature maps from all previous layers
- **Gradient Flow**: Direct gradient paths to all layers
- **Efficiency**: Fewer parameters than ResNet
- **Structure**: Dense blocks → Transition layers → Dense blocks

### Compound Scaling (EfficientNet)
- **Balance**: Simultaneously scale width, depth, resolution
- **Parameters**: α (width), β (depth), δ (resolution), γ (depth)
- **Efficient**: Better accuracy with fewer parameters

## GitHub Links
- **Repository**: https://github.com/Jung-woojin/CNN-From-Scratch-With-PyTorch
- **Main Branch**: main
- **License**: MIT (implied)

## 🧪 모델 테스트 및 비교 (model_test.py)

파편화된 개별 파일들을 통합한 테스트 프레임워크를 제공합니다.

### 설치 의존성

```bash
pip install torch torchvision pillow
```

### 주요 기능

1. **단일 모델 테스트**: 특정 아키텍처에 이미지 분류 실행
2. **모델 비교**: 모든 아키텍처의 정확도 및 추론 시간 비교
3. **벤치마킹**: 모델별 파라미터 수, 추론 시간, 처리량 분석

### 사용 방법

#### 🔍 단일 모델 테스트

```bash
# 기본 사용 (EfficientNet)
python model_test.py --image test.jpg

# 특정 모델 테스트
python model_test.py --model efficientnet --image test.jpg
python model_test.py --model densenet --image test.jpg --device cuda

# 상위 k개 결과 표시
python model_test.py --model mobilenetv3 --image test.jpg --top_k 10
```

#### ⚖️ 모델 비교

```bash
# 모든 모델 비교
python model_test.py --image test.jpg --compare

# 상위 3개 결과 표시
python model_test.py --image test.jpg --compare --top_k 3
```

**출력 예시:**
```
CNN ARCHITECTURE COMPARISON
================================================================================
Image: test.jpg

Model          Input Size   Top 1           Time (ms)      
-----------------------------------------------------------------
alexnet        224          68.25%          12.34          
densenet       224          75.12%          45.67          
efficientnet   224          78.43%          23.89          
mobilenetv3    224          72.56%          8.92           
vgg19          224          65.89%          89.23          
================================================================================

BEST PERFORMING MODELS
================================================================================
1. efficientnet: 78.43% accuracy
2. densenet: 75.12% accuracy
3. mobilenetv3: 72.56% accuracy
================================================================================
```

#### ⚡ 벤치마킹

```bash
# 모든 모델 벤치마킹 (10 회 반복 평균)
python model_test.py --image test.jpg --benchmark

# 사용자 정의 반복 횟수
python model_test.py --image test.jpg --benchmark --iterations 50
```

**출력 예시:**
```
ARCHITECTURE BENCHMARKING
================================================================================
Image: test.jpg

Model          Params (M)    Avg Time (ms)   Throughput (fps)
-----------------------------------------------------------------
alexnet        57.84         12.34           81.04          
densenet       8.04          45.67           21.90          
efficientnet   5.29          23.89           41.86          
mobilenetv3    5.40          8.92            112.11         
vgg19          134.31        89.23           11.21          
================================================================================
```

### 사용 가능한 모델

```
alexnet, vgg19, resnet50, inception, mobilenetv2,
xception, resnext, repvgg, googlenet, efficientnet,
squeezenet, densenet, nasnet, mobilenetv3
```

### 코드 사용 예시 (Python)

```python
from model_test import CNNTester

# 초기화
tester = CNNTester(device='cuda')

# 단일 모델 분류
model = tester.get_model('efficientnet')
results = tester.classify(model, 'test.jpg', top_k=5)
tester.print_results(results)

# 모든 모델 비교
results = tester.compare_architectures('test.jpg', top_k=3)

# 벤치마킹
tester.benchmark('test.jpg', iterations=20)
```

### 명령어 옵션

| 옵션 | 단축 | 설명 | 기본값 |
|------|------|------|--------|
| `--model` | `-m` | 테스트할 모델 이름 | `efficientnet` |
| `--image` | `-i` | 입력 이미지 경로 | *필수* |
| `--top_k` | `-k` | 상위 k개 결과 표시 | `5` |
| `--device` | `-d` | 사용 장치 (cpu/cuda) | `auto` |
| `--compare` | `-c` | 모든 모델 비교 | `False` |
| `--benchmark` | `-b` | 벤치마킹 실행 | `False` |
| `--iterations` | `-n` | 벤치마킹 반복 횟수 | `10` |

### 실행 스크립트

```bash
#!/bin/bash
# compare_all.sh - 모든 모델 비교 스크립트

IMAGE="${1:-test.jpg}"
DEVICE="${2:-auto}"

echo "Running CNN Architecture Comparison on $IMAGE..."
python model_test.py --image "$IMAGE" --compare --device "$DEVICE"
```

## Usage Example

```python
import torch
from EfficientNet import create_efficientnet_b0
from SqueezeNet import create_squeezenet
from DenseNet import densenet121
from NASNet import create_nasnet_mobile
from MobileNetV3 import create_mobilenetv3_large

# Create model
model = create_efficientnet_b0(num_classes=1000)

# Test
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"Output: {output.shape}")
```

## References

1. **AlexNet**: Krizhevsky et al. (2012) - ImageNet Classification with Deep CNN
2. **VGG**: Simonyan & Zisserman (2015) - Very Deep CNNs for Large-Scale Recognition
3. **ResNet**: He et al. (2015) - Deep Residual Learning for Image Recognition
4. **GoogLeNet**: Szegedy et al. (2015) - Going Deeper with Convolutions
5. **MobileNetV2**: Sandler et al. (2018) - MobileNetV2: Inverted Residuals and Linear Bottlenecks
6. **Xception**: Chollet (2017) - Xception: Deep Learning with Depthwise Separable Convolutions
7. **ResNeXt**: Xie et al. (2016) - Aggregated Residual Transformations for Deep Neural Networks
8. **RepVGG**: Chen et al. (2021) - Rethinking the Scale in Structure-Pruning
9. **EfficientNet**: Tan & Le (2019) - EfficientNet: Rethinking Model Scaling for CNNs
10. **SqueezeNet**: Iandola et al. (2016) - SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
11. **DenseNet**: Huang et al. (2017) - Densely Connected Convolutional Networks
12. **NASNet**: Zoph et al. (2018) - Learning Transferable Architectures from Scratch
13. **MobileNetV3**: Howard et al. (2019) - Searching for MobileNetV3

---

_PyTorch 로 직접 구현하며 CNN 의 핵심 아이디어를 정복합니다!_ 🚀
