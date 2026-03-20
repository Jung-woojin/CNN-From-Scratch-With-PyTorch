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
