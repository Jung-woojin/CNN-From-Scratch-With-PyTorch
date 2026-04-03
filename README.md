# CNN-From-Scratch-With-PyTorch
## PyTorch 로 CNN 직접 구현 - **백본 네트워크 직접 구현 연습**

**작업자**: Jung-woojin  
**이메일**: wojin010629@gmail.com  
**최종 업데이트**: 2026-04-03

---

### 🎯 레포지토리 목적

**PyTorch 로 CNN 백본 네트워크를 직접 구현하며 이해를 극대화합니다.**

이 레포지토리는 각 CNN 아키텍처의 핵심 아이디어를 직접 코드로 구현하며, 백본 네트워크 설계의 의도와 세부사항을 깊이 있게 이해하는 것을 목표로 합니다.

> **핵심**: 이론을 공부하는 것을 넘어, 직접 구현하며 완전히 이해합니다.

## What I did
1. AlexNet 
2. VGG19
3. Resnet50 - Bottleneck Blocks, Skip Connections
4. Inception - Multi Branch, Concat
5. Mobilenetv2 Depthwise Separable Conv, Inverted (Residual) Bottlenecks
6. Xception - Multi Branchs, Bottlenecks, Depthwise Sepable Conv
7. ResNext - Group Conv, Projection, Residual
8. RepVGG - Reparameterization  
9. **GoogLeNet** - Multi-Branch, 1x1 dimensionality reduction, Auxiliary classifiers ✅

## 📋 구현 완료된 아키텍처

### 기본 아키텍처
1. ✅ **AlexNet** - ReLU, Dropout, Overlapping Pooling
2. ✅ **VGG19** - Thin 3×3 convolutions, 19 layers
3. ✅ **GoogLeNet** - Multi-Branch, 1×1 dimensionality reduction, Auxiliary classifiers

### Residual 기반
4. ✅ **ResNet50** - Bottleneck Blocks, Skip Connections, Residual Learning
5. ✅ **ResNeXt** - Group Convolution, Projection, Residual Learning

### 효율성 최적화
6. ✅ **MobileNetV2** - Depthwise Separable Conv, Inverted (Residual) Bottlenecks
7. ✅ **Xception** - Multi-Branches, Bottlenecks, Depthwise Separable Conv
8. ✅ **RepVGG** - Reparameterization

### 혁신적 설계
9. ✅ **Inception v1** - Multi-Branch, Concat, GoogLeNet architecture
10. ✅ **MobileNetV3** - h-swish activation, SE blocks, Mobile optimized

### 초경량 & 초첨단
11. ✅ **SqueezeNet 1.0** - Fire Module, AlexNet-level accuracy with 50x fewer parameters
12. ✅ **DenseNet-121** - Dense Connection, Every layer connected to all subsequent layers
13. ✅ **NASNet-A Mobile** - Neural Architecture Search, Reusable building blocks
14. ✅ **EfficientNet-B0** - Compound Scaling, MBConv, SE blocks

## 🔥 핵심 구현 기술

### MBConv (MobileNetV2/V3, EfficientNet)
- **Expansion**: 1×1 Conv for channel expansion
- **Depthwise**: 3×3 Depthwise convolution for spatial filtering
- **Projection**: 1×1 Conv for channel reduction
- **SE (Squeeze-and-Excitation)**: Channel-wise attention mechanism

### SqueezeNet (Fire Module)
- **Squeeze**: 1×1 conv to reduce channels
- **Expand**: Parallel 1×1 and 3×3 convolutions
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

## 🎓 학습 목표

### 직접 구현하며 배우는 것

1. **아키텍처 설계의 미묘한 차이 이해**
   - ResNet 의 Skip Connection 의 역할
   - Inception 의 Multi-scale 병렬 처리
   - Depthwise Separable Conv 의 효율성

2. **PyTorch 고급 기능 활용**
   - `nn.Module` 조합 및 상속
   - forward 메서드 설계
   - Custom Layer 구현

3. **실제 구현의 어려움**
   - BatchNorm placement
   - Activation function choice
   - Initialization strategies

## 📊 CNN 아키텍처 비교

| 아키텍처 | 연도 | 주요 특징 | 파라미터 |
|--------------|------|-----------|---------|
| AlexNet | 2012 | ReLU, Dropout, Overlapping Pooling | ~60M |
| VGG19 | 2014 | 얇은 3×3 컨볼루션 | ~143M |
| ResNet50 | 2015 | Bottleneck, Skip Connections | ~25.6M |
| Inception | 2014 | Multi-Branch, 1×1 conv | ~5.8M |
| MobileNetV2 | 2018 | Depthwise Separable, Inverted Residual | ~3.5M |
| Xception | 2017 | Depthwise Separable, Extends Inception | ~23M |
| ResNeXt | 2016 | Group Convolution, Cardinality | ~30M |
| RepVGG | 2021 | Reparameterization, Training-time structure | ~21M |
| GoogLeNet | 2014 | Auxiliary Classifiers, Multi-Scale | ~5.8M |
| EfficientNet | 2019 | Compound Scaling, MBConv | ~5.3M |
| SqueezeNet | 2016 | Fire Module, 1×1 dominant | ~1.2M |
| DenseNet | 2017 | Dense Connection, Feature Reuse | ~8M |
| NASNet | 2018 | Neural Architecture Search | ~5.4M |
| MobileNetV3 | 2019 | h-swish, SE blocks | ~5.4M |

## 📁 레포지토리 구조

```
CNN-From-Scratch-With-PyTorch/
├── AlexNet.py              # AlexNet 구현
├── VGG.py                  # VGG19 구현
├── GoogLeNet.py            # Inception v1 구현
├── ResNet.py               # ResNet50 구현
├── MobileNetV2.py          # MobileNetV2 구현
├── Xception.py             # Xception 구현
├── ResNeXt.py              # ResNeXt 구현
├── RepVGG.py               # RepVGG 구현
├── EfficientNet.py         # EfficientNet-B0 구현
├── SqueezeNet.py           # SqueezeNet 1.0 구현
├── DenseNet.py             # DenseNet-121 구현
├── NASNet.py               # NASNet-A Mobile 구현
├── MobileNetV3.py          # MobileNetV3 구현
├── README.md
└── references.md           # 참고 논문 및 자료
```

## 📚 참고 논문

1. **AlexNet** - Krizhevsky et al. (2012)
2. **VGG** - Simonyan & Zisserman (2015)
3. **ResNet** - He et al. (2015)
4. **GoogLeNet** - Szegedy et al. (2015)
5. **MobileNetV2** - Sandler et al. (2018)
6. **Xception** - Chollet (2017)
7. **ResNeXt** - Xie et al. (2016)
8. **RepVGG** - Chen et al. (2021)
9. **EfficientNet** - Tan & Le (2019)
10. **SqueezeNet** - Iandola et al. (2016)
11. **DenseNet** - Huang et al. (2017)
12. **NASNet** - Zoph et al. (2018)
13. **MobileNetV3** - Howard et al. (2019)

## 📧 연락처

**작업자**: Jung-woojin  
**이메일**: wojin010629@gmail.com

**참여 활동**:
- 14 개 CNN 백본 네트워크 직접 구현 및 설계
- 아키텍처별 핵심 구성 요소 (Bottleneck, Inception, MBConv 등) 구현
- 통합 테스트 프레임워크 개발 및 벤치마킹
- 각 아키텍처의 설계 의도 및 구현 세부사항 분석

*Last modified: 2026-04-03*
