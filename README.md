# CNN 백본 네트워크 직접 구현 연습 🧠🏗️

**PyTorch 로 14 개 CNN 아키텍처 직접 구현하며 백본 네트워크 이해를 극대화합니다.**

**작업자**: Jung-woojin  
**이메일**: wojin010629@gmail.com  
**최종 업데이트**: 2026-04-03

---

## 🎯 이 레포지토리의 목적

**CNN 백본 네트워크 (Backbone Network) 를 직접 구현하며 학습합니다.**

- 각 아키텍처의 핵심 아이디어를 코드로 구현
- PyTorch 의 기본적인 `nn.Module`, `nn.Conv2d`, `nn.BatchNorm2d` 등을 활용
- 복잡한 모듈 (ResBlock, Inception, MBConv 등) 을 직접 설계
- 아키텍처 설계의 의도와 세부 사항을 깊이 있게 이해

> **핵심**: 이론을 공부하는 것을 넘어, 직접 코딩하며 완전히 이해합니다.

---

## 📋 구현 완료된 아키텍처

### 1️⃣ 기본 아키텍처

| # | 아키텍처 | 연도 | 특징 | 레이어 수 | 파라미터 |
|---|----------|------|------|---------|----------|
| 1 | **AlexNet** | 2012 | ReLU, Dropout, Overlapping Pool | 8 | ~60M |
| 2 | **VGG19** | 2014 | Thin 3×3 convolutions, 19 layers | 19 | ~143M |
| 3 | **GoogLeNet** | 2014 | Inception Module, Aux Classifiers | 22 | ~5.8M |

### 2️⃣ Residual 기반

| # | 아키텍처 | 연도 | 특징 | 레이어 수 | 파라미터 |
|---|----------|------|------|---------|----------|
| 4 | **ResNet50** | 2016 | Bottleneck Block, Skip Connection | 50 | ~25.6M |
| 5 | **ResNeXt** | 2017 | Group Convolution, Cardinality | - | ~30M |

### 3️⃣ 효율성 최적화

| # | 아키텍처 | 연도 | 특징 | 레이어 수 | 파라미터 |
|---|----------|------|------|---------|----------|
| 6 | **MobileNetV2** | 2018 | Inverted Residual, Depthwise Separable | - | ~3.5M |
| 7 | **Xception** | 2017 | Depthwise Separable, Extends Inception | - | ~23M |
| 8 | **MobileNetV3** | 2019 | h-swish, SE blocks, MobileOptimized | - | ~5.4M |

### 4️⃣ 혁신적 설계

| # | 아키텍처 | 연도 | 특징 | 레이어 수 | 파라미터 |
|---|----------|------|------|---------|----------|
| 9 | **Inception v1** | 2014 | Multi-Branch, 1×1 dimension reduction | - | ~5.8M |
| 10 | **RepVGG** | 2021 | Reparameterization, Train/Inference | - | ~21M |
| 11 | **EfficientNet-B0** | 2019 | Compound Scaling, MBConv, SE blocks | 100+ | ~5.3M |
| 12 | **SqueezeNet 1.0** | 2016 | Fire Module, 1×1 dominant | - | ~1.2M |

### 5️⃣ 최첨단

| # | 아키텍처 | 연도 | 특징 | 레이어 수 | 파라미터 |
|---|----------|------|------|---------|----------|
| 13 | **DenseNet-121** | 2017 | Dense Connection, Feature Reuse | 121 | ~8M |
| 14 | **NASNet-A Mobile** | 2018 | Neural Architecture Search | - | ~5.4M |

---

## 🔥 핵심 기술 구현

### ResNet Bottleneck Block

```python
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out
```

### Inception Module

```python
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, 
                 out_3x3_reduce, out_3x3,
                 out_5x5_reduce, out_5x5,
                 out_pool_proj):
        super().__init__()
        
        # 1x1 path
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, 1),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 path (1x1 reduction → 3x3)
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3_reduce, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_3x3_reduce, out_3x3, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 5x5 path (1x1 reduction → 5x5)
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5_reduce, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5_reduce, out_5x5, 5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # pooling path (1x1 projection)
        self.path4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool_proj, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return torch.cat([self.path1(x), self.path2(x), 
                         self.path3(x), self.path4(x)], dim=1)
```

### MBConv (MobileNetV2/V3, EfficientNet)

```python
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio,
                 stride, kernel_size=3, se_ratio=None):
        super().__init__()
        
        hidden_dim = in_channels * expand_ratio
        use_se = se_ratio and se_ratio > 0
        
        # Expansion: 1×1 Conv
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, 1)
        self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        # Depthwise Convolution
        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, 
                                   stride=stride, padding=kernel_size//2,
                                   groups=hidden_dim)
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        
        # Squeeze-and-Excitation
        if use_se:
            self.se = SELayer(hidden_dim, se_ratio)
        else:
            self.se = nn.Identity()
        
        # Projection: 1×1 Conv
        self.project = nn.Conv2d(hidden_dim, out_channels, 1)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.use_residual = stride == 1 and in_channels == out_channels
    
    def forward(self, x):
        residual = x
        
        # Expansion
        x = self.expand_conv(x)
        x = self.expand_bn(x)
        x = self.relu(x)
        
        # Depthwise
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.relu(x)
        
        # SE
        x = self.se(x)
        
        # Projection
        x = self.project(x)
        x = self.project_bn(x)
        
        # Residual connection
        if self.use_residual:
            x = x + residual
        
        return x
```

### SqueezeNet - Fire Module

```python
class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super().__init__()
        
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.squeeze_relu = nn.ReLU(inplace=True)
        
        self.expand_1x1 = nn.Conv2d(squeeze_channels, expand_channels, 1)
        self.expand_1x1_relu = nn.ReLU(inplace=True)
        
        self.expand_3x3 = nn.Conv2d(squeeze_channels, expand_channels, 3, padding=1)
        self.expand_3x3_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.squeeze_relu(self.squeeze(x))
        
        return torch.cat([
            self.expand_1x1_relu(self.expand_1x1(x)),
            self.expand_3x3_relu(self.expand_3x3(x))
        ], 1)
```

### DenseNet - Dense Block

```python
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], 1)
        
        return x

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, 3, padding=1, bias=False)
    
    def forward(self, x):
        out = self.relu(self.bn(x))
        return self.conv(out)
```

---

## 🧪 테스트 및 벤치마킹

### 통합 테스트 도구

`model_test.py` 를 통해 모든 아키텍처를 비교, 벤치마킹할 수 있습니다.

```bash
# 단일 모델 테스트
python model_test.py --image test.jpg

# 모든 모델 비교
python model_test.py --image test.jpg --compare

# 벤치마킹 (10 회 반복 평균)
python model_test.py --image test.jpg --benchmark
```

### 출력 예시

```
ARCHITECTURE COMPARISON
===================================
Image: test.jpg

Model          Top 1          Time (ms)    Params (M)
--------------- ---------- ---------- ----------
alexnet        72.50%        12.34       57.84
densenet       75.12%        45.67        8.04
efficientnet   78.43%        23.89        5.29
mobilenetv3    72.56%         8.92        5.40
vgg19          65.89%        89.23      134.31

BEST PERFORMING MODELS
===================================
1. efficientnet: 78.43% accuracy
2. densenet: 75.12% accuracy
3. mobilenetv3: 72.56% accuracy
```

---

## 📚 학습 목표

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

4. **모델 비교 및 분석**
   - 파라미터 효율성
   - 추론 속도
   - 정확도 트레이드오프

---

## 🎓 구현 순서 추천

**Beginner → Advanced**

1. **Step 1**: AlexNet (가장 기본)
2. **Step 2**: VGG19 (심층 CNN 이해)
3. **Step 3**: GoogLeNet (Inception module)
4. **Step 4**: ResNet50 (Skip connection, Bottleneck)
5. **Step 5**: MobileNetV2 (Depthwise, Inverted Residual)
6. **Step 6**: EfficientNet (Compound Scaling, MBConv)
7. **Step 7**: DenseNet (Dense connection)
8. **Step 8**: NASNet (Neural Architecture Search 기반)
9. **Step 9**: RepVGG (Reparameterization)
10. **Step 10**: MobileNetV3 (SE, h-swish)

---

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
├── model_test.py           # 통합 테스트 및 벤치마킹
├── README.md
└── requirements.txt
```

---

## 📖 참고 자료

### 주요 논문

1. **AlexNet** - Krizhevsky et al. (2012)
2. **VGG** - Simonyan & Zisserman (2015)
3. **GoogLeNet** - Szegedy et al. (2015)
4. **ResNet** - He et al. (2015)
5. **MobileNetV2** - Sandler et al. (2018)
6. **Xception** - Chollet (2017)
7. **ResNeXt** - Xie et al. (2016)
8. **EfficientNet** - Tan & Le (2019)
9. **SqueezeNet** - Iandola et al. (2016)
10. **DenseNet** - Huang et al. (2017)
11. **NASNet** - Zoph et al. (2018)
12. **MobileNetV3** - Howard et al. (2019)
13. **RepVGG** - Chen et al. (2021)

---

## 📧 연락처

**작업자**: Jung-woojin  
**이메일**: wojin010629@gmail.com

**참여 활동**:
- 14 개 CNN 백본 네트워크 직접 구현 및 설계
- 아키텍처별 핵심 구성 요소 (Bottleneck, Inception, MBConv 등) 구현
- 통합 테스트 프레임워크 개발 및 벤치마킹
- 각 아키텍처의 설계 의도 및 구현 세부사항 분석

*Last modified: 2026-04-03*
