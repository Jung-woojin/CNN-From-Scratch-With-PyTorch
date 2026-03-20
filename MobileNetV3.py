# MobileNetV3 Large - PyTorch Implementation
# Original MobileNetV3 (2019): "Search for MobileNetV3"


import torch
import torch.nn as nn
import torch.nn.functional as F


def h_swish(x):
    """Hard-swish activation function
    
    h-swish(x) = x * ReLU6(x + 3) / 6
    
    ReLU6-based approximation of Swish, more efficient for mobile devices
    """
    return x * F.relu6(x + 3, inplace=True) / 6


def h_sigmoid(x):
    """Hard-sigmoid activation function
    
    h-sigmoid(x) = ReLU6(x + 3) / 6
    
    ReLU6-based approximation of sigmoid, used for SE weights
    """
    return F.relu6(x + 3, inplace=True) / 6


class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) Module
    
    Performs adaptive feature recalibration by learning channel-wise weights.
    Used to boost performance with minimal computational overhead.
    
    Args:
        num_channels: Number of input channels
        se_ratio: Squeeze ratio (reduction factor)
    """
    
    def __init__(self, num_channels, se_ratio=4):
        super(SEModule, self).__init__()
        
        num_channels_hidden = max(1, num_channels // se_ratio)
        
        # Squeeze: Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: Two-layer MLP with hard-sigmoid
        self.fc1 = nn.Linear(num_channels, num_channels_hidden)
        self.fc2 = nn.Linear(num_channels_hidden, num_channels)
    
    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        
        # Squeeze
        x_squeezed = self.avg_pool(x)
        x_squeezed = x_squeezed.view(batch_size, num_channels)
        
        # Excitation
        x_attended = self.fc2(torch.relu(self.fc1(x_squeezed)))
        x_weights = h_sigmoid(x_attended)
        
        # Scale
        x_scaled = x * x_weights.unsqueeze(1).unsqueeze(2).expand_as(x)
        
        return x_scaled


class InvertedResidual(nn.Module):
    """Inverted Residual Block - Core building block of MobileNetV3
    
    Structure:
    1. Pointwise Conv (1x1) - Expansion: Input → Expanded
    2. Depthwise Conv (3x3) - Spatial filtering
    3. Squeeze-and-Excitation (optional)
    4. Pointwise Conv (1x1) - Projection: Expanded → Output
    
    Skip connection: Only if in_channels == out_channels and stride == 1
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (3 or 5)
        stride: Convolution stride
        expand_ratio: Expansion ratio (1x1 → expand_ratio * in_channels)
        use_se: Whether to use SE module
        activation: Activation function ('relu' or 'h-swish')
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, use_se, activation='relu'):
        super(InvertedResidual, self).__init__()
        
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.use_se = use_se
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        
        # Expansion layer (1x1 convolution)
        self.expansion = int(round(in_channels * expand_ratio))
        self.use_expansion = expand_ratio > 1
        
        if self.use_expansion:
            self.conv1 = nn.Conv2d(in_channels, self.expansion, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.expansion)
            self.activation = h_swish if activation == 'h-swish' else nn.ReLU6(inplace=True)
        else:
            self.conv1 = None
        
        # Depthwise convolution (3x3 or 5x5)
        self.conv2 = nn.Conv2d(self.expansion, self.expansion, kernel_size=kernel_size, 
                              stride=stride, padding=kernel_size//2, groups=self.expansion, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion)
        
        # Squeeze-and-Excitation (optional)
        self.se = SEModule(self.expansion) if use_se else nn.Identity()
        
        # Projection layer (1x1 convolution)
        self.conv3 = nn.Conv2d(self.expansion, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        if self.use_expansion:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        x = self.se(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.use_res_connect:
            x = x + self.activation(x)
        
        return x


class MobileNetV3Large(nn.Module):
    """MobileNetV3 Large Architecture
    
    Key Features:
    - Mobile Inverted Residual Bottleneck (MBConv)
    - h-swish activation function (ReLU6-based)
    - h-sigmoid activation function
    - Squeeze-and-Excitation (SE) modules in later layers
    - Depthwise Separable Convolutions
    - Efficiently designed for mobile devices
    
    Architecture:
    1. Stem: 3x3, stride 2, 16 channels
    2. Inverted Residual: 32, k=3, s=2, SE
    3. Inverted Residual: 16, k=3, s=1, no SE
    4. Inverted Residual: 64, k=3, s=2, SE
    5. Inverted Residual: 72, k=3, s=1, SE
    6. Inverted Residual: 88, k=3, s=1, SE
    7. Inverted Residual: 104, k=3, s=1, SE
    8. Inverted Residual: 96, k=3, s=1, no SE
    9. Inverted Residual: 160, k=3, s=1, SE
    10. Final Conv: 576 channels
    11. Global AvgPool + FC
    
    Args:
        num_classes: Number of output classes (default 1000)
        alpha: Width multiplier (0.75, 1.0, 1.25)
    """
    
    def __init__(self, num_classes=1000, alpha=1.0):
        super(MobileNetV3Large, self).__init__()
        
        self.num_classes = num_classes
        
        # Configuration for MobileNetV3 Large
        # [expand_ratio, channels, stride, kernel, SE, activation]
        self.layer_configs = [
            # [expand_ratio, out_channels, stride, kernel_size, use_se, activation]
            [3, int(16 * alpha * 1), 1, 3, False, 'relu'],       # Layer 1
            [3, int(16 * alpha * 2), 2, 3, True, 'h-swish'],    # Layer 2
            [3, int(16 * alpha * 3), 1, 3, False, 'relu'],      # Layer 3
            [3, int(64 * alpha * 2), 2, 3, True, 'h-swish'],    # Layer 4
            [3, int(64 * alpha * 3), 1, 3, True, 'h-swish'],    # Layer 5
            [3, int(64 * alpha * 4), 1, 3, True, 'h-swish'],    # Layer 6
            [3, int(64 * alpha * 5), 1, 3, True, 'h-swish'],    # Layer 7
            [3, int(64 * alpha * 6), 1, 3, False, 'relu'],      # Layer 8
            [6, int(64 * alpha * 7), 2, 3, True, 'h-swish'],    # Layer 9
            [6, int(64 * alpha * 8), 1, 3, True, 'h-swish'],    # Layer 10
            [6, int(64 * alpha * 9), 1, 3, True, 'h-swish'],    # Layer 11
            [6, int(64 * alpha * 10), 1, 3, True, 'h-swish'],   # Layer 12
            [6, int(64 * alpha * 11), 1, 3, True, 'h-swish'],   # Layer 13
            [6, int(64 * alpha * 12), 1, 3, True, 'h-swish'],   # Layer 14
            [6, int(64 * alpha * 13), 1, 3, True, 'h-swish'],   # Layer 15
            [6, int(64 * alpha * 14), 1, 3, True, 'h-swish'],   # Layer 16
            [6, int(64 * alpha * 15), 1, 3, True, 'h-swish'],   # Layer 17
            [6, int(64 * alpha * 16), 1, 3, True, 'h-swish'],   # Layer 18
        ]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, int(16 * alpha), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(16 * alpha)),
            h_swish()
        )
        
        # Create inverted residual blocks
        self.blocks = nn.ModuleList()
        
        for i, (expand_ratio, out_channels, stride, kernel_size, use_se, activation) in enumerate(self.layer_configs):
            self.blocks.append(InvertedResidual(
                int(16 * alpha) if i == 0 else int(16 * alpha * (i // 2) + 16),
                out_channels,
                kernel_size,
                stride,
                expand_ratio,
                use_se,
                activation
            ))
        
        # Final conv block
        self.final_conv = nn.Sequential(
            nn.Conv2d(int(16 * alpha * (len(self.layer_configs) // 2) + 16), 576, kernel_size=1, bias=False),
            nn.BatchNorm2d(576),
            h_swish()
        )
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(576, num_classes)
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Process through blocks
        for block in self.blocks:
            x = block(x)
        
        # Final conv
        x = self.final_conv(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# Factory function
def create_mobilenetv3_large(num_classes=1000, alpha=1.0):
    """Create MobileNetV3 Large model with specified number of classes"""
    model = MobileNetV3Large(num_classes=num_classes, alpha=alpha)
    return model


if __name__ == "__main__":
    # Create model
    model = create_mobilenetv3_large(num_classes=1000, alpha=1.0)
    
    # Test with dummy input
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    
    print(f"Input size: {x.shape}")
    print(f"Output size: {output.shape}")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Save model
    torch.save(model.state_dict(), 'mobilenetv3_large.pth')
    print("Model saved as mobilenetv3_large.pth")
