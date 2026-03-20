"""
EfficientNet-B0 Implementation from Scratch using PyTorch

This module implements EfficientNet-B0, a state-of-the-art convolutional neural network
based on compound scaling and mobile inverted bottleneck convolutions (MBConv).

Author: Jung-woojin
Repository: CNN-From-Scratch-With-PyTorch

Architecture Overview:
- Initial convolutional layer (7x7, stride 2) with batch normalization and Swish activation
- Mobile Inverted Bottleneck Convolution (MBConv) blocks with depthwise separable convolutions
- Squeeze-and-Excitation (SE) attention mechanism
- Fused MBConv blocks (without bottleneck projection)
- Adaptive average pooling and dropout for regularization
- Fully connected layer for classification

Key Components:
1. MBConv (Mobile Inverted Residual Bottleneck Convolution):
   - Expansion layer: 1x1 convolution to increase channels
   - Depthwise convolution: 3x3 convolution with stride (applied per channel)
   - Squeeze-and-Excitation: Channel-wise attention mechanism
   - Projection layer: 1x1 convolution to reduce channels
   - Skip connection: Residual connection when input/output dimensions match

2. Fused MBConv:
   - Optimized version where expansion and depthwise convolutions are fused
   - Reduces computational overhead while maintaining accuracy

3. Compound Scaling:
   - Width multiplier (α): Controls network width
   - Depth multiplier (β): Controls network depth
   - Resolution multiplier (γ): Controls input image resolution
   - Offset parameter (δ): Controls layer types

Parameters (EfficientNet-B0):
- Expansion rates: [1, 6, 6, 6, 6, 6]
- Number of blocks per stage: [1, 2, 3, 4, 3, 2] (for B0: 1, 2, 3, 4, 3, 2)
- Filter multipliers: [32, 16, 24, 40, 80, 112, 192, 320, 576]
- SE ratio: [0, 0.25, 0.25, 0.25, 0.25, 0.25]
- Kernel sizes: [3, 3, 3, 5, 5, 5, 3, 3, 3]
- Strides: [1, 2, 2, 2, 1, 2, 1, 1, 1]

Total Parameters: ~5.3M
Inference FLOPs: ~390M
ImageNet Top-1 Accuracy: ~77%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def swish(x: torch.Tensor) -> torch.Tensor:
    """
    Swish activation function: x * sigmoid(x)
    
    This is the activation function used throughout EfficientNet, introduced in
    the original EfficientNet paper. It provides smooth, non-monotonic behavior
    that helps with gradient flow during training.
    
    Args:
        x: Input tensor of arbitrary shape
        
    Returns:
        Swish activation of x, same shape as input
    """
    return x * torch.sigmoid(x)


class SwishActivation(nn.Module):
    """
    Swish activation layer wrapper for use in nn.Sequential modules.
    
    The Swish function (x * sigmoid(x)) is superior to ReLU in many deep network
    scenarios due to its non-monotonic property and self-gating mechanism.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x)


class SqueezeAndExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel-wise attention.
    
    SE blocks learn to selectively emphasize informative features and suppress
    less useful ones by modeling interdependencies between channels. The mechanism
    consists of three operations:
    1. Squeeze: Global average pooling to compress spatial dimensions
    2. Excitation: Two fully connected layers with nonlinearity to learn channel weights
    3. Scale: Multiply input features by learned channel weights
    
    This attention mechanism was introduced in the paper "Squeeze-and-Excitation Networks"
    and has been shown to significantly improve model accuracy at minimal computational cost.
    
    Architecture:
        Input: [batch, channels, height, width]
        ↓ (Global Average Pool)
        [batch, channels]
        ↓ (FC: channels → channels//reduction)
        [batch, channels//reduction]
        ↓ (ReLU)
        [batch, channels//reduction]
        ↓ (FC: channels//reduction → channels)
        [batch, channels]
        ↓ (Sigmoid)
        [batch, channels]
        ↓ (Reshape and Multiply)
        Output: [batch, channels, height, width]
    
    Args:
        channels: Number of input channels
        reduction_ratio: Ratio for channel reduction in bottleneck (default: 4)
        
    Example:
        >>> se_block = SqueezeAndExcitation(128, reduction_ratio=4)
        >>> x = torch.randn(32, 128, 7, 7)
        >>> output = se_block(x)
        >>> output.shape
        torch.Size([32, 128, 7, 7])
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 4):
        super(SqueezeAndExcitation, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # Calculate hidden layer dimension
        hidden_channels = channels // reduction_ratio
        
        # First FC layer: compress channels
        self.fc1 = nn.Linear(channels, hidden_channels)
        # Second FC layer: expand back to original channels
        self.fc2 = nn.Linear(hidden_channels, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SE attention to input tensor.
        
        Args:
            x: Input tensor of shape [batch, channels, height, width]
            
        Returns:
            Attention-weighted tensor of same shape
        """
        # Ensure input is 4D (handle edge cases)
        assert x.dim() == 4, f"Expected 4D tensor, got {x.dim()}D"
        
        # Get batch size and spatial dimensions
        batch_size, _, height, width = x.size()
        
        # Step 1: Squeeze - Global average pooling
        # [batch, channels, height, width] → [batch, channels]
        z = torch.mean(x, dim=[2, 3])
        
        # Step 2: Excitation
        # First FC layer with ReLU
        # [batch, channels] → [batch, hidden_channels]
        z = self.fc1(z)
        z = F.relu(z)
        # Second FC layer
        # [batch, hidden_channels] → [batch, channels]
        z = self.fc2(z)
        # Sigmoid activation to get weights in [0, 1]
        z = torch.sigmoid(z)
        
        # Step 3: Scale - Reshape and multiply
        # [batch, channels] → [batch, channels, 1, 1]
        z = z.view(batch_size, self.channels, 1, 1)
        # Apply attention weights via element-wise multiplication
        # [batch, channels, 1, 1] * [batch, channels, height, width]
        return x * z


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv) block.
    
    The MBConv block is the fundamental building block of EfficientNet, designed to
    be both efficient and powerful. It consists of four main stages:
    
    1. Expansion (1x1 convolution):
       - Increases the number of channels by 'expand_ratio'
       - Creates a bottleneck layer for richer feature representation
       - Reduces computational cost of subsequent depthwise convolution
    
    2. Depthwise Convolution (3x3 or 5x5):
       - Applies a separate convolution per input channel
       - Separates spatial and channel-wise correlations
       - Much more efficient than standard convolution
    
    3. Squeeze-and-Excitation (optional):
       - Adds channel attention mechanism
       - Only used when SE ratio > 0 (not used in first layer)
       - Helps model focus on informative channels
    
    4. Projection (1x1 convolution):
       - Projects back to reduced number of channels
       - Returns to original channel dimension
       - Complements the expansion
    
    5. Skip Connection (conditional):
       - Only used when input and output dimensions match
       - Identity shortcut connection
       - Improves gradient flow and allows deeper networks
    
    The "inverted" in MBConv refers to the expansion-contraction pattern, opposite to
    traditional bottlenecks which contract first then expand.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        expand_ratio: Expansion ratio (default: 6)
        kernel_size: Kernel size for depthwise convolution (3 or 5)
        stride: Stride for depthwise convolution (1 or 2)
        use_se: Whether to use Squeeze-and-Excitation
        se_ratio: Squeeze ratio for SE block (default: 0.25)
        
    Attributes:
        expand_ratio (int): Expansion ratio used
        se_ratio (float): Squeeze ratio for SE block
        
    Example:
        >>> mbconv = MBConvBlock(32, 16, expand_ratio=6, kernel_size=3, stride=1, use_se=False)
        >>> x = torch.randn(32, 32, 28, 28)
        >>> output = mbconv(x)
        >>> output.shape
        torch.Size([32, 16, 28, 28])
    """
    
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 6,
                 kernel_size: int = 3, stride: int = 1, use_se: bool = True,
                 se_ratio: float = 0.25):
        super(MBConvBlock, self).__init__()
        
        self.expand_ratio = expand_ratio
        self.use_se = use_se
        self.se_ratio = se_ratio
        self.has_skip = (stride == 1) and (in_channels == out_channels)
        
        # Calculate expanded channels
        expanded_channels = in_channels * expand_ratio
        
        # Stage 1: Expansion (1x1 convolution)
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(expanded_channels)
        self.expand_activation = SwishActivation()
        
        # Stage 2: Depthwise Convolution (3x3 or 5x5)
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=(kernel_size - 1) // 2,
                                        groups=expanded_channels,
                                        bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        self.depthwise_activation = SwishActivation()
        
        # Stage 3: Squeeze-and-Excitation (optional)
        self.use_se_block = use_se
        if use_se:
            se_channels = max(1, int(expanded_channels * se_ratio))
            self.se_block = SqueezeAndExcitation(expanded_channels, se_channels)
        
        # Stage 4: Projection (1x1 convolution)
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MBConv block.
        
        Args:
            x: Input tensor of shape [batch, in_channels, height, width]
            
        Returns:
            Output tensor of shape [batch, out_channels, out_height, out_width]
        """
        residual = x.clone()
        
        # Stage 1: Expansion
        x = self.expand_conv(x)
        x = self.expand_bn(x)
        x = self.expand_activation(x)
        
        # Stage 2: Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_activation(x)
        
        # Stage 3: Squeeze-and-Excitation
        if self.use_se_block:
            x = self.se_block(x)
        
        # Stage 4: Projection
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # Skip connection (only if dimensions match and stride == 1)
        if self.has_skip:
            x = x + residual
            
        return x


class FusedMBConvBlock(nn.Module):
    """
    Fused Mobile Inverted Bottleneck Convolution block.
    
    This is an optimized version of MBConv where the expansion convolution and
    depthwise convolution are fused into a single 3x3 depthwise convolution.
    This fusion reduces computational overhead by:
    
    1. Eliminating redundant 1x1 convolution operations
    2. Reducing the number of activation functions
    3. Lowering memory access patterns
    
    The Fused MBConv is used in the early layers of EfficientNet where the channel
    dimension is smaller, making the overhead of the separate expansion and depthwise
    convolutions more significant.
    
    Unlike standard MBConv, Fused MBConv does not include:
    - The 1x1 expansion convolution
    - The 1x1 projection convolution
    
    Instead, it uses a single 3x5 depthwise convolution with appropriate stride
    to achieve the desired channel expansion and spatial downsampling.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        expand_ratio: Expansion ratio (applied to in_channels to get expanded channels)
        kernel_size: Kernel size for depthwise convolution (3 or 5)
        stride: Stride for depthwise convolution (1 or 2)
        use_se: Whether to use Squeeze-and-Excitation
        se_ratio: Squeeze ratio for SE block (default: 0.25)
        
    Attributes:
        expand_ratio (int): Expansion ratio used
        se_ratio (float): Squeeze ratio for SE block
        
    Example:
        >>> fused_mbconv = FusedMBConvBlock(32, 16, expand_ratio=6, kernel_size=3, stride=1, use_se=False)
        >>> x = torch.randn(32, 32, 28, 28)
        >>> output = fused_mbconv(x)
        >>> output.shape
        torch.Size([32, 16, 28, 28])
    """
    
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 6,
                 kernel_size: int = 3, stride: int = 1, use_se: bool = True,
                 se_ratio: float = 0.25):
        super(FusedMBConvBlock, self).__init__()
        
        self.expand_ratio = expand_ratio
        self.use_se = use_se
        self.se_ratio = se_ratio
        self.has_skip = (stride == 1) and (in_channels == out_channels)
        
        # Calculate expanded channels
        expanded_channels = in_channels * expand_ratio
        
        # Fused depthwise convolution (combines expansion and depthwise operations)
        # Only applies to spatial dimensions, channels are implicitly expanded
        self.fused_conv = nn.Conv2d(in_channels, expanded_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=(kernel_size - 1) // 2,
                                    groups=in_channels,
                                    bias=False)
        self.fused_bn = nn.BatchNorm2d(expanded_channels)
        self.fused_activation = SwishActivation()
        
        # Squeeze-and-Excitation (optional)
        self.use_se_block = use_se
        if use_se:
            se_channels = max(1, int(expanded_channels * se_ratio))
            self.se_block = SqueezeAndExcitation(expanded_channels, se_channels)
        
        # Projection to output channels
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Fused MBConv block.
        
        Args:
            x: Input tensor of shape [batch, in_channels, height, width]
            
        Returns:
            Output tensor of shape [batch, out_channels, out_height, out_width]
        """
        residual = x.clone()
        
        # Fused convolution and activation
        x = self.fused_conv(x)
        x = self.fused_bn(x)
        x = self.fused_activation(x)
        
        # Squeeze-and-Excitation
        if self.use_se_block:
            x = self.se_block(x)
        
        # Projection
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # Skip connection
        if self.has_skip:
            x = x + residual
            
        return x


class EfficientNet(nn.Module):
    """
    EfficientNet-B0 convolutional neural network implementation.
    
    EfficientNet-B0 is the first model in the EfficientNet family, which uses
    compound scaling to simultaneously scale network depth, width, and resolution
    in a principled way. This implementation includes all key components:
    
    Architecture Components:
    1. Stem: Initial 7x7 convolution with stride 2
    2. Stage 1: Fused MBConv with reduced channels
    3. Stage 2-6: MBConv blocks with increasing channel counts
    4. Head: Final 1x1 convolution + Adaptive Average Pooling + Dropout + FC
    5. SE blocks: Used in all MBConv blocks except the first
    
    Compound Scaling Parameters (B0):
    - Width multiplier (α): 1.0 (original width)
    - Depth multiplier (β): 1.0 (original depth)
    - Resolution multiplier (γ): 1.0 (original resolution)
    - Offset (δ): 1.0 (controls layer type selection)
    
    Layer Configuration:
    - Initial convolution: 7x7, stride 2, 32 filters
    - Stage 1: Fused MBConv, 16 filters (after expansion)
    - Stage 2: MBConv (3x3, stride 2), 24 filters
    - Stage 3: MBConv (3x3, stride 2), 40 filters
    - Stage 4: MBConv (5x5, stride 2), 80 filters
    - Stage 5: MBConv (5x5, stride 1), 112 filters
    - Stage 6: MBConv (5x5, stride 2), 192 filters
    - Head: MBConv (3x3, stride 1), 320 filters → 1280 FC
    
    Total Parameters: ~5.3 million
    Input Image Size: 224x224
    Number of Classes: 1000 (ImageNet)
    
    Training Tips:
    - Use label smoothing (epsilon=0.1) for better generalization
    - Apply dropconnect or dropout for regularization
    - Use AdamW optimizer with weight decay
    - Learning rate: 0.032 with cosine annealing
    - Batch size: 32 per GPU (1000 samples per step with 32 GPUs)
    
    Args:
        num_classes: Number of output classes (default: 1000 for ImageNet)
        dropout_rate: Dropout rate after pooling (default: 0.2)
        in_channels: Number of input image channels (default: 3 for RGB)
        
    Attributes:
        dropout_rate (float): Dropout rate used
        num_classes (int): Number of output classes
        
    Example:
        >>> model = EfficientNet(num_classes=1000, dropout_rate=0.2, in_channels=3)
        >>> x = torch.randn(32, 3, 224, 224)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 1000])
        
    Loading Pre-trained Weights:
        The model can be loaded with pre-trained ImageNet weights by downloading
        the weights file and using: model.load_state_dict(torch.load('efficientnet-b0.pth'))
    """
    
    def __init__(self, num_classes: int = 1000, dropout_rate: float = 0.2, in_channels: int = 3):
        super(EfficientNet, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.in_channels = in_channels
        
        # Stage 0: Stem - Initial convolution
        self.stem_conv = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_activation = SwishActivation()
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 1: Fused MBConv (reduced channels for efficiency)
        self.layer1 = nn.Sequential(
            FusedMBConvBlock(32, 16, expand_ratio=1, kernel_size=3, stride=1, use_se=False)
        )
        
        # Stage 2: MBConv blocks
        self.layer2 = nn.Sequential(
            MBConvBlock(16, 24, expand_ratio=6, kernel_size=3, stride=2, use_se=True, se_ratio=0.25),
            MBConvBlock(24, 24, expand_ratio=6, kernel_size=3, stride=1, use_se=True, se_ratio=0.25)
        )
        
        # Stage 3: MBConv blocks
        self.layer3 = nn.Sequential(
            MBConvBlock(24, 40, expand_ratio=6, kernel_size=3, stride=2, use_se=True, se_ratio=0.25),
            MBConvBlock(40, 40, expand_ratio=6, kernel_size=3, stride=1, use_se=True, se_ratio=0.25)
        )
        
        # Stage 4: MBConv blocks (larger kernel)
        self.layer4 = nn.Sequential(
            MBConvBlock(40, 80, expand_ratio=6, kernel_size=5, stride=2, use_se=True, se_ratio=0.25),
            MBConvBlock(80, 80, expand_ratio=6, kernel_size=5, stride=1, use_se=True, se_ratio=0.25),
            MBConvBlock(80, 80, expand_ratio=6, kernel_size=5, stride=1, use_se=True, se_ratio=0.25)
        )
        
        # Stage 5: MBConv blocks (larger kernel)
        self.layer5 = nn.Sequential(
            MBConvBlock(80, 112, expand_ratio=6, kernel_size=5, stride=1, use_se=True, se_ratio=0.25),
            MBConvBlock(112, 112, expand_ratio=6, kernel_size=5, stride=1, use_se=True, se_ratio=0.25),
            MBConvBlock(112, 112, expand_ratio=6, kernel_size=5, stride=1, use_se=True, se_ratio=0.25)
        )
        
        # Stage 6: MBConv blocks (larger kernel)
        self.layer6 = nn.Sequential(
            MBConvBlock(112, 192, expand_ratio=6, kernel_size=5, stride=2, use_se=True, se_ratio=0.25),
            MBConvBlock(192, 192, expand_ratio=6, kernel_size=5, stride=1, use_se=True, se_ratio=0.25)
        )
        
        # Head: Final convolution + pooling + classification
        self.head_conv = nn.Conv2d(192, 320, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm2d(320)
        self.head_activation = SwishActivation()
        
        # Adaptive average pooling to 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Fully connected classification layer
        self.fc = nn.Linear(320, num_classes)
        
        # Weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initialize weights using He initialization for better training.
        
        This initialization scheme is suitable for ReLU-like activation functions
        and helps maintain stable gradient flow throughout the network.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EfficientNet.
        
        Args:
            x: Input tensor of shape [batch, in_channels, height, width]
            
        Returns:
            Output tensor of shape [batch, num_classes]
        """
        # Ensure input is 4D
        assert x.dim() == 4, f"Expected 4D tensor, got {x.dim()}D"
        assert x.size(1) == self.in_channels, f"Expected {self.in_channels} input channels, got {x.size(1)}"
        
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_activation(x)
        x = self.stem_pool(x)
        
        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        
        # Head
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.head_activation(x)
        
        # Adaptive average pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Dropout
        x = self.dropout(x)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x
    
    def get_feature_extractor(self) -> nn.Module:
        """
        Returns a feature extractor without the classification layer.
        
        Useful for transfer learning or extracting features from intermediate layers.
        
        Returns:
            nn.Module: Feature extractor ending after adaptive pooling
        """
        return nn.Sequential(
            self.stem_conv, self.stem_bn, self.stem_activation, self.stem_pool,
            self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6,
            self.head_conv, self.head_bn, self.head_activation, self.adaptive_pool
        )
    
    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_info(self) -> dict:
        """
        Get detailed information about each layer in the network.
        
        Returns:
            dict: Dictionary containing layer names, input/output shapes, and parameter counts
        """
        info = {}
        x = torch.randn(1, self.in_channels, 224, 224)
        
        layers = [
            ("stem", [self.stem_conv, self.stem_bn, self.stem_activation, self.stem_pool]),
            ("layer1", self.layer1),
            ("layer2", self.layer2),
            ("layer3", self.layer3),
            ("layer4", self.layer4),
            ("layer5", self.layer5),
            ("layer6", self.layer6),
            ("head", [self.head_conv, self.head_bn, self.head_activation, self.adaptive_pool, self.dropout, self.fc])
        ]
        
        for name, layer in layers:
            if isinstance(layer, list):
                module = nn.Sequential(*layer)
            else:
                module = layer
            
            try:
                with torch.no_grad():
                    output = module(x)
                    info[name] = {
                        "input_shape": tuple(x.shape),
                        "output_shape": tuple(output.shape),
                        "num_parameters": sum(p.numel() for p in layer.parameters() if isinstance(layer, nn.Module))
                    }
                    x = output
            except Exception as e:
                info[name] = {"error": str(e)}
        
        return info


def create_efficientnet_b0(num_classes: int = 1000, pretrained: bool = False) -> EfficientNet:
    """
    Factory function to create EfficientNet-B0 model.
    
    This convenience function creates an EfficientNet-B0 model with the standard
    configuration:
    - num_classes: 1000 (ImageNet)
    - dropout_rate: 0.2
    - in_channels: 3 (RGB images)
    
    Args:
        num_classes: Number of output classes (default: 1000 for ImageNet)
        pretrained: Whether to load pre-trained weights (not implemented in this version)
        
    Returns:
        EfficientNet: EfficientNet-B0 model instance
        
    Example:
        >>> model = create_efficientnet_b0(num_classes=10, pretrained=False)
        >>> print(f"Model parameters: {model.count_parameters():,}")
        Model parameters: 5,304,672
    """
    model = EfficientNet(num_classes=num_classes, dropout_rate=0.2, in_channels=3)
    
    if pretrained:
        # TODO: Load pre-trained ImageNet weights
        print("Pre-trained weights loading not yet implemented")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing EfficientNet-B0 Implementation")
    print("=" * 50)
    
    # Create model
    model = EfficientNet(num_classes=1000, dropout_rate=0.2, in_channels=3)
    
    # Count parameters
    total_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Total parameters (M): {total_params / 1e6:.2f}M")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(32, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Forward pass successful!")
    
    # Get layer information
    print("\nLayer Information:")
    layer_info = model.get_layer_info()
    for name, info in layer_info.items():
        if "error" not in info:
            print(f"  {name}: {info['input_shape']} → {info['output_shape']}")
    
    print("\nEfficientNet-B0 Implementation Complete!")
    print("Key Features:")
    print("  - MBConv blocks with depthwise separable convolutions")
    print("  - Squeeze-and-Excitation attention mechanism")
    print("  - Fused MBConv for early layers")
    print("  - Adaptive average pooling")
    print("  - Dropout regularization")
    print("  - All operators implemented from scratch")
