# EfficientNet (B0) - PyTorch Implementation
# Original EfficientNet (2019): Compound Scaling for ImageNet


import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) Block
    
    Mechanism:
    - Squeeze: Global average pooling to get channel-wise statistics
    - Excitation: Two-layer MLP with ReLU + Sigmoid to get channel weights
    - Scale: Multiply original feature map by learned channel weights
    
    This allows the network to perform adaptive feature recalibration,
    enhancing useful features and suppressing less useful ones.
    """
    
    def __init__(self, num_channels, reduction=4):
        super(SEBlock, self).__init__()
        
        # Calculate hidden layer size
        num_channels_hidden = max(1, num_channels // reduction)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Channel attention network
        self.fc1 = nn.Linear(num_channels, num_channels_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(num_channels_hidden, num_channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        
        # Squeeze: Global average pooling
        x_squeezed = self.avg_pool(x)
        x_squeezed = x_squeezed.view(batch_size, num_channels)
        
        # Excitation
        x_attended = self.fc1(x_squeezed)
        x_attended = self.relu(x_attended)
        x_attended = self.fc2(x_attended)
        x_weights = self.sigmoid(x_attended)
        
        # Scale: Multiply original feature map by learned weights
        x_scaled = x * x_weights.unsqueeze(1).unsqueeze(2).expand_as(x)
        
        return x_scaled


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck (MBConv) Block
    
    Structure:
    1. Pointwise Conv (1x1) - Expansion: Input → Expanded (expand_ratio * input_dim)
    2. Depthwise Conv (3x3) - Depthwise separable convolution
    3. Squeeze-and-Excitation (SE) - Channel attention (if use_se=True)
    4. Pointwise Conv (1x1) - Projection: Expanded → Output (output_dim)
    
    Skip connection: Only if input_dim == output_dim and stride == 1
    """
    
    def __init__(self, in_channels, out_channels, stride, expand_ratio, 
                 use_se, use_fused, input_layer_id=1):
        super(MBConvBlock, self).__init__()
        
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.use_se = use_se
        self.use_fused = use_fused
        
        hidden_dim = int(round(in_channels * expand_ratio))
        
        if use_fused:
            # Fused MBConv: No expansion conv, just Depthwise + Pointwise
            self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, 
                                  stride=2 if input_layer_id < 2 else 1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU6(inplace=True)
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2) if input_layer_id < 2 else nn.Identity()
        else:
            # Standard MBConv: Expansion → Depthwise → Projection
            self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            
            self.conv_dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, 
                                    stride=stride, padding=1, groups=hidden_dim, bias=False)
            self.bn_dw = nn.BatchNorm2d(hidden_dim)
            
            # Squeeze-and-Excitation
            self.se = SEBlock(hidden_dim) if use_se else nn.Identity()
            
            self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        if self.use_fused:
            x = self.pool(self.conv1(x))
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            identity = x
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            
            x = self.conv_dw(x)
            x = self.bn_dw(x)
            x = self.relu(x)
            
            x = self.se(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            identity = x
        
        if self.use_res_connect:
            x = self.relu(x + identity)
        else:
            x = self.relu(x)
        
        return x


class FusedConvBlock(nn.Module):
    """Fused Convolution Block (Early layers)
    
    Structure:
    1. Depthwise Conv (3x3)
    2. Pointwise Conv (1x1)
    
    Used in the first convolutional block for efficiency
    """
    
    def __init__(self, in_channels, out_channels, stride):
        super(FusedConvBlock, self).__init__()
        
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_channels)
        
        self.conv_pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        x = self.relu(x)
        
        x = self.conv_pw(x)
        x = self.bn_pw(x)
        x = self.relu(x)
        
        return x


class EfficientNetB0(nn.Module):
    """EfficientNet B0 Architecture
    
    Key Features:
    - Compound Scaling: Balance width, depth, resolution with hyperparameters
    - MBConv blocks with SE attention
    - Depthwise separable convolutions
    - Fused MBConv for early layers
    
    Parameters:
    - Width coefficient α = 1.0
    - Depth coefficient β = 1.0  
    - Resolution coefficient δ = 1.0
    - ImageNet input: 224x224 RGB images
    
    Returns:
    - 1000-class classification output
    """
    
    def __init__(self, num_classes=1000, dropout=0.2):
        super(EfficientNetB0, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6(inplace=True)
        
        # MBConv blocks configuration for B0
        # (input_dim, output_dim, stride, expand_ratio, use_se, use_fused, input_layer_id)
        self.mbconv_configs = [
            # (in_dim, out_dim, stride, expand_ratio, use_se, use_fused, layer_id)
            (32, 16, 1, 1, False, True, 1),      # Layer 1: Fused conv
            (16, 24, 2, 6, True, False, 2),       # Layer 2: MBConv with SE
            (24, 40, 2, 6, True, False, 3),       # Layer 3: MBConv with SE
            (40, 32, 1, 2, False, False, 4),      # Layer 4: MBConv no SE
            (32, 64, 2, 4, True, False, 5),       # Layer 5: MBConv with SE
            (64, 72, 1, 3, True, False, 6),       # Layer 6: MBConv with SE
            (72, 128, 2, 6, True, False, 7),      # Layer 7: MBConv with SE
            (128, 160, 1, 5, True, False, 8),     # Layer 8: MBConv with SE
            (160, 200, 1, 6, True, False, 9),     # Layer 9: MBConv with SE
            (200, 240, 2, 4, True, False, 10),    # Layer 10: MBConv with SE
            (240, 320, 1, 6, True, False, 11),    # Layer 11: MBConv with SE
            (320, 400, 1, 6, True, False, 12),    # Layer 12: MBConv with SE
        ]
        
        # Create MBConv blocks
        self.mbconv_blocks = nn.ModuleList()
        for i, (in_dim, out_dim, stride, expand_ratio, use_se, use_fused, layer_id) in enumerate(self.mbconv_configs):
            self.mbconv_blocks.append(MBConvBlock(
                in_dim, out_dim, stride, expand_ratio, use_se, use_fused, layer_id
            ))
        
        # Final convolution block
        self.final_conv = nn.Conv2d(320, 576, kernel_size=1, bias=False)
        self.bn_final = nn.BatchNorm2d(576)
        self.relu_final = nn.ReLU6(inplace=True)
        
        # Global average pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(576, num_classes)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        # MBConv blocks
        for block in self.mbconv_blocks:
            x = block(x)
        
        # Final convolution
        x = self.final_conv(x)
        x = self.bn_final(x)
        x = self.relu_final(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# Factory function
def create_efficientnet_b0(num_classes=1000, dropout=0.2):
    """Create EfficientNet B0 model with specified number of classes"""
    model = EfficientNetB0(num_classes=num_classes, dropout=dropout)
    return model


if __name__ == "__main__":
    # Create model
    model = create_efficientnet_b0(num_classes=1000)
    
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
    torch.save(model.state_dict(), 'efficientnet_b0.pth')
    print("Model saved as efficientnet_b0.pth")
