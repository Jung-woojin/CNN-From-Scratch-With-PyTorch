# DenseNet 121 - PyTorch Implementation
# Original DenseNet (2017): "Densely Connected Convolutional Networks"


import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """Dense Layer - Basic building block of DenseNet
    
    Structure:
    1. BatchNorm → ReLU → 1x1 Conv (reduce channels)
    2. BatchNorm → ReLU → 3x3 Conv (spatial conv)
    
    Output is concatenated with input (dense connection)
    
    Args:
        in_channels: Number of input channels
        growth_rate: Number of filters added by this layer
    """
    
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        
        bn_channels = in_channels
        conv_channels = growth_rate
        
        # 1x1 convolution to reduce dimensions
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, conv_channels, kernel_size=1, bias=False)
        
        # 3x3 convolution for spatial feature extraction
        self.norm2 = nn.BatchNorm2d(conv_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, 
                              padding=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        
        # Concatenate along channel dimension
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    """Dense Block - Multiple dense layers connected together
    
    Each layer's output is concatenated with all previous layers' outputs
    
    Args:
        num_layers: Number of layers in this block
        in_channels: Input channels to the block
        growth_rate: Number of filters added by each layer
    """
    
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        
        # Create dense layers
        self.layers = nn.ModuleList([
            DenseLayer(in_channels + i * growth_rate, growth_rate)
            for i in range(num_layers)
        ])
        
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.growth_rate = growth_rate
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x


class TransitionLayer(nn.Module):
    """Transition Layer - Connects DenseBlocks
    
    Functions:
    1. BatchNorm + ReLU
    2. 1x1 convolution to reduce channels (compression factor)
    3. Average pooling to reduce spatial dimensions
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        out = self.pool(out)
        return out


class DenseNet(nn.Module):
    """DenseNet-121 Architecture
    
    Key Features:
    - Dense connections: Each layer is connected to all subsequent layers
    - 1x1 convolutions reduce dimensionality (reduce-and-excite)
    - Transition layers connect dense blocks (reduce channels + spatial dims)
    - Global average pooling at the end
    
    Architecture:
    1. Conv1: 7x7, 64 filters
    2. MaxPool: 3x3, stride 2
    3. DenseBlock1: 6 layers, 32 filters each → 256 channels
    4. Transition1: Reduce 256→128 (compression=0.5)
    5. DenseBlock2: 12 layers, 32 filters each → 512 channels
    6. Transition2: Reduce 512→256
    7. DenseBlock3: 24 layers, 32 filters each → 1024 channels
    8. Transition3: Reduce 1024→512
    9. DenseBlock4: 16 layers, 32 filters each → 1536 channels
    10. BN + ReLU + AvgPool → Linear classifier
    
    Args:
        growth_rate: Number of filters added by each layer (default 32)
        block_config: Number of layers in each dense block (default [6, 12, 24, 16])
        num_classes: Number of output classes (default 1000)
    """
    
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=1000):
        super(DenseNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Number of filters in each dense block
        # Based on DenseNet-121: 64, 128, 256, 512, 1024 channels
        
        # DenseBlock1: 6 layers, growth_rate=32
        # Input: 64 → Output: 64 + 6*32 = 256
        self.denseblock1 = self._make_denseblock(6, 64, growth_rate)
        
        # Transition1: 256 → 128 (compression=0.5)
        self.transition1 = self._make_transition_layer(256, 128)
        
        # DenseBlock2: 12 layers, growth_rate=32
        # Input: 128 → Output: 128 + 12*32 = 512
        self.denseblock2 = self._make_denseblock(12, 128, growth_rate)
        
        # Transition2: 512 → 256 (compression=0.5)
        self.transition2 = self._make_transition_layer(512, 256)
        
        # DenseBlock3: 24 layers, growth_rate=32
        # Input: 256 → Output: 256 + 24*32 = 1024
        self.denseblock3 = self._make_denseblock(24, 256, growth_rate)
        
        # Transition3: 1024 → 512 (compression=0.5)
        self.transition3 = self._make_transition_layer(1024, 512)
        
        # DenseBlock4: 16 layers, growth_rate=32
        # Input: 512 → Output: 512 + 16*32 = 1536
        self.denseblock4 = self._make_denseblock(16, 512, growth_rate)
        
        # Final batchnorm + ReLU
        self.bn_final = nn.BatchNorm2d(1536)
        
        # Global average pooling and classifier
        self.classifier = nn.Linear(1536, num_classes)
    
    def _make_denseblock(self, num_layers, in_channels, growth_rate):
        """Create a DenseBlock with specified number of layers"""
        return DenseBlock(num_layers, in_channels, growth_rate)
    
    def _make_transition_layer(self, in_channels, out_channels):
        """Create a TransitionLayer"""
        return TransitionLayer(in_channels, out_channels)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # DenseBlock1
        x = self.denseblock1(x)
        x = self.transition1(x)
        
        # DenseBlock2
        x = self.denseblock2(x)
        x = self.transition2(x)
        
        # DenseBlock3
        x = self.denseblock3(x)
        x = self.transition3(x)
        
        # DenseBlock4
        x = self.denseblock4(x)
        
        # Final batchnorm + ReLU
        x = self.bn_final(x)
        x = self.relu(x)
        
        # Global average pooling
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        
        # Linear classifier
        x = self.classifier(x)
        
        return x


# Factory functions for different DenseNet variants
def densenet121(num_classes=1000):
    """Create DenseNet-121"""
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=num_classes)


def densenet169(num_classes=1000):
    """Create DenseNet-169"""
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), num_classes=num_classes)


def densenet201(num_classes=1000):
    """Create DenseNet-201"""
    return DenseNet(growth_rate=32, block_config=(6, 12, 48, 32), num_classes=num_classes)


def densenet161(num_classes=1000):
    """Create DenseNet-161"""
    return DenseNet(growth_rate=48, block_config=(6, 12, 36, 24), num_classes=num_classes)


if __name__ == "__main__":
    # Create DenseNet-121
    model = densenet121(num_classes=1000)
    
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
    torch.save(model.state_dict(), 'densenet121.pth')
    print("Model saved as densenet121.pth")
