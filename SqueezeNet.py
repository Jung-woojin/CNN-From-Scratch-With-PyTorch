# SqueezeNet 1.0 - PyTorch Implementation
# Original SqueezeNet (2016): "AlexNet-level accuracy with 50x fewer parameters"


import torch
import torch.nn as nn
import torch.nn.functional as F


class FireModule(nn.Module):
    """Fire Module - Core building block of SqueezeNet
    
    Structure:
    1. Squeeze: 1x1 convolution reducing input channels
    2. Expand: Split and apply 1x1 + 3x3 convolutions in parallel
    3. Concatenate results
    
    Key benefits:
    - Reduces parameters by using more 1x1 convolutions
    - 1x1 convolutions before 3x3 reduce computation
    - Maintains accuracy with fewer parameters
    
    Args:
        in_channels: Number of input channels
        squeeze_channels: Number of channels after squeeze (1x1 conv)
        expand_channels: Number of channels per expand branch (1x1 and 3x3)
    """
    
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()
        
        self.in_channels = in_channels
        self.squeeze_channels = squeeze_channels
        self.expand_channels = expand_channels
        
        # Squeeze: 1x1 convolution
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        # Expand: Split into 1x1 and 3x3 branches
        self.expand_1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand_1x1_activation = nn.ReLU(inplace=True)
        
        self.expand_3x3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1)
        self.expand_3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x_squeezed = self.squeeze_activation(self.squeeze(x))
        
        x_1x1 = self.expand_1x1_activation(self.expand_1x1(x_squeezed))
        x_3x3 = self.expand_3x3_activation(self.expand_3x3(x_squeezed))
        
        # Concatenate results
        output = torch.cat([x_1x1, x_3x3], dim=1)
        
        return output


class SqueezeNet(nn.Module):
    """SqueezeNet 1.0 Architecture
    
    Key Features:
    - Replaces most 3x3 convolutions with 1x1 convolutions
    - Replaces max pooling with 1x1 convolutions with stride 2
    - No dropout (achieves AlexNet accuracy with 50x fewer parameters)
    - 8 Fire modules with increasing channel sizes
    
    Architecture:
    1. Conv1: 64 channels, 7x7 kernel, stride 2
    2. MaxPool: 3x3, stride 2
    3. Fire2: 16 squeeze, 16 expand
    4. Fire3: 16 squeeze, 32 expand
    5. Fire4: 16 squeeze, 32 expand
    6. Fire5: 32 squeeze, 48 expand
    7. Fire6: 32 squeeze, 48 expand
    8. Fire7: 48 squeeze, 64 expand
    9. Fire8: 48 squeeze, 64 expand
    10. Conv9: 1x1, 1000 channels
    11. Dropout: 0.5
    12. Classifier
    
    Input: 224x224 RGB images
    Output: 1000-class ImageNet prediction
    """
    
    def __init__(self, num_classes=1000, version='1.0'):
        super(SqueezeNet, self).__init__()
        
        if version not in ['1.0', '1.1']:
            raise ValueError("Version must be '1.0' or '1.1'")
        
        self.version = version
        
        # Layer 1: Convolutional layer (7x7, 64 channels)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Layer 2: Fire module
        self.fire2 = FireModule(64, 16, 16)
        self.fire3 = FireModule(128, 16, 32)
        
        # Layer 3: Pooling layer (replaced by conv in 1.1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.fire4 = FireModule(256, 16, 32)
        self.fire5 = FireModule(320, 32, 48)
        
        # Layer 4: Pooling layer
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.fire6 = FireModule(512, 32, 48)
        self.fire7 = FireModule(512, 48, 64)
        self.fire8 = FireModule(768, 48, 64)
        
        # Layer 5: Fire module with 1000 output channels
        self.fire9 = FireModule(1024, 64, 64)
        
        # Layer 6: Dropout
        self.dropout = nn.Dropout(p=0.5)
        
        # Layer 7: Final conv layer
        self.conv10 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.relu10 = nn.ReLU(inplace=True)
        
        # Layer 8: Average pooling
        self.avgpool = nn.AvgPool2d(13, stride=1)
        
        # Layer 9: Classifier
        self.classifier = nn.Conv2d(num_classes, num_classes, kernel_size=1)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.pool3(x)
        
        x = self.fire4(x)
        x = self.fire5(x)
        x = self.pool4(x)
        
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        
        x = self.fire9(x)
        x = self.dropout(x)
        x = self.relu10(self.conv10(x))
        x = self.avgpool(x)
        x = self.classifier(x)
        
        # Output: 1x1 conv with num_classes channels
        x = x.view(x.size(0), -1)
        return x


# Factory function
def create_squeezenet(num_classes=1000, version='1.0'):
    """Create SqueezeNet model with specified number of classes"""
    model = SqueezeNet(num_classes=num_classes, version=version)
    return model


if __name__ == "__main__":
    # Create model
    model = create_squeezenet(num_classes=1000, version='1.0')
    
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
    torch.save(model.state_dict(), 'squeezenet.pth')
    print("Model saved as squeezenet.pth")
