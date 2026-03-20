# GoogLeNet (Inception v1) - PyTorch Implementation
# Original GoogLeNet (2014): First to use Inception modules


import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """Inception Module - Core component of GoogLeNet
    
    Multiple parallel branches with different filter sizes
    followed by 1x1 convolution for dimensionality reduction
    """
    
    def __init__(self, in_channels, out_channels_1x1, out_channels_3x3_reduce, 
                 out_channels_3x3, out_channels_5x5_reduce, out_channels_5x5, 
                 out_channels_pool):
        super(InceptionModule, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_1x1, kernel_size=1),
            nn.BatchNorm2d(out_channels_1x1),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution branch (1x1 -> 3x3)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(out_channels_3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels_3x3_reduce, out_channels_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels_3x3),
            nn.ReLU(inplace=True)
        )
        
        # 5x5 convolution branch (1x1 -> 5x5)
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(out_channels_5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels_5x5_reduce, out_channels_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels_5x5),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 max pooling branch (1x1 conv after pooling)
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels_pool, kernel_size=1),
            nn.BatchNorm2d(out_channels_pool),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        # Concatenate all branches
        output = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)
        return output


class GoogLeNet(nn.Module):
    """GoogLeNet (Inception v1) Architecture
    
    Key Features:
    - Inception modules with multi-scale convolutions
    - Auxiliary classifiers for better gradient flow
    - Aggressive downsampling and dropout for regularization
    - Average pooling at the end
    
    Input: 224x224 RGB images
    Output: 1000-class ImageNet prediction
    """
    
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()
        
        # Helper function for inception module
        def inception_block(in_channels, c1, c2, c3, c4):
            return InceptionModule(
                in_channels, c1[0], c2[0], c2[1], c3[0], c3[1], c4[0]
            )
        
        # Helper to create inception blocks with given parameters
        self.conv1_7x7 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Helper to create inception blocks with given parameters
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=1)
        
        # Inception 2a (384x28x28 -> 384x28x28)
        self.inception2a = inception_block(
            192,
            c1=[64],  # 1x1 output
            c2=[96, 128],  # 3x3 output (reduce, then conv)
            c3=[16, 32],  # 5x5 output (reduce, then conv)
            c4=[32]  # pool output
        )
        
        # Inception 2b (384x28x28 -> 512x28x28)
        self.inception2b = inception_block(
            384,
            c1=[128],
            c2=[128, 192],
            c3=[32, 96],
            c4=[32]
        )
        
        # Max pooling after Inception 2b
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Inception 3a (512x28x28 -> 512x28x28)
        self.inception3a = inception_block(
            512,
            c1=[192],
            c2=[96, 208],
            c3=[16, 48],
            c4=[64]
        )
        
        # Inception 3b (512x28x28 -> 512x28x28)
        self.inception3b = inception_block(
            512,
            c1=[160],
            c2=[112, 224],
            c3=[24, 64],
            c4=[64]
        )
        
        # Max pooling after Inception 3b
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Inception 4a (512x14x14 -> 512x14x14)
        self.inception4a = inception_block(
            512,
            c1=[96],
            c2=[128, 176],
            c3=[32, 48],
            c4=[32]
        )
        
        # Inception 4b (512x14x14 -> 512x14x14)
        self.inception4b = inception_block(
            512,
            c1=[128],
            c2=[144, 192],
            c3=[32, 48],
            c4=[32]
        )
        
        # Auxiliary classifier 1
        self.avgpool1 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv_aux1 = nn.Conv2d(512, 128, kernel_size=1)
        self.fc_aux1_1 = nn.Linear(128 * 4 * 4, 1024)
        self.dropout_aux1 = nn.Dropout(p=0.7)
        self.fc_aux1_2 = nn.Linear(1024, num_classes)
        
        # Inception 4c (512x14x14 -> 512x14x14)
        self.inception4c = inception_block(
            512,
            c1=[112],
            c2=[144, 160],
            c3=[24, 64],
            c4=[32]
        )
        
        # Inception 4d (512x14x14 -> 512x14x14)
        self.inception4d = inception_block(
            512,
            c1=[132],
            c2=[160, 160],
            c3=[32, 64],
            c4=[32]
        )
        
        # Inception 4e (512x14x14 -> 1024x14x14)
        self.inception4e = inception_block(
            512,
            c1=[256],
            c2=[160, 320],
            c3=[32, 128],
            c4=[64]
        )
        
        # Max pooling after Inception 4e
        self.maxpool6 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Auxiliary classifier 2
        self.avgpool2 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv_aux2 = nn.Conv2d(1024, 128, kernel_size=1)
        self.fc_aux2_1 = nn.Linear(128 * 4 * 4, 1024)
        self.dropout_aux2 = nn.Dropout(p=0.7)
        self.fc_aux2_2 = nn.Linear(1024, num_classes)
        
        # Inception 5a (1024x7x7 -> 1024x7x7)
        self.inception5a = inception_block(
            1024,
            c1=[256],
            c2=[160, 320],
            c3=[32, 128],
            c4=[64]
        )
        
        # Inception 5b (1024x7x7 -> 1024x7x7)
        self.inception5b = inception_block(
            1024,
            c1=[384],
            c2=[192, 384],
            c3=[48, 128],
            c4=[64]
        )
        
        # Global average pooling
        self.avgpool_final = nn.AvgPool2d(kernel_size=7, stride=1)
        
        # Final fully connected layer
        self.dropout_final = nn.Dropout(p=0.4)
        self.fc_final = nn.Linear(1024, num_classes)
    
    def forward(self, x, return_features=False):
        # Stem
        x = self.conv1_7x7(x)
        x = self.maxpool2(x)
        x = self.maxpool3(x)
        
        # Inception 2
        x = self.inception2a(x)
        x = self.inception2b(x)
        x = self.maxpool4(x)
        
        # Inception 3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool5(x)
        
        # Inception 4
        x = self.inception4a(x)
        x = self.inception4b(x)
        x_aux1 = self.avgpool1(x)
        x_aux1 = self.conv_aux1(x_aux1)
        x_aux1 = x_aux1.view(x_aux1.size(0), -1)
        x_aux1 = self.fc_aux1_1(x_aux1)
        x_aux1 = F.relu(x_aux1)
        x_aux1 = self.dropout_aux1(x_aux1)
        x_aux1 = self.fc_aux1_2(x_aux1)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x_aux2 = self.avgpool2(x)
        x_aux2 = self.conv_aux2(x_aux2)
        x_aux2 = x_aux2.view(x_aux2.size(0), -1)
        x_aux2 = self.fc_aux2_1(x_aux2)
        x_aux2 = F.relu(x_aux2)
        x_aux2 = self.dropout_aux2(x_aux2)
        x_aux2 = self.fc_aux2_2(x_aux2)
        x = self.maxpool6(x)
        
        # Inception 5
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # Final pooling and classification
        x = self.avgpool_final(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_final(x)
        x = self.fc_final(x)
        
        if return_features:
            return x, x_aux1, x_aux2
        return x
    
    def get_auxiliary_loss(self, x, labels, return_features=False):
        """Get auxiliary classifier outputs"""
        # Same forward pass but return auxiliary outputs
        x = self.conv1_7x7(x)
        x = self.maxpool2(x)
        x = self.maxpool3(x)
        x = self.inception2a(x)
        x = self.inception2b(x)
        x = self.maxpool4(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool5(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        
        # First auxiliary classifier
        x_aux1 = self.avgpool1(x)
        x_aux1 = self.conv_aux1(x_aux1)
        x_aux1 = x_aux1.view(x_aux1.size(0), -1)
        x_aux1 = self.fc_aux1_1(x_aux1)
        x_aux1 = F.relu(x_aux1)
        x_aux1 = self.dropout_aux1(x_aux1)
        x_aux1 = self.fc_aux1_2(x_aux1)
        
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool6(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool_final(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_final(x)
        x = self.fc_final(x)
        
        return x, x_aux1, x_aux2


# Example usage and model initialization
def create_googlenet(num_classes=1000):
    """Create GoogLeNet model with specified number of classes"""
    model = GoogLeNet(num_classes=num_classes)
    return model


if __name__ == "__main__":
    # Create model
    model = create_googlenet(num_classes=1000)
    
    # Test with dummy input
    x = torch.randn(1, 3, 224, 224)
    output, aux1, aux2 = model(x, return_features=True)
    
    print(f"Input size: {x.shape}")
    print(f"Main output size: {output.shape}")
    print(f"Auxiliary output 1 size: {aux1.shape}")
    print(f"Auxiliary output 2 size: {aux2.shape}")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Save model
    torch.save(model.state_dict(), 'googlenet.pth')
    print("Model saved as googlenet.pth")
