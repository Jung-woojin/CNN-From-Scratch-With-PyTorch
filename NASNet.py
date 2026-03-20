# NASNet-A (Mobile) - PyTorch Implementation
# Original NASNet (2018): "Learning Transferable Architectures from Scratch"


import torch
import torch.nn as nn
import torch.nn.functional as F


class ReusableCell(nn.Module):
    """Reusable Cell - Basic building block of NASNet
    
    Repeats this cell multiple times to create the full architecture.
    Each cell has multiple operations that are concatenated together.
    
    Args:
        num_concat: Number of concatenated branches
        stem_filters: Number of filters in the stem
    """
    
    def __init__(self, num_concat, stem_filters):
        super(ReusableCell, self).__init__()
        
        self.num_concat = num_concat
        self.stem_filters = stem_filters
        
        # Define multiple operations for each branch
        self.operations = nn.ModuleList()
        
        # Operations include:
        # - MaxPooling3x3
        # - SeparableConv3x3
        # - SeparableConv5x5
        # - AveragePooling3x3
        # - Skip connection
        
        # Create operations for each input
        for i in range(num_concat):
            ops = nn.ModuleList()
            
            # Skip connection
            ops.append(nn.Identity())
            
            # 3x3 max pooling
            ops.append(nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(stem_filters, stem_filters, kernel_size=1, bias=False)
            ))
            
            # 3x3 separable convolution
            ops.append(nn.Sequential(
                nn.Conv2d(stem_filters, stem_filters, kernel_size=3, stride=1, padding=1, groups=stem_filters, bias=False),
                nn.BatchNorm2d(stem_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_filters, stem_filters, kernel_size=1, bias=False),
                nn.BatchNorm2d(stem_filters)
            ))
            
            # 5x5 separable convolution
            ops.append(nn.Sequential(
                nn.Conv2d(stem_filters, stem_filters, kernel_size=5, stride=1, padding=2, groups=stem_filters, bias=False),
                nn.BatchNorm2d(stem_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_filters, stem_filters, kernel_size=1, bias=False),
                nn.BatchNorm2d(stem_filters)
            ))
            
            # 3x3 average pooling
            ops.append(nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1),
                nn.Conv2d(stem_filters, stem_filters, kernel_size=1, bias=False)
            ))
            
            self.operations.append(ops)
    
    def forward(self, inputs):
        """Forward pass with skip connections"""
        outputs = []
        
        for i, ops in enumerate(self.operations):
            # Skip connection to first input
            outputs.append(inputs[i])
            
            # Apply operations to subsequent inputs
            for j in range(i + 1, len(inputs)):
                op = ops[j - (i + 1) if j > i + 1 else j]
                x = op(inputs[j])
                outputs.append(x)
        
        # Concatenate all outputs
        return torch.cat(outputs, dim=1)


class Aggregation(nn.Module):
    """Aggregation Block - Combines outputs from multiple cells
    
    Performs skip connections and concatenation across different cell outputs
    
    Args:
        filters: Number of filters
        reduction: Filter reduction factor
    """
    
    def __init__(self, filters, reduction=4):
        super(Aggregation, self).__init__()
        
        self.filters = filters
        self.reduction = reduction
        
        # 1x1 convolution to reduce channels
        self.conv = nn.Conv2d(filters, filters // reduction, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(filters // reduction)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, inputs):
        # Apply skip connections and averaging
        output = sum(inputs) / len(inputs)
        output = self.conv(output)
        output = self.bn(output)
        output = self.relu(output)
        
        return output


class Stem(nn.Module):
    """Stem Block - Initial layers of NASNet
    
    Structure:
    1. 1x1 convolution
    2. 3x3 separable convolution
    3. 3x3 max pooling
    
    Args:
        num_filters: Number of filters
    """
    
    def __init__(self, num_filters):
        super(Stem, self).__init__()
        
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, 
                              stride=1, padding=1, groups=num_filters, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.pool(x)
        
        return x


class NASNetANMobile(nn.Module):
    """NASNet-A Mobile Architecture
    
    Key Features:
    - Reusable building blocks (aggregation, cells)
    - Maximum-pooling layers
    - Separable depthwise convolutions
    - Skip connections throughout
    - Automatically learned architecture
    
    Mobile version parameters:
    - 18 cells total
    - 384 output channels
    - ~5.4M parameters
    
    Args:
        num_classes: Number of output classes (default 1000)
    """
    
    def __init__(self, num_classes=1000):
        super(NASNetANMobile, self).__init__()
        
        self.num_classes = num_classes
        
        # Stem
        self.stem = Stem(62)
        
        # Number of filters at each layer
        # [62, 124, 244, 484, 764]
        self.filters = [62, 124, 244, 484, 764]
        
        # Cells (reusable structure)
        # Each cell has 4 concatenated branches
        self.cells = nn.ModuleList()
        
        # Cell 1: 18 repeats
        for i in range(18):
            self.cells.append(ReusableCell(num_concat=4, stem_filters=62))
        
        # Reduction layers between cells
        self.reductions = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(3, stride=2),
                nn.Conv2d(62, 124, kernel_size=1, bias=False),
                nn.BatchNorm2d(124),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Cell 2: 18 repeats
        for i in range(18):
            self.cells.append(ReusableCell(num_concat=4, stem_filters=124))
        
        self.reductions.append(nn.Sequential(
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(124, 244, kernel_size=1, bias=False),
            nn.BatchNorm2d(244),
            nn.ReLU(inplace=True)
        ))
        
        # Cell 3: 18 repeats
        for i in range(18):
            self.cells.append(ReusableCell(num_concat=4, stem_filters=244))
        
        self.reductions.append(nn.Sequential(
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(244, 484, kernel_size=1, bias=False),
            nn.BatchNorm2d(484),
            nn.ReLU(inplace=True)
        ))
        
        # Final aggregation
        self.aggregation = Aggregation(484, reduction=4)
        
        # Final classifier
        self.bn_final = nn.BatchNorm2d(121)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(121, num_classes)
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Process through cells with reductions
        outputs = []
        cell_idx = 0
        reduction_idx = 0
        
        for i in range(54):  # 18 cells * 3 stages
            if i > 0 and i % 18 == 0:
                # Apply reduction
                x = self.reductions[reduction_idx](x)
                outputs.append(x)
                reduction_idx += 1
            
            # Apply cell
            outputs.append(self.cells[cell_idx](outputs if cell_idx == 0 else [outputs[-1]]))
            cell_idx += 1
        
        # Final aggregation
        x = self.aggregation(outputs[-5:])
        
        # Final processing
        x = self.bn_final(x)
        x = F.relu(x, inplace=True)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# Factory function
def create_nasnet_mobile(num_classes=1000):
    """Create NASNet-A Mobile model with specified number of classes"""
    model = NASNetANMobile(num_classes=num_classes)
    return model


if __name__ == "__main__":
    # Create model
    model = create_nasnet_mobile(num_classes=1000)
    
    # Test with dummy input
    x = torch.randn(1, 3, 331, 331)  # NASNet expects larger input
    output = model(x)
    
    print(f"Input size: {x.shape}")
    print(f"Output size: {output.shape}")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Save model
    torch.save(model.state_dict(), 'nasnet_mobile.pth')
    print("Model saved as nasnet_mobile.pth")
