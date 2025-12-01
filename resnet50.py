import torch

class stem(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(stem, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    
class conv1x1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=True):
        super(conv1x1, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        if activation:
            self.relu = torch.nn.ReLU(inplace=True)
        else:
            self.relu = torch.nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv3x3(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(conv3x3, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class conv_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(conv_block, self).__init__()
        self.conv1 = conv1x1(in_channels, out_channels, stride)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * 4, activation=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = conv1x1(in_channels, out_channels * 4, stride, activation=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class identity_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(identity_block, self).__init__()
        self.conv1 = conv1x1(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * 4, activation=False)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += identity
        out = self.relu(out)
        return out
    
class ResNet50(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.stem = stem(3, 64)
        
        self.layer1 = torch.nn.Sequential(
            conv_block(64, 64, stride=1),
            identity_block(256, 64),
            identity_block(256, 64)
        )
        
        self.layer2 = torch.nn.Sequential(
            conv_block(256, 128, stride=2),
            identity_block(512, 128),
            identity_block(512, 128),
            identity_block(512, 128)
        )
        
        self.layer3 = torch.nn.Sequential(
            conv_block(512, 256, stride=2),
            identity_block(1024, 256),
            identity_block(1024, 256),
            identity_block(1024, 256),
            identity_block(1024, 256),
            identity_block(1024, 256)
        )
        self.layer4 = torch.nn.Sequential(
            conv_block(1024, 512, stride=2),
            identity_block(2048, 512),
            identity_block(2048, 512)
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50(num_classes=10).to(device)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}  Trainable: {trainable:,}")
    
    try:
        from torchinfo import summary
        summary(model, input_size=(1, 3, 224, 224), device=str(device))
    except Exception as e:
        print("torchinfo not available:", e)
        x = torch.randn(1, 3, 224, 224).to(device)
        y = model(x)
        print("Output shape:", y.shape)
