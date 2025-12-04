import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True):
        super(ConvBlock, self).__init__()
        p = kernel_size//2
        self.Conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=p, bias=False)
        self.Bn = nn.BatchNorm2d(out_channels)
        self.Act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.Conv(x)
        x = self.Bn(x)
        x = self.Act(x)
        
        return x
    

class Custom_AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(Custom_AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            ConvBlock(3, 96, 11, 4),
            nn.MaxPool2d(3, 2),
        )
        self.layer2 = nn.Sequential(
            ConvBlock(96, 256, 5),
            nn.MaxPool2d(3,2)
        )
        self.layer3 = nn.Sequential(
            ConvBlock(256, 384, 3),
            ConvBlock(384,384,3),
            ConvBlock(384, 256, 3),
            nn.MaxPool2d(3,2)
        ) 
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((6,6)),
            nn.Flatten(),
            nn.Linear(6*6*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.head(x)
        
        return x
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Custom_AlexNet(num_classes=10).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for  p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} Trainable: {trainable:,}")
    
    try:
        from torchinfo import summary
        summary(model, input_size=(1, 3, 224, 224), device=str(device))   
    except Exception as e:
        print("torchinfo not available:", e)
        x = torch.randn(1, 3, 224, 224).to(device)
        y = model(x)
        print("Output shape:", y.shape)
        
    