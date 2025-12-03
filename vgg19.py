import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, activation = True):
        super(ConvBlock, self).__init__()
        self.Conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.Bn = torch.nn.BatchNorm2d(out_channels)
        self.Act = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.Conv(x)
        x = self.Bn(x)
        x = self.Act(x)
        
        return x
    
    
class MaxPool(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(MaxPool, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0)
        
    def forward(self, x):
        x = self.maxpool(x)
        
        return x   
    
    
    
    
class Custom_VGG(torch.nn.Module):
    def __init__(self, num_classes = 10):
        super(Custom_VGG, self).__init__()
        
        self.MaxPool1 = MaxPool()
        
        self.layer1 = torch.nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
        )
        
        self.layer2 = torch.nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
        )
        
        self.layer3 = torch.nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
        )
        
        self.layer4 = torch.nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        )

        self.layer5 = torch.nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        )
        
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((7,7)),
            torch.nn.Flatten(),
            torch.nn.Linear(512*7*7, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, num_classes)
                        
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.MaxPool1(x)
        x = self.layer2(x)
        x = self.MaxPool1(x)
        x = self.layer3(x)
        x = self.MaxPool1(x)
        x = self.layer4(x)
        x = self.MaxPool1(x)
        x = self.layer5(x)
        x = self.MaxPool1(x)
        x = self.head(x)
        
        return x
        
if __name__ == "__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Custom_VGG(num_classes=10).to(device)
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
    
        