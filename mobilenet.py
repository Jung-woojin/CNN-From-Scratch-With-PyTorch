import torch
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        if activation == True:
            self.act = torch.nn.ReLU(inplace=True)
        else:
            self.act = torch.nn.Identity()
            
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
    
        return x
    
class DWSConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, expansion=2, activation=True):
        super(DWSConv, self).__init__()
        #만약 ConvBlock을 불러올 때 에러가 있으면 아래 구조를 사용
        #self.conv1 = torch.nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, stride=stride, padding=padding, bias=False)
        #self.bn1= torch.nn.BatchNorm2d(out_channels*expansion)
        #self.act1 = torch.nn.ReLU(out_channels*expansion 
        pad = kernel_size//2
        self.conv1_1 = ConvBlock(in_channels, in_channels*expansion, kernel_size=1, stride=1)
        if stride == 2:
            self.DWC = torch.nn.Conv2d(in_channels * expansion, in_channels*expansion, kernel_size, stride=2, padding=pad, groups=in_channels*expansion, bias=False)
        else: 
            self.DWC = torch.nn.Conv2d(in_channels * expansion, in_channels*expansion, kernel_size, stride=1, padding=pad, groups=in_channels*expansion, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(in_channels*expansion)
        self.act1 = torch.nn.ReLU(inplace=True)
        self.conv1_2 = ConvBlock(in_channels*expansion, out_channels, kernel_size=1, stride=1, activation=False)
        
        
    def forward(self, x):
        identity = x
        x = self.conv1_1(x)
        x = self.DWC(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1_2(x)
        if identity.shape ==x.shape:
            x +=identity
        
        return x    
    
    
class Custom_MobileNet_V2(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = ConvBlock(3, 32, 3, stride=2)
        
        self.layer1 = torch.nn.Sequential(
            DWSConv(32, 16, 3),
        )
        
        self.layer2 = torch.nn.Sequential(
            DWSConv(16, 24, 3, stride=2, expansion=4),
            DWSConv(24, 24, 3, expansion=4),
        )
        
        self.layer3 = torch.nn.Sequential(
            DWSConv(24, 32, 3, stride=2, expansion=4),
            DWSConv(32, 32, 3, expansion=4),
            DWSConv(32, 32, 3, expansion=4),
        )
        
        self.layer4 = torch.nn.Sequential(
            DWSConv(32, 64, 3, stride=2, expansion=4),
            DWSConv(64, 64, 3, expansion=4),
            DWSConv(64, 64, 3, expansion=4),
        )    
            
        self.layer5 = torch.nn.Sequential(
            DWSConv(64, 96, 3, expansion=4),
            DWSConv(96, 96, 3, expansion=4),
            DWSConv(96, 96, 3, expansion=4),
        )
        
        self.layer6 = torch.nn.Sequential(
            DWSConv(96, 160, 3, stride=2, expansion=4),
            DWSConv(160, 160, 3, expansion=4),
            DWSConv(160, 160, 3, expansion=4),
        )
        
        self.layer7 = torch.nn.Sequential(
            DWSConv(160,320,3, expansion=4),
            DWSConv(320,320,3, expansion=4),
            DWSConv(320,320,3, expansion=4),
        )
        
        self.head = torch.nn.Sequential(
            ConvBlock(320,1280, 7),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.head(x)

        return x
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Custom_MobileNet_V2(num_classes=10).to(device)
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
        
        
        
        
        
        
        
        
        
        
    
    