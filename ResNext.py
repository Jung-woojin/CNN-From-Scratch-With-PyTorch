import torch.nn as nn
import torch

class Resnext_Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resnext_Downsample, self).__init__()
        card = out_channels//2
        card = max(32, (card // 32) * 32)
        self.Resnext_Block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=card, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(card),
            nn.SiLU(),
            nn.Conv2d(in_channels=card, out_channels=card, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(card),
            nn.SiLU(),
            nn.Conv2d(in_channels=card, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),

        )
        

        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0, bias=False)
        self.projection_bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x):
        identity = x
        x = self.Resnext_Block(x)
        identity = self.projection(identity)
        identity = self.projection_bn(identity)
        x += identity
        x = self.act(x)
        return x



class Resnext_Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, width):
        super(Resnext_Bottleneck, self).__init__()
        card = int(in_channels * width)
        card = max(32, (card // 32) * 32)
        self.Resnext_Block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=card, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(card),
            nn.SiLU(),
            nn.Conv2d(in_channels=card, out_channels=card, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(card),
            nn.SiLU(),
            nn.Conv2d(in_channels=card, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),

        )
        self.projection_bn = nn.Identity()
        self.projection = nn.Identity()
        if  in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.projection_bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x):
        identity = x
        x = self.Resnext_Block(x)
        identity = self.projection(identity)
        identity = self.projection_bn(identity)
        x += identity
        x = self.act(x)
        return x
    

class ResNext(nn.Module):
    def __init__(self, num_classes = 10):
        super(ResNext, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)    
        )
        self.layer1 = nn.Sequential(
            Resnext_Bottleneck(64,256,width=2),
            Resnext_Bottleneck(256,256,width=0.5),
            Resnext_Bottleneck(256,256,width=0.5)
        )
        self.layer2 = nn.Sequential(
            Resnext_Downsample(256,512),
            Resnext_Bottleneck(512,512,width=0.5),
            Resnext_Bottleneck(512,512,width=0.5),
            Resnext_Bottleneck(512,512,width=0.5), 
        )
        self.layer3 = nn.Sequential(
            Resnext_Downsample(512,1024),
            Resnext_Bottleneck(1024,1024,width=0.5),
            Resnext_Bottleneck(1024,1024,width=0.5),
            Resnext_Bottleneck(1024,1024,width=0.5), 
        )
        self.layer4 = nn.Sequential(
            Resnext_Downsample(1024,2048),
            Resnext_Bottleneck(2048,2048,width=0.5),
            Resnext_Bottleneck(2048,2048,width=0.5),
            Resnext_Bottleneck(2048,2048,width=0.5), 
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 2048),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 4096),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        
        return x
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNext(num_classes=10).to(device)
    print(model)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {params:,}, Trainable: {trainable:,} ")
    
    try:
        from torchinfo import summary
        summary(model, (1, 3, 224, 224), device=str(device))
    except Exception as e:
        print("torchinfo not available", e)
        x = torch.randn(1, 3, 244, 244).to(device)
        y = model(x)
        print("Ouptut Shape:", y.shape)
        
    