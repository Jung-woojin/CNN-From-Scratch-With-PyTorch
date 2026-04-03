import torch.nn as nn
import torch

class DWConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride =1, padding =1, Activation=True):
        super(DWConv, self).__init__()
        p = kernel_size//2
        self.DWC = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=p, groups=in_channels, bias=False)
        
    def forward(self, x):
        x = self.DWC(x)
        
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, Activation = True):
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
    
class Xeception_Bottleneck_stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Xeception_Bottleneck_stem, self).__init__()
        self.Branch1 = nn.Sequential(
            DWConv(in_channels=in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DWConv(in_channels=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.Branch2 = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(out_channels)
        )

        
    def forward(self, x):
        Branch1 = self.Branch1(x)
        Branch2 = self.Branch2(x)
        x = Branch1+Branch2
        return x 
    
    
    
    
    
    
    
class Xeception_Bottleneck_main(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Xeception_Bottleneck_main, self).__init__()
        self.Branch1 = nn.Sequential(
            nn.ReLU(inplace=True),
            DWConv(in_channels=in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DWConv(in_channels=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.Branch2 = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(out_channels)
        )

        
    def forward(self, x):
        Branch1 = self.Branch1(x)
        Branch2 = self.Branch2(x)
        x = Branch1+Branch2
        return x
    
    
    
class Xeception_Bottleneck_residual(nn.Module):
    def __init__(self, in_channels):
        super(Xeception_Bottleneck_residual, self).__init__()
        self.Branch1 = nn.Sequential(
            nn.ReLU(inplace=True),
            DWConv(in_channels=in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            DWConv(in_channels=in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            DWConv(in_channels=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        
    def forward(self, x):
        identity = x
        x = self.Branch1(x)
        x += identity
        return x
        
    
class Custom_xception(nn.Module):
    def __init__(self, num_classes):
        super(Custom_xception, self).__init__()
        self.stem = nn.Sequential(
            ConvBlock(3, 32, 3, 2),
            ConvBlock(32, 64)
        )
        self.layer1 = Xeception_Bottleneck_stem(64, 128)
        self.layer2 = Xeception_Bottleneck_main(128, 256)
        self.layer3 = Xeception_Bottleneck_main(256,728)
        self.layer4 =nn.Sequential(
            Xeception_Bottleneck_residual(728),
            Xeception_Bottleneck_residual(728),
            Xeception_Bottleneck_residual(728),
            Xeception_Bottleneck_residual(728),
            Xeception_Bottleneck_residual(728),
            Xeception_Bottleneck_residual(728),
            Xeception_Bottleneck_residual(728),
            Xeception_Bottleneck_residual(728),
        )
        self.layer5 = Xeception_Bottleneck_main(728, 1024)
        self.head = nn.Sequential(
            DWConv(1024),
            nn.Conv2d(in_channels=1024, out_channels=1536, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            DWConv(1536),
            nn.Conv2d(in_channels=1536, out_channels=2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.head(x)
        
        return x
    
if __name__ == "__main__":
    from torchinfo import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Custom_xception(num_classes=10).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}  Trainable: {trainable:,}")
    
    try:  
        summary(model, input_size=(1, 3, 224, 224), device=str(device)) 
    except Exception as e:
        print("torchinfo not available:", e)
        x = torch.randn(1, 3, 224, 224).to(device)
        y = model(x)
        print("Output Shape:", y.shape)
            