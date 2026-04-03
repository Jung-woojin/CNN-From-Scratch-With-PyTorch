import torch
import torchinfo

class convblcok(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=True):
        super(convblcok, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
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

class Maxpool(torch.nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super(Maxpool, self).__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        
    def forward(self, x):
        x = self.pool(x)
        return x
    
class inception(torch.nn.Module):
    def __init__(self, in_channels, branch1_out1, branch2_out1, branch2_out2, branch3_out1, branch3_out2, branch3_out3, branch4_out1, out_channels):
        super(inception, self).__init__()
        self.branch1 = convblcok(in_channels, branch1_out1, kernel_size=1)

        self.branch2 = torch.nn.Sequential(
            convblcok(in_channels, branch2_out1, kernel_size=1),
            convblcok(branch2_out1, branch2_out2, kernel_size=3, padding=1)
        )

        self.branch3 = torch.nn.Sequential(
            convblcok(in_channels, branch3_out1, kernel_size=1),
            convblcok(branch3_out1, branch3_out2, kernel_size=3, padding=1),
            convblcok(branch3_out2, branch3_out3, kernel_size=3, padding=1)
        )

        self.branch4 = torch.nn.Sequential(
            Maxpool(kernel_size=3, stride=1, padding=1),
            convblcok(in_channels, branch4_out1, kernel_size=1)
        )
        self.projection = convblcok(branch1_out1 + branch2_out2 + branch3_out3 + branch4_out1, out_channels, kernel_size=1, activation=False)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        outputs = self.projection(torch.cat(outputs, 1))
        if identity.shape == outputs.shape:
            outputs += identity
        outputs = self.relu(outputs)
        return outputs
    
class Custom_Inception_Net(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = torch.nn.Sequential(
            convblcok(3, 64, 3, stride=2, padding=1),
            convblcok(64, 64, 3, padding=1),
            convblcok(64, 128, 3, padding=1),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
        )
        
        self.stage1 = torch.nn.Sequential(
            inception(128,32,32,32,32,48,64,32,128),
            inception(128,32,32,32,32,48,64,32,128),
        )
        
        self.reduction1 = torch.nn.Sequential(
            convblcok(128, 256, 3, stride=2, padding=1)
        )
        
        self.stage2 = torch.nn.Sequential(
            inception(256,64,64,64,64,96,128,64,256),
            inception(256,64,64,64,64,96,128,64,256),
        )
        
        self.reduction2 = torch.nn.Sequential(
            convblcok(256, 512, 3, stride=2, padding=1)
        )
        
        self.stage3 = torch.nn.Sequential(
            inception(512,128,128,128,128,192,256,128,512),
            inception(512,128,128,128,128,192,256,128,512),
        )
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.reduction1(x)
        x = self.stage2(x)
        x = self.reduction2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Custom_Inception_Net(num_classes=10).to(device)
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
