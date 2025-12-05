import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels):
        super(DenseLayer, self).__init__()
        self.DenseLayer  = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1, bias= False)
        )
        
    def forward(self, x):
        x = self.DenseLayer(x)
        return x
    
class DenseBlock(nn.Module):
    def __init__(self, in_channels, n, transition, transition_on = True):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        k = 32
        for i in range(n):
            layer_in = in_channels + k * i
            self.layers.append(DenseLayer(layer_in))
        if transition_on:    
            self.transition = nn.Sequential(
                nn.BatchNorm2d(in_channels + k *n),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels= in_channels + k *n, out_channels=int((in_channels + k *n)*transition), kernel_size=1, stride=1, padding=0, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
        else:
            self.transition=nn.Identity()  
    def forward(self, x):
        features = x
        for layer in self.layers:
            new_feature = layer(features)
            features = torch.cat([features, new_feature], dim=1)
        features = self.transition(features)
        return features
        
        
class Custom_DenseNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(Custom_DenseNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = DenseBlock(64, 6, 0.5)
        self.layer2 = DenseBlock(128, 12, 0.5)
        self.layer3 = DenseBlock(256, 24, 0.5)
        self.layer4 = DenseBlock(512, 16, 1, False)
        self.head = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes)
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
    model = Custom_DenseNet(num_classes=10).to(device)
    print(model)
    
    try:
        from torchinfo import summary
        summary(model, input_size=(1, 3, 244, 244), device=str(device))
    except Exception as e:
        print("torchinfo not available:", e)