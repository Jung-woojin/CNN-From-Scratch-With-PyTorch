import torch
import torch.nn as nn


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBnAct, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(InceptionModule, self).__init__()
        self.branch1 = ConvBnAct(in_channels, c1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBnAct(in_channels, c2[0], kernel_size=1),
            ConvBnAct(c2[0], c2[1], kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBnAct(in_channels, c3[0], kernel_size=1),
            ConvBnAct(c3[0], c3[1], kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnAct(in_channels, c4, kernel_size=1),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            ConvBnAct(in_channels, 128, kernel_size=1),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.stem = nn.Sequential(
            ConvBnAct(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBnAct(64, 64, kernel_size=1),
            ConvBnAct(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.inception3a = InceptionModule(192, 64, (96, 128), (16, 32), 32)   # 256
        self.inception3b = InceptionModule(256, 128, (128, 192), (32, 96), 64)  # 480
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, (96, 208), (16, 48), 64)   # 512
        self.inception4b = InceptionModule(512, 160, (112, 224), (24, 64), 64)  # 512
        self.inception4c = InceptionModule(512, 128, (128, 256), (24, 64), 64)  # 512
        self.inception4d = InceptionModule(512, 112, (144, 288), (32, 64), 64)  # 528
        self.inception4e = InceptionModule(528, 256, (160, 320), (32, 128), 128)  # 832
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, (160, 320), (32, 128), 128)  # 832
        self.inception5b = InceptionModule(832, 384, (192, 384), (48, 128), 128)  # 1024

        if aux_logits:
            self.aux1 = AuxClassifier(512, num_classes)
            self.aux2 = AuxClassifier(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        aux1 = self.aux1(x) if self.aux1 is not None and self.training else None
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if self.aux2 is not None and self.training else None
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.head(x)

        if self.aux_logits and self.training:
            return x, aux1, aux2
        return x


def create_googlenet(num_classes=1000, aux_logits=False):
    return GoogLeNet(num_classes=num_classes, aux_logits=aux_logits)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_googlenet(num_classes=1000, aux_logits=True).to(device)
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
        y = model.eval()(x)
        print("Output shape:", y.shape)
