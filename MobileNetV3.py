import torch
import torch.nn as nn


class HSigmoid(nn.Module):
    def forward(self, x):
        return torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0


class HSwish(nn.Module):
    def __init__(self):
        super(HSwish, self).__init__()
        self.hsigmoid = HSigmoid()

    def forward(self, x):
        return x * self.hsigmoid(x)


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, use_hswish=False):
        super(ConvBnAct, self).__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if use_hswish:
            self.act = HSwish()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        squeezed = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, squeezed, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeezed, channels, kernel_size=1),
            HSigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expansion, stride, use_se, use_hswish):
        super(InvertedResidual, self).__init__()
        hidden = in_channels * expansion
        self.use_skip = stride == 1 and in_channels == out_channels

        layers = [
            ConvBnAct(in_channels, hidden, kernel_size=1, stride=1, use_hswish=use_hswish),
            ConvBnAct(hidden, hidden, kernel_size=kernel_size, stride=stride, groups=hidden, use_hswish=use_hswish),
        ]
        if use_se:
            layers.append(SEBlock(hidden))
        layers.extend(
            [
                nn.Conv2d(hidden, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            out = out + x
        return out


class Custom_MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=10):
        super(Custom_MobileNetV3_Large, self).__init__()
        self.stem = ConvBnAct(3, 16, kernel_size=3, stride=2, use_hswish=True)

        cfg = [
            # k, exp, out, se, hs, stride
            (3, 1, 16, False, False, 1),
            (3, 4, 24, False, False, 2),
            (3, 3, 24, False, False, 1),
            (5, 3, 40, True, False, 2),
            (5, 3, 40, True, False, 1),
            (5, 3, 40, True, False, 1),
            (3, 6, 80, False, True, 2),
            (3, 2, 80, False, True, 1),
            (3, 2, 80, False, True, 1),
            (3, 2, 80, False, True, 1),
            (3, 6, 112, True, True, 1),
            (3, 6, 112, True, True, 1),
            (5, 6, 160, True, True, 2),
            (5, 6, 160, True, True, 1),
            (5, 6, 160, True, True, 1),
        ]

        layers = []
        in_channels = 16
        for k, exp, out_channels, use_se, use_hs, stride in cfg:
            layers.append(InvertedResidual(in_channels, out_channels, k, exp, stride, use_se, use_hs))
            in_channels = out_channels
        self.blocks = nn.Sequential(*layers)

        self.head = nn.Sequential(
            ConvBnAct(in_channels, 960, kernel_size=1, stride=1, use_hswish=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0),
            HSwish(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def create_mobilenetv3_large(num_classes=10):
    return Custom_MobileNetV3_Large(num_classes=num_classes)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_mobilenetv3_large(num_classes=10).to(device)
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
