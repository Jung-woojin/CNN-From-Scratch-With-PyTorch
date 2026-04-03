import torch
import torch.nn as nn


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, activation=True):
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
        if activation:
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()

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
            nn.Conv2d(channels, squeezed, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeezed, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, kernel_size, stride):
        super(MBConv, self).__init__()
        hidden = in_channels * expansion
        self.use_skip = stride == 1 and in_channels == out_channels

        blocks = []
        if expansion != 1:
            blocks.append(ConvBnAct(in_channels, hidden, kernel_size=1, stride=1))
        else:
            hidden = in_channels
        blocks.append(ConvBnAct(hidden, hidden, kernel_size=kernel_size, stride=stride, groups=hidden))
        blocks.append(SEBlock(hidden, reduction=4))
        blocks.append(ConvBnAct(hidden, out_channels, kernel_size=1, stride=1, activation=False))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            out = out + x
        return out


class Custom_EfficientNet_B0(nn.Module):
    def __init__(self, num_classes=10):
        super(Custom_EfficientNet_B0, self).__init__()
        self.stem = ConvBnAct(3, 32, kernel_size=3, stride=2)

        cfg = [
            # expansion, channels, repeats, stride, kernel
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ]

        layers = []
        in_channels = 32
        for expansion, out_channels, repeats, stride, kernel in cfg:
            for i in range(repeats):
                block_stride = stride if i == 0 else 1
                layers.append(MBConv(in_channels, out_channels, expansion, kernel, block_stride))
                in_channels = out_channels
        self.blocks = nn.Sequential(*layers)

        self.head = nn.Sequential(
            ConvBnAct(in_channels, 1280, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def create_efficientnet_b0(num_classes=10):
    return Custom_EfficientNet_B0(num_classes=num_classes)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_efficientnet_b0(num_classes=10).to(device)
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
