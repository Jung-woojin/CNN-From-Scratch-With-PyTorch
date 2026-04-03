import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(SeparableConv, self).__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=pad, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class NASCell(nn.Module):
    def __init__(self, in_prev, in_curr, out_channels, reduction=False):
        super(NASCell, self).__init__()
        stride = 2 if reduction else 1
        self.preprocess_prev = ConvBnAct(in_prev, out_channels, kernel_size=1, stride=stride, padding=0)
        self.preprocess_curr = ConvBnAct(in_curr, out_channels, kernel_size=1, stride=stride, padding=0)

        self.b1_left = SeparableConv(out_channels, out_channels, kernel_size=5, stride=1)
        self.b1_right = SeparableConv(out_channels, out_channels, kernel_size=3, stride=1)

        self.b2_left = SeparableConv(out_channels, out_channels, kernel_size=3, stride=1)
        self.b2_right = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.b3_left = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b3_right = SeparableConv(out_channels, out_channels, kernel_size=3, stride=1)

        self.b4_left = SeparableConv(out_channels, out_channels, kernel_size=3, stride=1)
        self.b4_right = SeparableConv(out_channels, out_channels, kernel_size=5, stride=1)

        self.b5_left = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.b5_right = nn.Identity()

    def forward(self, prev, curr):
        prev = self.preprocess_prev(prev)
        curr = self.preprocess_curr(curr)
        if prev.shape[-2:] != curr.shape[-2:]:
            prev = F.interpolate(prev, size=curr.shape[-2:], mode="nearest")

        h1 = self.b1_left(curr) + self.b1_right(prev)
        h2 = self.b2_left(prev) + self.b2_right(curr)
        h3 = self.b3_left(curr) + self.b3_right(prev)
        h4 = self.b4_left(prev) + self.b4_right(curr)
        h5 = self.b5_left(h2) + self.b5_right(h1)

        out = torch.cat([h1, h2, h3, h4, h5], dim=1)
        return out


class Custom_NASNet_A_Mobile(nn.Module):
    def __init__(self, num_classes=10):
        super(Custom_NASNet_A_Mobile, self).__init__()
        self.stem = nn.Sequential(
            ConvBnAct(3, 32, kernel_size=3, stride=2, padding=1),
            ConvBnAct(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.cell1 = NASCell(32, 32, out_channels=24, reduction=True)
        self.cell2 = NASCell(32, 120, out_channels=24, reduction=False)
        self.cell3 = NASCell(120, 120, out_channels=48, reduction=True)
        self.cell4 = NASCell(120, 240, out_channels=48, reduction=False)
        self.cell5 = NASCell(240, 240, out_channels=48, reduction=False)

        self.head = nn.Sequential(
            ConvBnAct(240, 1280, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        prev = x
        curr = x

        new = self.cell1(prev, curr)
        prev, curr = curr, new

        new = self.cell2(prev, curr)
        prev, curr = curr, new

        new = self.cell3(prev, curr)
        prev, curr = curr, new

        new = self.cell4(prev, curr)
        prev, curr = curr, new

        new = self.cell5(prev, curr)
        curr = new
        x = self.head(curr)
        return x


def create_nasnet_mobile(num_classes=10):
    return Custom_NASNet_A_Mobile(num_classes=num_classes)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_nasnet_mobile(num_classes=10).to(device)
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
