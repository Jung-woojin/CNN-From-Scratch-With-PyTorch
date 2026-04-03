import torch
import torch.nn as nn


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
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


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.act = nn.ReLU(inplace=True)

        if deploy:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True,
            )
        else:
            self.branch_3x3 = ConvBnAct(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                activation=False,
            )
            self.branch_1x1 = ConvBnAct(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                activation=False,
            )
            if stride == 1 and in_channels == out_channels:
                self.branch_identity = nn.BatchNorm2d(in_channels)
            else:
                self.branch_identity = None

    def forward(self, x):
        if self.deploy:
            return self.act(self.reparam_conv(x))

        out = self.branch_3x3(x) + self.branch_1x1(x)
        if self.branch_identity is not None:
            out = out + self.branch_identity(x)
        out = self.act(out)
        return out


class RepVGG(nn.Module):
    def __init__(self, num_classes=10, deploy=False):
        super(RepVGG, self).__init__()
        self.deploy = deploy

        self.stage0 = RepVGGBlock(3, 64, stride=2, deploy=deploy)
        self.stage1 = self._make_stage(64, 64, num_blocks=2, stride=1, deploy=deploy)
        self.stage2 = self._make_stage(64, 128, num_blocks=4, stride=2, deploy=deploy)
        self.stage3 = self._make_stage(128, 256, num_blocks=8, stride=2, deploy=deploy)
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2, deploy=deploy)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, deploy):
        blocks = []
        for i in range(num_blocks):
            block_stride = stride if i == 0 else 1
            block_in = in_channels if i == 0 else out_channels
            blocks.append(RepVGGBlock(block_in, out_channels, stride=block_stride, deploy=deploy))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x


def create_repvgg(num_classes=10, deploy=False):
    return RepVGG(num_classes=num_classes, deploy=deploy)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_repvgg(num_classes=10, deploy=False).to(device)
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
