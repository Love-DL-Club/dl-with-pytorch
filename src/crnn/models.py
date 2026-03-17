import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 5), stride=(2, 1)):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
        )

        self.downsample = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        x_ = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x_ = self.downsample(x_)

        x += x_
        x = self.relu(x)

        return x


class CRNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.c1 = BasicBlock(in_channels=3, out_channels=64)
        self.c2 = BasicBlock(in_channels=64, out_channels=64)
        self.c3 = BasicBlock(in_channels=64, out_channels=64)
        self.c4 = BasicBlock(in_channels=64, out_channels=64)
        self.c5 = nn.Conv2d(64, 64, kernel_size=(2, 5))

        self.gru = nn.GRU(64, 64, batch_first=False)

        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)

        x = x.view(x.shape[0], 64, -1)
        x = x.permute(2, 0, 1)

        x, _ = self.gru(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        x = F.log_softmax(x, dim=-1)

        return x
