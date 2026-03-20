import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dims=256, channels=3):
        super().__init__()

        self.l1 = nn.Sequential(nn.Linear(dims, 128 * 8 * 8))

        self.conv_blocks0 = nn.Sequential(nn.BatchNorm2d(128))
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(channels, affine=False),
        )

    def forward(self, z):
        out = self.l1(z.view(z.shape[0], -1))
        out = out.view(out.shape[0], -1, 8, 8)

        out = self.conv_blocks0(out)
        out = nn.functional.interpolate(out, scale_factor=2)
        out = self.conv_blocks1(out)
        out = nn.functional.interpolate(out, scale_factor=2)
        out = self.conv_blocks2(out)

        return out
