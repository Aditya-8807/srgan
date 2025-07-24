import torch
import torch.nn as nn

class RRDB(nn.Module):
    def __init__(self, in_channels):
        super(RRDB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return x + out

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_rrdb=23):
        super(Generator, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.rrdb_blocks = nn.Sequential(*[RRDB(64) for _ in range(num_rrdb)])
        self.upsample1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        initial_feature = self.relu(self.initial_conv(x))
        out = self.rrdb_blocks(initial_feature)
        out = initial_feature + out
        out = self.upsample1(out)
        out = self.upsample2(out)
        out = self.final_conv(out)
        return out
