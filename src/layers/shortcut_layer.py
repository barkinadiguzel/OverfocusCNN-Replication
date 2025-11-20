import torch
import torch.nn as nn

class ShortcutLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.use_conv = in_channels != out_channels or stride != 1
        if self.use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.use_conv:
            return self.bn(self.conv(x))
        return x
