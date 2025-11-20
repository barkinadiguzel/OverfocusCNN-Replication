import torch
import torch.nn as nn
from .conv_layer import ConvLayer
from .shortcut_layer import ShortcutLayer

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, stride=stride)
        self.conv2 = ConvLayer(out_channels, out_channels)
        self.shortcut = ShortcutLayer(in_channels, out_channels, stride)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.shortcut(x)
