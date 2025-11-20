import torch
import torch.nn as nn
from ..layers.conv_layer import ConvLayer
from ..layers.residual_block import ResidualBlock
from ..layers.pool_layers.maxpool_layer import MaxPoolLayer
from ..layers.pool_layers.avgpool_layer import AvgPoolLayer
from ..layers.flatten_layer import FlattenLayer
from ..layers.fc_layer import FCLayer
from ..layers.saliency_regularization import SGDrop

class OverfocusCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = ConvLayer(3, 64)
        self.res1 = ResidualBlock(64, 128)
        self.pool1 = MaxPoolLayer()
        self.res2 = ResidualBlock(128, 256)
        self.pool2 = MaxPoolLayer()
        self.flatten = FlattenLayer()
        self.fc = FCLayer(256, num_classes)
        self.sgdrop = SGDrop(rho=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.pool1(x)
        x = self.res2(x)
        x = self.pool2(x)
        x = self.sgdrop(x, class_scores=None)  # placeholder, class_scores eğitime göre verilecek
        x = self.flatten(x)
        x = self.fc(x)
        return x
