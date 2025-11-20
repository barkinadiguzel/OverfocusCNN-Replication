import torch.nn as nn

class AvgPoolLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        return self.pool(x)
