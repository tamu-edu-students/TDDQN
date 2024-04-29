import torch.nn as nn


class ResGate(nn.Module):
    """Residual skip connection"""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return x + y
