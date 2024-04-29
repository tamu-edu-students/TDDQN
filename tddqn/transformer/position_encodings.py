from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


class PositionEncoding(nn.Module):
    def __init__(self, context_len: int, embed_dim: int):
        super().__init__()
        position = torch.arange(context_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim)
        )
        pos_encoding = torch.zeros(1, context_len, embed_dim)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.position_encoding = nn.Parameter(pos_encoding, requires_grad=False)

    def forward(self):
        return self.position_encoding
