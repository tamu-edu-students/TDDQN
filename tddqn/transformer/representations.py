from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn


class ObservationEmbeddingRepresentation(nn.Module):
    def __init__(
        self,
        vocab_sizes: int, obs_dim: int, embed_per_obs_dim: int, outer_embed_size: int
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_sizes, embed_per_obs_dim),
            nn.Flatten(start_dim=-2),
            nn.Linear(embed_per_obs_dim * obs_dim, outer_embed_size),
        )

    def forward(self, obs: torch.Tensor):
        batch, seq = obs.size(0), obs.size(1)
        # Flatten batch and seq dims
        obs = torch.flatten(obs, start_dim=0, end_dim=1)
        obs_embed = self.embedding(obs)
        obs_embed = obs_embed.reshape(batch, seq, obs_embed.size(-1))
        return obs_embed