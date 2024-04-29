import torch.nn as nn
import torch
from typing import Optional
from tddqn.transformer.representations import ObservationEmbeddingRepresentation
from utils import torch_utils



class DQN(nn.Module):
    """DQN https://www.nature.com/articles/nature14236.pdf"""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        embed_per_obs_dim: int,
        action_dim: int,
        inner_embed_size: int,
        obs_vocab_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.obs_embed = (
            ObservationEmbeddingRepresentation(
                vocab_sizes=obs_vocab_size,
                obs_dim=obs_dim,
                embed_per_obs_dim=embed_per_obs_dim,
                outer_embed_size=inner_embed_size,
            )
        )

        self.ffn = nn.Sequential(
            nn.Linear(inner_embed_size, inner_embed_size),
            nn.ReLU(),
            nn.Linear(inner_embed_size, num_actions),
        )
        self.apply(torch_utils.init_weights)

    def forward(self, x: torch.tensor):
        return self.ffn(self.obs_embed(x))

