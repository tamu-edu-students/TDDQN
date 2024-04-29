import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
from tddqn.networks.gates import ResGate
from tddqn.transformer.transformer import TransformerLayer
from utils import torch_utils
from tddqn.transformer.representations import ObservationEmbeddingRepresentation
from tddqn.transformer.position_encodings import PositionEncoding


class TDDQN(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        embed_per_obs_dim: int,
        action_dim: int,
        inner_embed_size: int,
        num_heads: int,
        num_layers: int,
        history_len: int,
        dropout: float = 0.0,
        discrete: bool = False,
        vocab_sizes: Optional[Union[np.ndarray, int]] = None,
        **kwargs,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.discrete = discrete
        obs_output_dim = inner_embed_size - action_dim

        self.action_embedding = None 

        self.obs_embedding = (
            ObservationEmbeddingRepresentation(
                vocab_sizes=vocab_sizes,
                obs_dim=obs_dim,
                embed_per_obs_dim=embed_per_obs_dim,
                outer_embed_size=obs_output_dim,
            )
        )

        self.position_embedding = PositionEncoding(context_len=history_len, embed_dim=inner_embed_size)
        self.dropout = nn.Dropout(dropout)

        attn_gate = ResGate()
        mlp_gate = ResGate()

        transformer_block = TransformerLayer
        self.transformer_layers = nn.Sequential(
            *[
                transformer_block(num_heads, inner_embed_size, history_len, dropout, attn_gate, mlp_gate,)
                for _ in range(num_layers)
            ]
        )

        self.ffn = nn.Sequential(
            nn.Linear(inner_embed_size, inner_embed_size),
            nn.ReLU(),
            nn.Linear(inner_embed_size, num_actions),
        )
        self.history_len = history_len
        self.apply(torch_utils.init_weights)

    def forward(self, obss: torch.Tensor) -> torch.Tensor:
        history_len = obss.size(1)
        token_embeddings = self.obs_embedding(obss)
        working_memory = self.transformer_layers(self.dropout(token_embeddings + self.position_embedding()[:, :history_len, :]))
        output = self.ffn(working_memory)
        return output[:, -history_len:, :]
