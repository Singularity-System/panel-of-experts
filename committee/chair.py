import copy
import torch
import torch.nn as nn
from typing import Optional


class Chair(nn.Module):
    """Chair: processes a single committee representation vector.

    Takes the compressed representation (B, D) and produces a refined
    representation through a single TransformerEncoderLayer.

    Args:
        d_model: Hidden dimension
        n_head: Number of attention heads
        d_ff: Feed-forward dimension
        num_layers: Number of layers (1 for pilot)
    """

    def __init__(self, d_model: int = 128, n_head: int = 4, d_ff: int = 256, num_layers: int = 1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff,
            batch_first=True, activation='gelu', norm_first=False)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) single committee representation vector

        Returns:
            (B, D) refined committee representation
        """
        # TransformerEncoderLayer expects (B, S, D), so add sequence dim
        x = x.unsqueeze(1)  # (B, 1, D)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x).squeeze(1)  # (B, D)
