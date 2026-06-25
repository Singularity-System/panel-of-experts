import copy
import torch
import torch.nn as nn


class Chair(nn.Module):
    """Chair: fuses N expert vectors via self-attention.

    Takes (B, N, D) — N expert vectors — and fuses them via self-attention
    on the N dimension, then aggregates to (B, D).

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
            x: (B, N, D) N expert vectors per sample

        Returns:
            (B, D) aggregated committee representation
        """
        # Self-attention across N experts
        for layer in self.layers:
            x = layer(x)  # (B, N, D) — attention across expert dimension
        # Aggregate: mean over experts
        x = x.mean(dim=1)  # (B, D)
        return self.norm(x)
