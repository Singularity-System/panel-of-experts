import copy
import torch
import torch.nn as nn


class Chair(nn.Module):
    """Chair: fuses N expert vectors per position.

    Takes (B, S, N, D) and fuses N experts at each position via self-attention,
    outputting (B, S, D). Preserves sequence dimension for LM training.

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
            x: (B, S, N, D) N expert vectors per (B, S) position

        Returns:
            (B, S, D) fused representation per position
        """
        B, S, N, D = x.shape
        # Reshape: (B*S, N, D) — treat each position independently
        x = x.view(B * S, N, D)
        # Self-attention across N experts at each position
        for layer in self.layers:
            x = layer(x)  # (B*S, N, D)
        # Aggregate: mean over experts → (B*S, D)
        x = x.mean(dim=1)  # (B*S, D)
        # Reshape back: (B, S, D)
        x = x.view(B, S, D)
        return self.norm(x)
