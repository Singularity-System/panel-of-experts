import copy
import torch
import torch.nn as nn
from typing import Optional


class Chair(nn.Module):
    """Chair: single-layer Transformer for committee fusion.

    Takes the compressed committee representation and produces final output.

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

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) input
            attention_mask: (B, S) attention mask

        Returns:
            (B, S, D) output
        """
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~(attention_mask.bool())

        output = x
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=key_padding_mask)
        return self.norm(output)
