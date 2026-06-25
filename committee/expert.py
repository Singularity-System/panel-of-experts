import copy
import torch
import torch.nn as nn
from typing import Optional


class Expert(nn.Module):
    """Lightweight Transformer expert.

    Each expert processes the FULL sequence independently.
    No masking, no routing - just a standard Transformer.

    Args:
        num_layers: Number of Transformer layers (2-3 for pilot)
        d_model: Hidden dimension
        n_head: Number of attention heads
        d_ff: Feed-forward dimension
    """

    def __init__(self, num_layers: int = 2, d_model: int = 128, n_head: int = 4, d_ff: int = 256):
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
            x: (B, S, D) input embeddings
            attention_mask: (B, S) attention mask (1 = valid, 0 = padding)

        Returns:
            (B, S, D) output embeddings
        """
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~(attention_mask.bool())

        output = x
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=key_padding_mask)
        return self.norm(output)
