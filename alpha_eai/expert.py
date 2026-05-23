import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from typing import Optional


class Expert(nn.Module):
    """Expert: GPT2 transformer blocks that accepts input_embeds directly.
    Embedding layers are excluded to avoid redundant params."""

    def __init__(self, num_layers: int = 5, d_model: int = 256, n_head: int = 4, d_ff: int = 512):
        super().__init__()
        hf_config = GPT2Config(
            vocab_size=1,  # no embedding needed
            n_layer=num_layers,
            n_embd=d_model,
            n_head=n_head,
            n_inner=d_ff,
            use_cache=False,
        )
        self.transformer = GPT2Model(hf_config)
        # Remove the unused wte embedding (vocab_size=1 makes it tiny)
        self.d_model = self.transformer.config.n_embd

    def forward(self, input_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(input_embeds.shape[:2], device=input_embeds.device, dtype=torch.long)
        outputs = self.transformer(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state
