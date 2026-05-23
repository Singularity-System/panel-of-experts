import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from typing import Optional


class ExpertFusion(nn.Module):
    def __init__(self, num_experts: int, d_model: int, n_head: int = 4):
        super().__init__()
        self.expert_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, expert_outputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, N, D = expert_outputs.shape
        expert_outputs_2d = expert_outputs.reshape(B * S, N, D)
        attn_output, _ = self.expert_attention(expert_outputs_2d, expert_outputs_2d, expert_outputs_2d, need_weights=False)
        fused = attn_output.mean(dim=1)
        fused = self.ln(fused)
        return fused.reshape(B, S, D)


class PostProcessing(nn.Module):
    """Post-processing transformer without input embedding layers.
    Position encoding is disabled to avoid duplication with PoEModel's wpe."""

    def __init__(self, num_layers: int = 6, d_model: int = 256, n_head: int = 4, d_ff: int = 512):
        super().__init__()
        hf_config = GPT2Config(
            vocab_size=1,
            n_layer=num_layers,
            n_embd=d_model,
            n_head=n_head,
            n_inner=d_ff,
            use_cache=False,
        )
        self.transformer = GPT2Model(hf_config)
        # Zero position embedding to avoid double-encoding
        self.transformer.wpe.weight.data.zero_()
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
