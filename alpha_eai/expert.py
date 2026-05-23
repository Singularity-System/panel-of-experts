import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from typing import Optional


class Expert(nn.Module):
    def __init__(self, config_id: str = "gpt2", num_layers: Optional[int] = None, d_model: Optional[int] = None, n_head: Optional[int] = None, d_ff: Optional[int] = None):
        super().__init__()
        if num_layers is not None or d_model is not None:
            hf_config = GPT2Config(
                vocab_size=50257,
                n_layer=num_layers or 5,
                n_embd=d_model or 256,
                n_head=n_head or 4,
                n_inner=d_ff or 512,
                use_cache=False,
            )
            self.transformer = GPT2Model(hf_config)
        else:
            self.transformer = GPT2Model.from_pretrained(config_id)
            if num_layers is not None:
                self.transformer.h = self.transformer.h[:num_layers]

        self.d_model = self.transformer.config.n_embd

    def forward(self, input_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_embeds: (batch, seq_len, d_model)
        # extend hf transformer to accept embeddings directly
        if attention_mask is None:
            attention_mask = torch.ones(input_embeds.shape[:2], device=input_embeds.device, dtype=torch.long)

        outputs = self.transformer(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.last_hidden_state
