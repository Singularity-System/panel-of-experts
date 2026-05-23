import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from typing import Optional


class BaselineTransformer(nn.Module):
    """Same-depth baseline: plain Transformer without PoE."""

    def __init__(self, num_layers=11, d_model=256, n_head=4, d_ff=512,
                 vocab_size=50257, max_seq_len=256):
        super().__init__()
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_layer=num_layers,
            n_embd=d_model,
            n_head=n_head,
            n_inner=d_ff,
            use_cache=False,
        )
        self.transformer = GPT2Model(self.config)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Reuse embedding weights (weight tying)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, S = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        return {"loss": loss, "logits": logits}
