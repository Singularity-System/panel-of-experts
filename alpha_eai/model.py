import torch
import torch.nn as nn
from typing import List, Optional, TypeVar, Type, Tuple

from .config import PoEConfig
from .router import Router
from .expert import Expert
from .fusion import ExpertFusion, PostProcessing

T = TypeVar("T", bound=nn.Module)


class PoEModel(nn.Module):
    def __init__(self, config: PoEConfig, expert_type: Type[Expert] = Expert):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(0.1)

        self.router = Router(config.d_model, config.num_experts, config.top_k)

        self.experts = nn.ModuleList([
            expert_type(
                config_id=config.expert_variant,
                num_layers=config.expert_num_layers,
                d_model=config.d_model,
                n_head=config.n_head,
                d_ff=config.d_ff,
            )
            for _ in range(config.num_experts)
        ])

        self.fusion = ExpertFusion(config.num_experts, config.d_model, config.n_head)

        self.post_processing = PostProcessing(
            config_id=config.expert_variant,
            num_layers=config.post_processing_num_layers,
            d_model=config.d_model,
            n_head=config.n_head,
            d_ff=config.d_ff,
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

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

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> dict:
        B, S = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        pos_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.wte(input_ids) + self.wpe(pos_ids)
        x = self.dropout(x)

        router_weights, router_indices = self.router(x)

        expert_outputs = []
        for expert in self.experts:
            out = expert(x, attention_mask)
            expert_outputs.append(out)

        stacked = torch.stack(expert_outputs, dim=2)  # (B, S, N, D)
        fused = self.fusion(stacked, attention_mask)

        pp_out = self.post_processing(fused, attention_mask)

        logits = self.lm_head(pp_out)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"loss": loss, "logits": logits}
