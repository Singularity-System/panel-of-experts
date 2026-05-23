import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate_linear = nn.Linear(d_model, num_experts)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: (batch, seq_len, d_model)
        # returns: (weights, indices) each (batch, seq_len, top_k)
        logits = self.gate_linear(hidden_states)  # (batch, seq_len, num_experts)
        weights = F.softmax(logits, dim=-1)

        if self.top_k >= self.num_experts:
            return weights, torch.arange(self.num_experts, device=hidden_states.device).unsqueeze(0).unsqueeze(0).expand(weights.shape[0], weights.shape[1], -1)

        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)
        top_weights = F.normalize(top_weights, p=1, dim=-1)
        return top_weights, top_indices
