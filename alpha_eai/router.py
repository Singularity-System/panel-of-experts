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
        self._full_probs = None  # for LB loss
        self._indices = None

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate_linear(hidden_states)  # (B, S, num_experts)
        full_probs = F.softmax(logits, dim=-1)
        self._full_probs = full_probs

        if self.top_k >= self.num_experts:
            indices = torch.arange(self.num_experts, device=hidden_states.device).unsqueeze(0).unsqueeze(0).expand(full_probs.shape[0], full_probs.shape[1], -1)
            self._indices = indices
            return full_probs, indices

        top_weights, top_indices = torch.topk(full_probs, self.top_k, dim=-1)
        top_weights = F.normalize(top_weights, p=1, dim=-1)
        self._indices = top_indices
        return top_weights, top_indices

    def auxiliary_load_balance_loss(self) -> torch.Tensor:
        """Switch Transformer: N * Σ(f_i * P_i). 0 if not called in training."""
        if self._full_probs is None or self._indices is None:
            return torch.tensor(0.0, device=self.gate_linear.weight.device)

        P = self._full_probs.mean(dim=[0, 1])  # (num_experts,) soft probs
        flat_idx = self._indices.reshape(-1)
        one_hot = torch.zeros(flat_idx.shape[0], self.num_experts, device=flat_idx.device)
        one_hot.scatter_(1, flat_idx.unsqueeze(1), 1.0)
        f = one_hot.float().mean(dim=0)  # (num_experts,) hard freqs
        return self.num_experts * (f * P).sum()
