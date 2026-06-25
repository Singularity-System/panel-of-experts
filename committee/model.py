import torch
import torch.nn as nn
from typing import Optional

from .expert import Expert
from .chair import Chair
from .loss import div_loss


class CommitteeConfig:
    """Configuration for Committee model."""
    def __init__(
        self,
        num_experts: int = 8,
        expert_num_layers: int = 2,
        d_model: int = 128,
        n_head: int = 4,
        d_ff: int = 256,
        vocab_size: int = 50257,
        max_seq_len: int = 256,
        chair_num_layers: int = 1,
        div_loss_weight: float = 0.5,
    ):
        self.num_experts = num_experts
        self.expert_num_layers = expert_num_layers
        self.d_model = d_model
        self.n_head = n_head
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.chair_num_layers = chair_num_layers
        self.div_loss_weight = div_loss_weight


class CommitteeModel(nn.Module):
    """Committee Architecture.

    Design (correct for LM):
        1. Broadcast input to all experts
        2. Each expert processes the full sequence → (B, S, D)
        3. Stack N experts → (B, S, N, D) — preserves S!
        4. Chair fuses N experts per position → (B, S, D)
        5. LM head → (B, S, V)
        6. Shift loss: position i predicts position i+1

    compress_sequence is NOT used in training — it's reserved for inference
    or other tasks where sequence compression is needed.

    Args:
        config: CommitteeConfig
    """

    def __init__(self, config: CommitteeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.d_model = config.d_model

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(0.1)

        # Experts
        self.experts = nn.ModuleList([
            Expert(
                num_layers=config.expert_num_layers,
                d_model=config.d_model,
                n_head=config.n_head,
                d_ff=config.d_ff
            )
            for _ in range(config.num_experts)
        ])

        # Chair: fuses N expert vectors per position via self-attention
        self.chair = Chair(
            d_model=config.d_model,
            n_head=config.n_head,
            d_ff=config.d_ff,
            num_layers=config.chair_num_layers
        )

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        # Store for diversity loss
        self._last_expert_vectors = None

        # Hyperparameter
        self.div_loss_weight = config.div_loss_weight

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

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        """
        Args:
            input_ids: (B, S) input token IDs
            attention_mask: (B, S) attention mask
            labels: (B, S) labels for loss

        Returns:
            dict with 'loss', 'logits'
        """
        B, S = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Embed
        pos_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.wte(input_ids) + self.wpe(pos_ids)
        x = self.dropout(x)  # (B, S, D)

        # Step 1: Parallel expert processing
        # Stack input N times → (N, B, S, D), process all experts in parallel
        N = self.num_experts
        x_stacked = x.unsqueeze(0).expand(N, -1, -1, -1)  # (N, B, S, D)

        expert_outputs = []
        for i, expert in enumerate(self.experts):
            out = expert(x_stacked[i], attention_mask)  # (B, S, D)
            expert_outputs.append(out)

        # Step 2: Stack N experts → (B, S, N, D) — preserves S!
        expert_vectors = torch.stack(expert_outputs, dim=2)  # (B, S, N, D)

        # Step 3: Chair fuses N experts per position → (B, S, D)
        chair_out = self.chair(expert_vectors)  # (B, S, D)

        # Step 4: LM head → (B, S, V)
        logits = self.lm_head(chair_out)  # (B, S, V)

        # Step 5: Shift loss — position i predicts position i+1
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()  # (B, S-1, V)
            shift_labels = labels[..., 1:].contiguous()  # (B, S-1)
            shift_mask = attention_mask[..., 1:].contiguous()  # (B, S-1)
            mask_expanded = shift_mask.view(-1)
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1))[mask_expanded.bool()],
                shift_labels.view(-1)[mask_expanded.bool()]
            )

        # Store for diversity loss
        self._last_expert_vectors = expert_vectors  # (B, S, N, D)

        return {"loss": loss, "logits": logits}

    def diversity_loss(self) -> torch.Tensor:
        """Compute diversity loss per-sample, then average.

        (B, S, N, D) → average over S → (B, N, D) → per-sample div_loss → average over B
        """
        if self._last_expert_vectors is None:
            return torch.tensor(0.0, device=self.wte.weight.device)

        # (B, S, N, D) → mean over S → (B, N, D)
        expert_means = self._last_expert_vectors.mean(dim=1)  # (B, N, D)

        B = expert_means.size(0)
        losses = []
        for b in range(B):
            vectors = expert_means[b]  # (N, D)
            losses.append(div_loss(vectors))
        return torch.stack(losses).mean()
