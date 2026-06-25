import torch
import torch.nn as nn
from typing import Optional

from .expert import Expert
from .compress import compress_sequence
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

    Design (correct):
        1. Broadcast input to all experts
        2. Each expert processes the full sequence → (B, S, D)
        3. compress_sequence per expert: (B, S, D) → (B, D) — compresses sequence, keeps expert!
        4. Stack N experts → (B, N, D)
        5. Chair processes (B, N, D) — self-attention across N experts → (B, D)
        6. LM head → (B, V)

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

        # Chair: fuses N expert vectors via self-attention
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

        # Step 1: Broadcast to all experts
        # Each expert processes full sequence → (B, S, D)
        expert_outputs = []
        for expert in self.experts:
            out = expert(x, attention_mask)  # (B, S, D)
            expert_outputs.append(out)

        # Step 2: Compress sequence for EACH expert → (B, D)
        # compress_sequence operates on the sequence dimension (S), not expert dimension!
        # This preserves N expert vectors!
        expert_vectors = []
        for out in expert_outputs:  # out: (B, S, D)
            # Apply tree-based merge on sequence dimension
            compressed = compress_sequence(out)  # (B, D)
            expert_vectors.append(compressed)

        # Stack: (B, N, D) — N expert vectors preserved!
        expert_vectors = torch.stack(expert_vectors, dim=1)  # (B, N, D)

        # Step 3: Chair — self-attention across N experts
        # Chair receives (B, N, D), does self-attention on N dimension
        # → aggregates to (B, D)
        chair_out = self.chair(expert_vectors)  # (B, D)

        # Step 4: LM head → predict next token
        logits = self.lm_head(chair_out)  # (B, V)

        # Loss: predict the last valid token in each sequence
        loss = None
        if labels is not None:
            last_indices = attention_mask.sum(dim=1).long() - 1  # (B,)
            last_tokens = labels[torch.arange(B), last_indices]  # (B,)
            loss = torch.nn.functional.cross_entropy(logits, last_tokens)

        # Store for diversity loss
        self._last_expert_vectors = expert_vectors  # (B, N, D)

        return {"loss": loss, "logits": logits}

    def diversity_loss(self) -> torch.Tensor:
        """Compute diversity loss per-sample, then average."""
        if self._last_expert_vectors is None:
            return torch.tensor(0.0, device=self.wte.weight.device)

        B = self._last_expert_vectors.size(0)
        losses = []
        for b in range(B):
            vectors = self._last_expert_vectors[b]  # (N, D)
            losses.append(div_loss(vectors))
        return torch.stack(losses).mean()
