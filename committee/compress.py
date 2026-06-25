import torch
import torch.nn.functional as F


def merge(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Merge two vectors using dot product similarity.

    Args:
        a: (..., D) first vector
        b: (..., D) second vector

    Returns:
        (..., D) merged vector
    """
    # Normalize
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)

    # Dot product similarity
    sim = (a_norm * b_norm).sum(dim=-1, keepdim=True)  # (..., 1)

    # Alpha: direction similar → larger weight
    alpha = torch.sigmoid(sim)  # (..., 1)

    return alpha * a + (1 - alpha) * b


def compress_sequence(sequences: torch.Tensor) -> torch.Tensor:
    """Compress a sequence of vectors using tree-based merge.

    Tree structure:
        [a, b, c, d, e]
        → [merge(a,b), merge(c,d), merge(e,a)]  ← odd element merges with first
        → [merge(merge(a,b), merge(c,d)), merge(e,a)]
        → [merge(..., merge(e,a))]

    All elements participate in merging — no orphan inheritance.

    Args:
        sequences: (B, N, D) batch of N expert vectors

    Returns:
        (B, D) single committee representation
    """
    if sequences.dim() == 2:
        sequences = sequences.unsqueeze(0)  # (N, D) → (1, N, D)

    B, N, D = sequences.shape

    # Tree-based merge
    current = sequences  # (B, N, D)
    while current.size(1) > 1:
        seq = current.size(1)
        new_seq = []
        for i in range(0, seq, 2):
            if i + 1 < seq:
                merged = merge(current[:, i, :], current[:, i + 1, :])  # (B, D)
                new_seq.append(merged.unsqueeze(1))  # (B, 1, D)
            else:
                # Odd element: merge with first element of this round instead of inheriting
                merged = merge(current[:, i, :], current[:, 0, :])  # (B, D)
                new_seq.append(merged.unsqueeze(1))
        current = torch.cat(new_seq, dim=1)  # (B, new_seq_len, D)

    return current.squeeze(1)  # (B, D)
