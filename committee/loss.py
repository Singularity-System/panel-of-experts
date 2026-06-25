import torch
import math


def div_loss(vectors: torch.Tensor) -> torch.Tensor:
    """Von Neumann entropy diversity loss.

    Forces expert output vectors to be orthogonal in the embedding space.

    Args:
        vectors: (N, D) N expert vectors, D dimensions

    Returns:
        Scalar loss (0 = max diversity, 1 = no diversity)
    """
    if vectors.size(0) <= 1:
        return torch.tensor(0.0, device=vectors.device)

    # Normalize
    norms = vectors.norm(dim=-1, keepdim=True)
    if (norms < 1e-8).any():
        return torch.tensor(0.0, device=vectors.device)

    normalized = vectors / norms  # (N, D)

    # Gram matrix (cosine similarity)
    gram = normalized @ normalized.T  # (N, N)

    # Eigenvalues
    eigvals = torch.linalg.eigvalsh(gram)  # (N,)
    eigvals = torch.clamp(eigvals, min=1e-8)  # positive
    total = eigvals.sum()
    p = eigvals / total  # probability distribution

    # Von Neumann entropy
    entropy = -(p * p.log()).sum()
    max_entropy = math.log(float(vectors.size(0)))

    # Normalize to [0, 1]
    return 1.0 - entropy / max_entropy
