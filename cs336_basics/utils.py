from __future__ import annotations

import torch
from einops import einsum, reduce

def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax along dimension `dim` using a numerically stable formulation.
    """
    # torch.max return (result.values, result.indices)
    max_value = torch.max(in_features, dim=dim, keepdim=True).values
    shifted = in_features - max_value
    exp_shifted = torch.exp(shifted)
    return exp_shifted / torch.sum(exp_shifted, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Args:
        Q: (..., queries, d_k)
        K: (..., keys, d_k)
        V: (..., keys, d_v)
        mask: optional boolean tensor broadcastable to (..., queries, keys)

    Returns:
        Tensor of shape (..., queries, d_v)
    """
    d_k = Q.shape[-1]

    # 1. Compute attention scores.
    # scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k ** 0.5)
    scores = Q @ K.transpose(-1, -2) / (d_k ** 0.5)

    # 2. If a mask is provided, apply it so masked-out positions get zero probability.
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # 3. Normalize scores across the key dimension.
    attention_probs = softmax(scores, dim=-1)

    # 4. Use the attention probabilities to combine values.
    # output = einsum(attention_probs, V, "... queries keys, ... keys d_v -> ... queries d_v")
    output = attention_probs @ V

    return output


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute average cross-entropy loss from unnormalized logits.

    Args:
        inputs: (..., vocab_size)
        targets: (...)

    Returns:
        Scalar tensor containing the average loss.
    """
    shifted = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    log_denom = torch.log(torch.sum(torch.exp(shifted), dim=-1, keepdim=True))
    target_logits = torch.gather(shifted, dim=-1, index=targets.unsqueeze(-1))
    losses = -target_logits + log_denom
    return losses.mean()
