from __future__ import annotations

import torch
from einops import einsum, reduce
import numpy as np
import math

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


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Return the learning rate at iteration `it` for a linear warmup
    followed by cosine decay schedule.

    Args:
        it: current iteration
        max_learning_rate: alpha_max
        min_learning_rate: alpha_min
        warmup_iters: T_w
        cosine_cycle_iters: T_c
    """
    # 1. Warmup phase
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate

    # 2. Cosine decay phase
    if warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate) 

    # 3. After the cosine schedule ends, stay at the minimum LR
    return min_learning_rate


def gradient_clipping(
    parameters,
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    # 1. Collect only parameters that actually have gradients.
    params_with_grads = [p for p in parameters if p.grad is not None]

    # 2. Compute the total squared L2 norm across all gradients.
    total_squared_norm = sum(p.grad.data.pow(2).sum() for p in params_with_grads)

    # 3. Take the square root to get the global L2 norm.
    total_norm = torch.sqrt(total_squared_norm)

    # 4. Compute the clipping coefficient.
    clip_coef = max_l2_norm / (total_norm + eps)

    # 5. If clipping is needed, scale gradients in-place.
    if total_norm > max_l2_norm:
        for p in params_with_grads:
            p.grad.data *= clip_coef 
        

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of input/target token sequences from a 1D token-id array.

    Returns:
        inputs:  (batch_size, context_length)
        targets: (batch_size, context_length)
    """
    # 1. Sample random start positions for each example.
    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)

    # 2. Build input sequences of length `context_length`.
    inputs = np.stack([dataset[i:i+context_length] for i in start_indices])

    # 3. Build next-token targets shifted by one position.
    targets = np.stack([dataset[i+1:i+1+context_length] for i in start_indices])

    # 4. Convert to torch tensors on the requested device.
    inputs = torch.from_numpy(inputs).to(device)
    targets = torch.from_numpy(targets).to(device)

    return inputs, targets



