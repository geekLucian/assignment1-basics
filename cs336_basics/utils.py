from __future__ import annotations

import torch

def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax along dimension `dim` using a numerically stable formulation.
    """
    # torch.max return (result.values, result.indices)
    max_value = torch.max(in_features, dim=dim, keepdim=True).values
    shifted = in_features - max_value
    exp_shifted = torch.exp(shifted)
    return exp_shifted / torch.sum(exp_shifted, dim=dim, keepdim=True)
