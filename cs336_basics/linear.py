from __future__ import annotations

import torch
from torch import nn
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features,
                in_features,
                device=device,
                dtype=dtype
            )
        )

        std = math.sqrt(2.0 / (in_features + out_features))

        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., in_features)
        # weight shape: (out_features, in_features)
        y = einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")
        return y
    
