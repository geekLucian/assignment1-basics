from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # Learnable scale parameter
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Normalize over the last dimension only
        # 1. compute mean square
        # keepdim=True, make it (..., d_model) -> (..., 1)
        # for later easier / x
        mean_square = torch.mean(x*x, dim=-1, keepdim=True)

        # 2. compute rms denominator
        rms = torch.sqrt(mean_square + self.eps)

        # 3. normalize
        normalized = x / rms

        # 4. apply learned scale
        result = normalized * self.weight

        return result.to(in_dtype)
