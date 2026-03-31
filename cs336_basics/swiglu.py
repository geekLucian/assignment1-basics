from __future__ import annotations

import math

import torch
from torch import nn
from .linear import Linear

def round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_ff is None:
            # Start from approximately 8/3 * d_model, then round to a multiple of 64
            raw_d_ff = d_model * 8 // 3
            d_ff = round_up_to_multiple(raw_d_ff, 64)

        self.d_model = d_model
        self.d_ff = d_ff

        # Three weight matrices:
        # W1: projects to gate branch
        # W3: projects to value branch
        # W2: projects back to d_model
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)

        # gate branch
        x1 = self.w1(x)

        # value branch
        x3 = self.w3(x)

        # SiLU(x1) = x1 * sigmoid(x1)
        silu = x1 * torch.sigmoid(x1)

        # GLU-style elementwise product
        hidden = silu * x3

        # project back down
        out = self.w2(hidden)

        return out
