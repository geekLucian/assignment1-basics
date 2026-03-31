from __future__ import annotations

import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        if d_k % 2 != 0:
            raise ValueError("RoPE requires d_k to be even.")

        inv_freq = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] / torch.pow(theta, inv_freq[None, :])

        # These are registered as buffers so they move with the module across devices.
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # Rotate each adjacent pair in the last dimension:
        # [a, b, c, d, e, f] -> [-b, a, -d, c, -f, e].
        # x_even = [a, c, e]
        x_even = x[..., ::2]
        # x_odd [b, d, f]
        x_odd = x[..., 1::2]
        # rotated = [[-b, a], [-d, c], [-f, e]]
        rotated = torch.stack((-x_odd, x_even), dim=-1)
        # flatten = [-b, a, -d, c, -f, e]
        # [..., x_2k, x_2k+1] -> [..., -x_2k+1, x_2k]
        return rotated.flatten(start_dim=-2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # Expand from (..., seq_len, d_k // 2) to (..., seq_len, d_k)
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)

        return x * cos + self._rotate_half(x) * sin
