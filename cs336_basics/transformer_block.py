from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .rope import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        d_k = d_model // num_heads
        rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=d_k,
            max_seq_len=max_seq_len,
            device=device
        )
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            device=device,
            dtype=dtype
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., seq_len, d_model)

        Returns:
            (..., seq_len, d_model)
        """
        # First pre-norm sublayer:
        # y = x + MultiHeadSelfAttention(RMSNorm(x))
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        y = x + self.attn(self.ln1(x), token_positions)

        # Second pre-norm sublayer:
        # out = y + SwiGLU(RMSNorm(y))
        out = y + self.ffn(self.ln2(y))

        return out
