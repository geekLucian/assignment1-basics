from __future__ import annotations

import torch
from torch import nn, Tensor
from einops import rearrange
from jaxtyping import Bool, Float, Int
from .utils import scaled_dot_product_attention
from .linear import Linear
from .rope import RotaryPositionalEmbedding

# Template for adding optional RoPE support:
# class MultiHeadSelfAttention(nn.Module):
#     def __init__(..., rope: RotaryPositionalEmbedding | None = None) -> None:
#         ...
#         self.rope = rope
#
#     def forward(
#         self,
#         x: torch.Tensor,
#         token_positions: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         q = ...
#         k = ...
#         v = ...
#         q = rearrange(q, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads)
#         k = rearrange(k, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads)
#         v = rearrange(v, "... seq_len (heads d_v) -> ... heads seq_len d_v", heads=self.num_heads)
#
#         if self.rope is not None:
#             q = self.rope(q, token_positions)
#             k = self.rope(k, token_positions)
#
#         attn_output = ...
#         return ...
#
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.q_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, num_heads * self.d_v, device=device, dtype=dtype)
        self.output_proj = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype)

        self.rope = rope


    def forward(self, x: torch.Tensor, token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> torch.Tensor:
        """
        Args:
            x: (..., seq_len, d_model)

        Returns:
            (..., seq_len, d_model)
        """
        seq_len = x.shape[-2]

        # 1. Project inputs to Q, K, V.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape into heads.
        # Expected shape after reshape/transposition:
        # (..., num_heads, seq_len, d_k) for q and k
        # (..., num_heads, seq_len, d_v) for v
        q = rearrange(q, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads)
        k = rearrange(k, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads)
        v = rearrange(v, "... seq_len (heads d_v) -> ... heads seq_len d_v", heads=self.num_heads)

        # 3. Build a causal mask of shape (seq_len, seq_len).
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # 4. Apply scaled dot-product attention per head.
        if self.rope is not None:
            token_positions = rearrange(token_positions, "... seq_len -> ... 1 seq_len")
            token_positions = token_positions.expand(*token_positions.shape[:-2], self.num_heads, token_positions.shape[-1])
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        attn_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # 5. Merge heads back to (..., seq_len, d_model).
        attn_output = rearrange(attn_output, "... heads seq_len d_v -> ... seq_len (heads d_v)")

        # 6. Final output projection.
        output = self.output_proj(attn_output)

        return output
