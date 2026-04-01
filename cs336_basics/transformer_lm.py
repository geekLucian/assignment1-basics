from __future__ import annotations

import torch
from torch import nn

from .embedding import Embedding
from .rmsnorm import RMSNorm
from .linear import Linear
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    theta=rope_theta,
                    max_seq_len=context_length,
                    device=device,
                    dtype=dtype
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            in_indices: (..., seq_len)

        Returns:
            (..., seq_len, vocab_size)
        """
        # 1. Embed token ids
        x = self.token_embeddings(in_indices)

        # 2. Pass through all transformer blocks
        for layer in self.layers:
            x = layer(x)

        # 3. Final normalization
        x = self.ln_final(x)

        # 4. Project to vocabulary logits
        logits = self.lm_head(x)

        return logits
