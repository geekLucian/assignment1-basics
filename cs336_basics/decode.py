from __future__ import annotations

import torch
from .utils import softmax


@torch.no_grad()
def decode(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    end_of_text_token_id: int = 0,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Args:
        model: language model returning logits of shape (..., seq_len, vocab_size)
        prompt: token ids, shape (1, prompt_len) or (..., prompt_len)
        max_new_tokens: maximum number of tokens to generate
        end_of_text_token_id: stop token id
        temperature: temperature for sampling
        top_p: nucleus sampling threshold

    Returns:
        Tensor containing prompt + generated tokens.
    """
    model.eval()

    tokens = prompt

    for _ in range(max_new_tokens):
        # 1. Run model on current tokens.
        logits = model(tokens)

        # 2. Take logits for the next-token distribution only.
        next_token_logits = logits[..., -1, :]

        # 3. Apply temperature scaling.
        if temperature != 0:
            next_token_logits = next_token_logits / temperature

        # 4. Convert logits to probabilities.
        probs = softmax(next_token_logits, dim=-1)

        # 5. Apply top-p / nucleus sampling if requested.
        if top_p > 0:
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            keep_mask = cumulative_probs <= top_p

            # renormalize kept probabilities
            filtered_probs = sorted_probs * keep_mask
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            sampled_idx_in_sorted = torch.multinomial(filtered_probs, num_samples=1)
            next_token = torch.gather(sorted_indices, dim=-1, index=sampled_idx_in_sorted)
        else:
            next_token =  torch.multinomial(probs, num_samples=1)

        # 6. Append sampled token to the running sequence.
        tokens = torch.cat([tokens, next_token], dim=-1)

        # 7. Stop if <|endoftext|> is generated.
        if (next_token == end_of_text_token_id).all():
            break

    return tokens
