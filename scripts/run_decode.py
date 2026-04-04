from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.adamw import AdamW
from cs336_basics.bpe import BPETokenizer
from cs336_basics.decode import decode
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run decoding from a trained checkpoint.")

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--end-of-text-token", type=str, default="<|endoftext|>")

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = BPETokenizer.from_files(
        str(args.vocab_path),
        str(args.merges_path),
        special_tokens=[args.end_of_text_token],
    )

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )

    optimizer = AdamW(model.parameters(), lr=1e-3)
    _ = load_checkpoint(args.checkpoint, model, optimizer)

    prompt_ids = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    end_of_text_token_id = tokenizer.special_token_to_id[args.end_of_text_token.encode("utf-8")]

    output_ids = decode(
        model=model,
        prompt=prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        end_of_text_token_id=end_of_text_token_id,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    output_text = tokenizer.decode(output_ids[0].tolist())
    print(output_text)


if __name__ == "__main__":
    main()
