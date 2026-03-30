from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

from cs336_basics.bpe import train_bpe


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for byte_value in range(2**8):
        if byte_value not in bs:
            bs.append(byte_value)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(codepoint) for codepoint in cs), strict=True))


def encode_token_for_serialization(token: bytes, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[byte_value] for byte_value in token)


def save_vocab(vocab: dict[int, bytes], output_path: Path) -> None:
    byte_encoder = gpt2_bytes_to_unicode()
    serialized_vocab = {
        encode_token_for_serialization(token, byte_encoder): token_id
        for token_id, token in vocab.items()
    }
    output_path.write_text(json.dumps(serialized_vocab, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_merges(merges: list[tuple[bytes, bytes]], output_path: Path) -> None:
    byte_encoder = gpt2_bytes_to_unicode()
    lines = [
        f"{encode_token_for_serialization(left, byte_encoder)} {encode_token_for_serialization(right, byte_encoder)}"
        for left, right in merges
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer and save vocab/merges.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-train.txt"),
        help="Path to the training corpus.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset label used in metadata and default output directory naming.",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Maximum vocabulary size.")
    parser.add_argument(
        "--special-token",
        action="append",
        default=None,
        help="Special token to reserve in the vocabulary. Can be passed multiple times.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of worker processes for pretoken counting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where vocab and merges will be written. Defaults to artifacts/<dataset_name>_bpe_<vocab_size>.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name or args.input_path.stem
    special_tokens = args.special_token or ["<|endoftext|>"]
    output_dir = args.output_dir or Path("artifacts") / f"{dataset_name}_bpe_{args.vocab_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab, merges = train_bpe(
        input_path=str(args.input_path),
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        num_processes=args.num_processes,
    )

    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    metadata_path = output_dir / "metadata.json"

    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "input_path": str(args.input_path),
                "vocab_size": args.vocab_size,
                "special_tokens": special_tokens,
                "num_processes": args.num_processes,
                "num_vocab_items": len(vocab),
                "num_merges": len(merges),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved vocab to {vocab_path}")
    print(f"Saved merges to {merges_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
