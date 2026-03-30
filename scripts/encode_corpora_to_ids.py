from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cs336_basics.bpe import BPETokenizer


def encode_file_to_uint16(
    input_path: Path,
    tokenizer: BPETokenizer,
    output_path: Path,
) -> dict[str, int]:
    token_ids: list[int] = []
    with input_path.open(encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            token_ids.append(token_id)

    max_token_id = max(token_ids, default=0)
    if max_token_id > np.iinfo(np.uint16).max:
        raise ValueError(f"Token id {max_token_id} exceeds uint16 capacity")

    array = np.asarray(token_ids, dtype=np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)

    return {
        "num_tokens": int(array.size),
        "max_token_id": int(max_token_id),
        "input_bytes": input_path.stat().st_size,
        "output_bytes": output_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode corpora into uint16 token-id arrays.")
    parser.add_argument("--tinystories-train", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--tinystories-valid", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--owt-train", type=Path, default=Path("data/owt_train.txt"))
    parser.add_argument("--owt-valid", type=Path, default=Path("data/owt_valid.txt"))
    parser.add_argument("--tinystories-vocab", type=Path, default=Path("artifacts/tinystories_bpe_10000/vocab.json"))
    parser.add_argument("--tinystories-merges", type=Path, default=Path("artifacts/tinystories_bpe_10000/merges.txt"))
    parser.add_argument("--owt-vocab", type=Path, default=Path("artifacts/owt_train_bpe_32000/vocab.json"))
    parser.add_argument("--owt-merges", type=Path, default=Path("artifacts/owt_train_bpe_32000/merges.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/token_ids"))
    args = parser.parse_args()

    tinystories_tokenizer = BPETokenizer.from_files(
        str(args.tinystories_vocab),
        str(args.tinystories_merges),
        special_tokens=["<|endoftext|>"],
    )
    owt_tokenizer = BPETokenizer.from_files(
        str(args.owt_vocab),
        str(args.owt_merges),
        special_tokens=["<|endoftext|>"],
    )

    outputs = {
        "tinystories_train": encode_file_to_uint16(
            args.tinystories_train,
            tinystories_tokenizer,
            args.output_dir / "tinystories_train_uint16.npy",
        ),
        "tinystories_valid": encode_file_to_uint16(
            args.tinystories_valid,
            tinystories_tokenizer,
            args.output_dir / "tinystories_valid_uint16.npy",
        ),
        "owt_train": encode_file_to_uint16(
            args.owt_train,
            owt_tokenizer,
            args.output_dir / "owt_train_uint16.npy",
        ),
        "owt_valid": encode_file_to_uint16(
            args.owt_valid,
            owt_tokenizer,
            args.output_dir / "owt_valid_uint16.npy",
        ),
    }

    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
