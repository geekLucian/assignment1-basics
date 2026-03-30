from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from cs336_basics.bpe import BPETokenizer

END_OF_TEXT = b"<|endoftext|>"


def sample_documents(
    input_path: Path,
    num_documents: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    sampled: list[bytes] = []
    carry = b""
    seen = 0

    with input_path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            carry += chunk
            parts = carry.split(END_OF_TEXT)
            carry = parts.pop()
            for part in parts:
                document = part.strip(b"\n")
                if not document:
                    continue
                seen += 1
                if len(sampled) < num_documents:
                    sampled.append(document)
                    continue
                replace_index = rng.randrange(seen)
                if replace_index < num_documents:
                    sampled[replace_index] = document

    tail_document = carry.strip(b"\n")
    if tail_document:
        seen += 1
        if len(sampled) < num_documents:
            sampled.append(tail_document)
        else:
            replace_index = rng.randrange(seen)
            if replace_index < num_documents:
                sampled[replace_index] = tail_document

    return [document.decode("utf-8") for document in sampled]


def compression_ratio(tokenizer: BPETokenizer, documents: list[str]) -> dict[str, float | int]:
    total_bytes = sum(len(document.encode("utf-8")) for document in documents)
    total_tokens = sum(len(tokenizer.encode(document)) for document in documents)
    return {
        "documents": len(documents),
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample documents and measure tokenizer compression ratios.")
    parser.add_argument("--tinystories-path", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--owt-path", type=Path, default=Path("data/owt_train.txt"))
    parser.add_argument("--tinystories-vocab", type=Path, default=Path("artifacts/tinystories_bpe_10000/vocab.json"))
    parser.add_argument("--tinystories-merges", type=Path, default=Path("artifacts/tinystories_bpe_10000/merges.txt"))
    parser.add_argument("--owt-vocab", type=Path, default=Path("artifacts/owt_train_bpe_32000/vocab.json"))
    parser.add_argument("--owt-merges", type=Path, default=Path("artifacts/owt_train_bpe_32000/merges.txt"))
    parser.add_argument("--num-documents", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/compression_experiment.json"))
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

    tinystories_docs = sample_documents(args.tinystories_path, args.num_documents, args.seed)
    owt_docs = sample_documents(args.owt_path, args.num_documents, args.seed)

    results = {
        "seed": args.seed,
        "num_documents": args.num_documents,
        "tinystories_sample": {
            "tinystories_tokenizer": compression_ratio(tinystories_tokenizer, tinystories_docs),
            "owt_tokenizer": compression_ratio(owt_tokenizer, tinystories_docs),
        },
        "owt_sample": {
            "tinystories_tokenizer": compression_ratio(tinystories_tokenizer, owt_docs),
            "owt_tokenizer": compression_ratio(owt_tokenizer, owt_docs),
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
