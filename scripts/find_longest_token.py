from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path


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


@lru_cache
def gpt2_unicode_to_bytes() -> dict[str, int]:
    return {value: key for key, value in gpt2_bytes_to_unicode().items()}


def decode_serialized_token(token: str) -> bytes:
    byte_decoder = gpt2_unicode_to_bytes()
    return bytes(byte_decoder[character] for character in token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the longest token in a serialized GPT-2 style vocab.")
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("artifacts/tinystories_bpe_10000/vocab.json"),
        help="Path to the vocab JSON file.",
    )
    args = parser.parse_args()

    with open(args.vocab_path, encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)

    longest_token = max(vocab, key=lambda token: (len(decode_serialized_token(token)), len(token), token))
    longest_token_bytes = decode_serialized_token(longest_token)

    print(f"token id: {vocab[longest_token]}")
    print(f"serialized token: {longest_token}")
    print(f"byte length: {len(longest_token_bytes)}")
    print(f"raw bytes: {longest_token_bytes!r}")
    try:
        print(f"utf-8 text: {longest_token_bytes.decode('utf-8')}")
    except UnicodeDecodeError:
        print("utf-8 text: <not valid utf-8>")


if __name__ == "__main__":
    main()
