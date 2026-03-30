from __future__ import annotations

import heapq
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import lru_cache
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_PATTERN = re.compile(PAT)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    assert desired_num_chunks > 0, "desired_num_chunks must be positive"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks == 1 or not split_special_tokens:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    max_token_length = max(len(token) for token in split_special_tokens)
    search_size = mini_chunk_size + max_token_length - 1

    for boundary_index in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[boundary_index]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(search_size)
            if mini_chunk == b"":
                chunk_boundaries[boundary_index] = file_size
                break

            found_positions = [
                position
                for token in split_special_tokens
                if (position := mini_chunk.find(token)) != -1
            ]
            if found_positions:
                chunk_boundaries[boundary_index] = initial_position + min(found_positions)
                break

            initial_position += mini_chunk_size
            file.seek(initial_position)

    return sorted(set(chunk_boundaries))


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> Iterable[str]:
    if not special_tokens:
        yield text
        return

    split_pattern = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    yield from re.split(split_pattern, text)


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    for segment in _split_on_special_tokens(text, special_tokens):
        for match in PRETOKEN_PATTERN.finditer(segment):
            counts[match.group().encode("utf-8")] += 1
    return counts


def _process_chunk(args: tuple[str, int, int, tuple[str, ...]]) -> Counter[bytes]:
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    return _pretoken_counts(chunk, list(special_tokens))


def _pretoken_counts_from_file(
    input_path: str,
    special_tokens: list[str],
    num_processes: int | None,
) -> Counter[bytes]:
    if num_processes is None:
        num_processes = min(os.cpu_count() or 1, 4)

    file_size = os.path.getsize(input_path)
    if num_processes <= 1 or file_size < 1_000_000 or not special_tokens:
        with open(input_path, encoding="utf-8") as f:
            return _pretoken_counts(f.read(), special_tokens)

    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_token_bytes)

    if len(boundaries) <= 2:
        with open(input_path, encoding="utf-8") as f:
            return _pretoken_counts(f.read(), special_tokens)

    chunk_args = [
        (input_path, start, end, tuple(special_tokens))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(processes=min(num_processes, len(chunk_args))) as pool:
        partial_counts = pool.map(_process_chunk, chunk_args)

    counts: Counter[bytes] = Counter()
    for chunk_counts in partial_counts:
        counts.update(chunk_counts)
    return counts


def _merge_word(word: tuple[int, ...], pair: tuple[int, int], merged_token_id: int) -> tuple[int, ...]:
    left, right = pair
    merged: list[int] = []
    index = 0
    last_index = len(word) - 1

    while index < len(word):
        if index < last_index and word[index] == left and word[index + 1] == right:
            merged.append(merged_token_id)
            index += 2
        else:
            merged.append(word[index])
            index += 1
    return tuple(merged)


def _count_word_pairs(word: tuple[int, ...]) -> Counter[tuple[int, int]]:
    return Counter(zip(word, word[1:]))


def _descending_bytes_key(token: bytes) -> tuple[int, ...]:
    return tuple(-(byte_value + 1) for byte_value in token) + (0,)


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


def _decode_serialized_token(token: str) -> bytes:
    byte_decoder = gpt2_unicode_to_bytes()
    return bytes(byte_decoder[character] for character in token)


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.special_tokens = special_tokens or []
        self.id_to_token = dict(vocab)
        self.token_to_id = {token: token_id for token_id, token in self.id_to_token.items()}
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in self.token_to_id:
                token_id = len(self.id_to_token)
                self.id_to_token[token_id] = special_token_bytes
                self.token_to_id[special_token_bytes] = token_id
        self.vocab = dict(self.id_to_token)
        self.special_token_to_id = {
            token.encode("utf-8"): self.token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
        }
        self.special_token_pattern = (
            re.compile("|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)))
            if self.special_tokens
            else None
        )
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.encode_cache: dict[bytes, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> BPETokenizer:
        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            serialized_vocab: dict[str, int] = json.load(vocab_file)
        vocab = {
            token_id: _decode_serialized_token(serialized_token)
            for serialized_token, token_id in serialized_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                cleaned_line = line.rstrip()
                if not cleaned_line:
                    continue
                left, right = cleaned_line.split(" ")
                merges.append((_decode_serialized_token(left), _decode_serialized_token(right)))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        cached = self.encode_cache.get(pretoken)
        if cached is not None:
            return cached

        parts = [bytes([byte_value]) for byte_value in pretoken]
        if len(parts) == 1:
            result = [self.token_to_id[parts[0]]]
            self.encode_cache[pretoken] = result
            return result

        while True:
            best_index = -1
            best_rank = None
            for index in range(len(parts) - 1):
                pair = (parts[index], parts[index + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = index
            if best_index == -1:
                break
            merged = parts[best_index] + parts[best_index + 1]
            parts[best_index : best_index + 2] = [merged]

        result = [self.token_to_id[part] for part in parts]
        self.encode_cache[pretoken] = result
        return result

    def _encode_text_segment(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in PRETOKEN_PATTERN.finditer(text):
            ids.extend(self._encode_pretoken(match.group().encode("utf-8")))
        return ids

    def encode(self, text: str) -> list[int]:
        if not self.special_token_pattern:
            return self._encode_text_segment(text)

        ids: list[int] = []
        last_end = 0
        for match in self.special_token_pattern.finditer(text):
            if match.start() > last_end:
                ids.extend(self._encode_text_segment(text[last_end : match.start()]))
            ids.append(self.special_token_to_id[match.group().encode("utf-8")])
            last_end = match.end()
        if last_end < len(text):
            ids.extend(self._encode_text_segment(text[last_end:]))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: Iterable[int]) -> str:
        decoded = b"".join(self.id_to_token[token_id] for token_id in ids)
        return decoded.decode("utf-8", errors="replace")


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretoken_counts = _pretoken_counts_from_file(input_path, special_tokens, num_processes)

    token_bytes: list[bytes] = [token.encode("utf-8") for token in special_tokens]
    token_bytes.extend(bytes([byte_value]) for byte_value in range(256))
    token_heap_keys: list[tuple[int, ...]] = [_descending_bytes_key(token) for token in token_bytes]

    byte_offset = len(special_tokens)
    word_counts: dict[tuple[int, ...], int] = {
        tuple(byte_offset + byte_value for byte_value in pretoken): count
        for pretoken, count in pretoken_counts.items()
    }

    word_pair_counts: dict[tuple[int, ...], Counter[tuple[int, int]]] = {}
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)

    for word, count in word_counts.items():
        local_pair_counts = _count_word_pairs(word)
        if not local_pair_counts:
            continue
        word_pair_counts[word] = local_pair_counts
        for pair, occurrences in local_pair_counts.items():
            pair_counts[pair] += occurrences * count
            pair_to_words[pair].add(word)

    pair_heap: list[tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, int]]] = [
        (-count, token_heap_keys[pair[0]], token_heap_keys[pair[1]], pair)
        for pair, count in pair_counts.items()
    ]
    heapq.heapify(pair_heap)

    merges: list[tuple[bytes, bytes]] = []
    target_merges = max(0, vocab_size - len(token_bytes))

    for _ in range(target_merges):
        best_pair: tuple[int, int] | None = None
        while pair_heap:
            neg_count, _, _, candidate_pair = heapq.heappop(pair_heap)
            if pair_counts.get(candidate_pair) == -neg_count:
                best_pair = candidate_pair
                break
        if best_pair is None:
            break

        left, right = best_pair
        merges.append((token_bytes[left], token_bytes[right]))

        merged_token_id = len(token_bytes)
        token_bytes.append(token_bytes[left] + token_bytes[right])
        token_heap_keys.append(_descending_bytes_key(token_bytes[merged_token_id]))

        affected_words = list(pair_to_words.get(best_pair, ()))
        changed_pairs: set[tuple[int, int]] = set()
        for word in affected_words:
            count = word_counts.pop(word)
            old_pair_counts = word_pair_counts.pop(word)

            for pair, occurrences in old_pair_counts.items():
                updated_count = pair_counts[pair] - occurrences * count
                if updated_count:
                    pair_counts[pair] = updated_count
                else:
                    del pair_counts[pair]
                changed_pairs.add(pair)

                pair_words = pair_to_words[pair]
                pair_words.discard(word)
                if not pair_words:
                    del pair_to_words[pair]

            merged_word = _merge_word(word, best_pair, merged_token_id)
            word_counts[merged_word] = word_counts.get(merged_word, 0) + count

            merged_word_pair_counts = word_pair_counts.get(merged_word)
            if merged_word_pair_counts is None:
                merged_word_pair_counts = _count_word_pairs(merged_word)
                if merged_word_pair_counts:
                    word_pair_counts[merged_word] = merged_word_pair_counts

            for pair, occurrences in merged_word_pair_counts.items():
                pair_counts[pair] += occurrences * count
                pair_to_words[pair].add(merged_word)
                changed_pairs.add(pair)

        for pair in changed_pairs:
            count = pair_counts.get(pair)
            if count is None:
                continue
            heapq.heappush(pair_heap, (-count, token_heap_keys[pair[0]], token_heap_keys[pair[1]], pair))

    vocab = {token_id: token for token_id, token in enumerate(token_bytes)}
    return vocab, merges
