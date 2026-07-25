from abc import ABC
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import os
from typing import BinaryIO

import regex


class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""

    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""

    vocab: dict[int, bytes]  # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index


def split_pretokens(text: str):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return regex.finditer(PAT, text)


PretokenCounts = dict[tuple[bytes, ...], int]
PairCounts = dict[tuple[bytes, bytes], int]


def get_stats(pretoken_counts: PretokenCounts) -> PairCounts:
    out = defaultdict(int)
    for pretoken, count in pretoken_counts.items():
        for a, b in zip(pretoken[:-1], pretoken[1:]):
            out[(a, b)] += count
    return out


def main():
    parser = argparse.ArgumentParser(description="Read a file as raw bytes.")
    parser.add_argument(
        "file",
        type=str,
        help="Path to the file to read",
    )
    parser.add_argument("--merges", "-m", type=int, default=4, help="Number of merges")
    args = parser.parse_args()

    with open(args.file, "rb") as file:
        chunk = file.read().decode("utf-8", errors="ignore")
        # this is the most naive possible solution
        vocab: dict[int, bytes] = {}
        vocab[0] = b"<|endoftext|>"
        for i in range(256):
            vocab[i + 1] = bytes(i)  # icky, but makes more sense IMO.

        pretoken_counts: PretokenCounts = defaultdict(int)
        pretokens = split_pretokens(chunk)
        for pretoken in pretokens:
            pretoken_counts[tuple(map(lambda x: x.encode("utf-8"), pretoken.group()))] += 1

        pair_counts = get_stats(pretoken_counts)
        to_merge = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        print(to_merge)


if __name__ == "__main__":
    main()
