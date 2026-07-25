from abc import ABC
import argparse
from collections import defaultdict
from dataclasses import dataclass

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


def merge_token(tokens: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    out = []
    flag = False
    for a, b in zip(tokens[:-1], tokens[1:]):
        if (a, b) == pair:
            out.append(a + b)
            flag = False
        else:
            out.append(a)
            flag = True
    if flag:
        out.append(tokens[-1])
    return tuple(out)


def merge_counts(pretoken_counts: PretokenCounts, pair: tuple[bytes, bytes]) -> None:
    to_delete = set()
    to_add = {}
    for token, count in pretoken_counts.items():
        if pair in zip(token[:-1], token[1:]):
            to_delete.add(token)
            to_add[merge_token(token, pair)] = count

    for delete in to_delete:
        del pretoken_counts[delete]

    for k, v in to_add.items():
        pretoken_counts[k] = v


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
        raw = file.read().decode("utf-8", errors="ignore")

        escaped = "".join(regex.split("|".join(["<|endoftext|>"]), raw))

        vocab: dict[int, bytes] = {i: bytes(i) for i in range(256)}

        pretoken_counts: PretokenCounts = defaultdict(int)
        pretokens = split_pretokens(escaped)

        for pretoken in pretokens:
            pretoken_counts[tuple(map(lambda x: x.encode("utf-8"), pretoken.group()))] += 1

        for i in range(args.merges):
            pair_counts = get_stats(pretoken_counts)
            to_merge = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
            merge_counts(pretoken_counts, to_merge)
            print("merged:", to_merge)
            print(pretoken_counts)
            vocab[len(vocab)] = to_merge[0] + to_merge[1]

        print("\nFinal vocab:")
        for i in range(args.merges):
            print(vocab[257 + i])


if __name__ == "__main__":
    main()
