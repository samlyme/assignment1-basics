from abc import ABC
import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import os
import pickle
from collections.abc import Iterable

import regex


class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""

    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


def split_pretokens(text: Iterable[str]):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for str in text:
        yield from regex.finditer(PAT, str)


type TokenId = int
TokenPair = tuple[TokenId, TokenId]
Word = tuple[TokenId, ...]
PretokenCounts = dict[Word, int]
PairCounts = dict[tuple[TokenId, TokenId], int]


def get_stats(pretoken_counts: PretokenCounts) -> PairCounts | None:
    out = defaultdict(int)
    for pretoken, count in pretoken_counts.items():
        if len(pretoken) < 2:
            continue

        for a, b in zip(pretoken[:-1], pretoken[1:]):
            out[(a, b)] += count
    return out if out else None


def merge_token(tokens: Word, pair: TokenPair, newTokenId: TokenId) -> Word:
    assert len(tokens) >= 2

    out: list[TokenId] = []
    i = 0
    while i < len(tokens) - 1:
        a, b = tokens[i], tokens[i + 1]
        if (a, b) == pair:
            out.append(newTokenId)
            i += 2
        else:
            out.append(a)
            i += 1

    if i < len(tokens):
        out.append(tokens[i])

    return tuple(out)


def merge_counts(pretoken_counts: PretokenCounts, pair: TokenPair, newTokenId: TokenId) -> None:
    to_delete = set()
    to_add = {}
    for token, count in pretoken_counts.items():
        if len(token) < 2:
            continue

        if pair in zip(token[:-1], token[1:]):
            to_delete.add(token)
            to_add[merge_token(token, pair, newTokenId)] = count

    for delete in to_delete:
        del pretoken_counts[delete]

    for k, v in to_add.items():
        pretoken_counts[k] = v


# TODO: make this use ID's internally. Then use a global delta-based approach
# for the pair counts.
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges: list[tuple[bytes, bytes]] = []

    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    assert all(bytes([k]) == v for k, v in vocab.items())
    assert len(vocab) < vocab_size

    with open(input_path, "rb") as file:
        raw = file.read().decode("utf-8", errors="ignore")
        docs = regex.splititer("|".join(map(regex.escape, special_tokens)), raw)
        pretokens = split_pretokens(docs)
        pretoken_counts: PretokenCounts = defaultdict(int)

        for pretoken in pretokens:
            word = tuple(map(int, pretoken.group().encode("utf-8", errors="ignore")))
            pretoken_counts[word] += 1

        while len(vocab) < vocab_size - len(special_tokens):
            pair_counts = get_stats(pretoken_counts)
            if pair_counts is None:
                break
            to_merge = max(pair_counts.items(), key=lambda x: (x[1], (vocab[x[0][0]], vocab[x[0][1]])))[0]
            newTokenId = len(vocab)
            vocab[newTokenId] = vocab[to_merge[0]] + vocab[to_merge[1]]
            merges.append((vocab[to_merge[0]], vocab[to_merge[1]]))
            merge_counts(pretoken_counts, to_merge, newTokenId)

        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode("utf-8")

    return vocab, merges


@dataclass(frozen=True)
class BPEParams:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


def main():
    parser = argparse.ArgumentParser(description="Read a file as raw bytes.")
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the file to read",
    )
    parser.add_argument("--vocab-size", type=int, default=256 + 17, help="Final vocab size")
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default=f"out/bpe_params_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl",
        help="Where to output trained params.",
    )
    args = parser.parse_args()

    vocab, merges = train_bpe(args.input_path, args.vocab_size, ["<|endoftext|>"])

    params = BPEParams(vocab, merges)
    with open(args.output_path, "wb") as out:
        pickle.dump(params, out)


if __name__ == "__main__":
    main()
