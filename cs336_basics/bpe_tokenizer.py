import argparse
from collections.abc import Iterable
from collections.abc import Iterator
import heapq
import pickle

import numpy as np
import regex

from cs336_basics.pretokenizer import split_pretokens


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.vocab_index = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_index = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens: set[str] = set(special_tokens) if special_tokens else set()
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)
            assert type(vocab) is dict and all(type(k) is int and type(v) is bytes for k, v in vocab.items())

        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)
            assert type(merges) is list and all(
                type(i) is tuple and type(i[0]) is bytes and type(i[1]) is bytes for i in merges
            )

        return Tokenizer(vocab, merges, special_tokens)

    def _apply_merge(
        self, symbol: tuple[bytes, ...], merge: tuple[bytes, bytes]
    ) -> tuple[tuple[bytes, ...], list[int]]:
        if len(symbol) < 2:
            return symbol, []
        if merge not in zip(symbol[:-1], symbol[1:]):
            return symbol, []

        out = []
        out_indices = []  # which indices are the is the new item?
        i = 0
        while i < len(symbol) - 1:
            a, b = symbol[i], symbol[i + 1]

            if (a, b) == merge:
                out_indices.append(len(out))
                out.append(a + b)
                i += 2
            else:
                out.append(a)
                i += 1

        if i < len(symbol):
            out.append(symbol[i])

        return tuple(out), out_indices

    def encode(self, string: str) -> list[int]:
        out: list[int] = []

        if self.special_tokens_sorted:
            separator = "|".join(map(regex.escape, self.special_tokens_sorted))
            docs = regex.splititer(f"({separator})", string)
        else:
            docs = [string]

        for doc in docs:
            if doc in self.special_tokens:
                out.append(self.vocab_index[doc.encode("utf-8")])
                continue

            pretokens = split_pretokens([doc])

            for pretoken in pretokens:
                pretoken_str = pretoken.group()
                pretoken_bytes = pretoken_str.encode("utf-8")

                symbol = tuple(bytes([b]) for b in pretoken_bytes)

                assert len(symbol) >= 1
                if len(symbol) == 1:
                    out.append(self.vocab_index[symbol[0]])
                    continue

                merge_queue = [
                    (self.merges_index[pair], pair)
                    for pair in zip(symbol[:-1], symbol[1:])
                    if pair in self.merges_index
                ]
                heapq.heapify(merge_queue)

                while merge_queue:
                    # this is a bad assertion, merges can remove pairs,
                    # but we do not remove them from the queue
                    # assert len(symbol) > 1

                    _, merge = heapq.heappop(merge_queue)
                    symbol, affected = self._apply_merge(symbol, merge)
                    for i in affected:
                        left, right = i - 1, i + 1
                        if left >= 0:
                            pair = symbol[left], symbol[i]
                            if pair in self.merges_index:
                                rank = self.merges_index[pair]
                                heapq.heappush(merge_queue, (rank, pair))
                        if right < len(symbol):
                            pair = symbol[i], symbol[right]
                            if pair in self.merges_index:
                                rank = self.merges_index[pair]
                                heapq.heappush(merge_queue, (rank, pair))

                for token in symbol:
                    out.append(self.vocab_index[token])

        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for str in iterable:
            yield from self.encode(str)

    def decode(self, indices: list[int]) -> str:
        out = b"".join(self.vocab[i] for i in indices)
        return out.decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "vocab_filepath",
        type=str,
    )
    parser.add_argument(
        "merges_filepath",
        type=str,
    )
    parser.add_argument("input_filepath", type=str)
    parser.add_argument("output_filepath", type=str)

    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_filepath, args.merges_filepath, ["<|endoftext|>"])

    with open(args.input_filepath, "rb") as file:
        input = file.read().decode()

    res = tokenizer.encode(input)
    tokens = np.array(res, dtype=np.uint16)

    np.save(args.output_file, tokens)


if __name__ == "__main__":
    main()
