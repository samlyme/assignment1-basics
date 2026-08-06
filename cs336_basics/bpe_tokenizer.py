import argparse
from collections.abc import Iterable
from collections.abc import Iterator
import pickle

import regex

from cs336_basics.pretokenizer import split_pretokens


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.vocab_index = {v: k for k, v in vocab.items()}
        self.merges = merges
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

    def _apply_merge(self, symbol: tuple[bytes, ...], merge: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        if len(symbol) < 2:
            return symbol
        if merge not in zip(symbol[:-1], symbol[1:]):
            return symbol

        out = []
        i = 0
        while i < len(symbol) - 1:
            a, b = symbol[i], symbol[i + 1]

            if (a, b) == merge:
                out.append(a + b)
                i += 2
            else:
                out.append(a)
                i += 1

        if i < len(symbol):
            out.append(symbol[i])

        return tuple(out)

    def encode(self, string: str) -> list[int]:
        out: list[int] = []

        if self.special_tokens_sorted:
            separator = "|".join(map(regex.escape, self.special_tokens_sorted))
            docs = regex.splititer(f"({separator})", string)
        else:
            docs = [string]

        for doc in docs:
            if doc in self.special_tokens:
                out.append(self.vocab_index[doc.encode("utf-8", errors="ignore")])
                continue

            pretokens = split_pretokens([doc])

            for pretoken in pretokens:
                pretoken_str = pretoken.group()
                pretoken_bytes = pretoken_str.encode("utf-8", errors="ignore")

                symbol = tuple(bytes([b]) for b in pretoken_bytes)

                for merge in self.merges:
                    symbol = self._apply_merge(symbol, merge)

                for token in symbol:
                    out.append(self.vocab_index[token])

        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for str in iterable:
            yield from self.encode(str)

    def decode(self, indices: list[int]) -> str:
        out = b"".join(self.vocab[i] for i in indices)
        return out.decode("utf-8", errors="ignore")


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

    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_filepath, args.merges_filepath, ["<|endoftext|>"])

    with open(args.input_filepath, "rb") as file:
        input = file.read().decode()

    print(tokenizer.encode(input))


if __name__ == "__main__":
    main()
