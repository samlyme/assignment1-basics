from abc import ABC


class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""

    def __init__(self, vocab, merges, special_tokens=None):
        raise NotImplementedError

    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


class BPETokenizer(Tokenizer):
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def decode(self, indices: list[int]) -> str:
        out = b"".join(self.vocab[index] for index in indices)
        return out.decode(encoding="utf-8", errors="replace")
