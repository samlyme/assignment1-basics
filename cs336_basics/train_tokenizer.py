from abc import ABC
import argparse
from collections import Counter, defaultdict
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
type PretokenId = int
TokenIdPair = tuple[TokenId, TokenId]
Word = tuple[TokenId, ...]
type PretokenIdCounts = Counter[PretokenId]
PairCounts = dict[tuple[TokenId, TokenId], int]


PretokenVocab = dict[PretokenId, Word]


def pretokenize(raw: str, special_tokens: list[str]) -> tuple[PretokenVocab, PretokenIdCounts]:
    pretoken_vocab = {}
    seen = set()

    pretoken_counts = Counter()

    docs = regex.splititer("|".join(map(regex.escape, special_tokens)), raw)
    pretokens = split_pretokens(docs)

    for pretoken in pretokens:
        # Can't immediately use pretoken ID's since i want to parallelize later.
        word = tuple(map(int, pretoken.group().encode("utf-8", errors="ignore")))
        pretoken_counts[word] += 1

        if word not in seen:
            seen.add(word)
            pretoken_vocab[len(pretoken_vocab)] = word

    pretoken_id_counts = Counter(
        {pretoken_id: pretoken_counts[pretoken] for pretoken_id, pretoken in pretoken_vocab.items()}
    )
    return pretoken_vocab, pretoken_id_counts


type TokenVocab = dict[int, bytes]
type TokenPairCounts = Counter[TokenIdPair]
type TokenPairInPretokens = dict[TokenIdPair, set[PretokenId]]


def init_stats(
    pretoken_vocab: PretokenVocab, pretoken_id_counts: PretokenIdCounts
) -> tuple[TokenVocab, TokenPairCounts, TokenPairInPretokens]:
    token_vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    token_pair_counts: TokenPairCounts = Counter()
    token_pair_in_pretokens: TokenPairInPretokens = defaultdict(set)

    for pretoken_id, counts in pretoken_id_counts.items():
        pretoken_repr = pretoken_vocab[pretoken_id]

        if len(pretoken_repr) >= 2:
            for pair in zip(pretoken_repr[:-1], pretoken_repr[1:]):
                token_pair_counts[pair] += counts
                token_pair_in_pretokens[pair].add(pretoken_id)

    assert len(token_pair_counts) != 0
    assert len(token_pair_in_pretokens) != 0

    return token_vocab, token_pair_counts, token_pair_in_pretokens


def merge_pair(word: Word, to_merge: TokenIdPair, merged_token_id: TokenId) -> Word:
    out: list[TokenId] = []
    i = 0
    while i < len(word) - 1:
        a, b = word[i], word[i + 1]
        if (a, b) == to_merge:
            out.append(merged_token_id)
            i += 2
        else:
            out.append(a)
            i += 1

    if i < len(word):
        out.append(word[i])

    return tuple(out)


type PretokenVocabDiff = dict[PretokenId, tuple[Word, Word]]


def update_pretoken_vocab(
    pretoken_vocab: PretokenVocab, affected_pretoken_ids: set[PretokenId], to_merge: TokenIdPair, new_token_id: TokenId
):
    pretoken_vocab_diff = {
        pretoken_id: (pretoken_vocab[pretoken_id], merge_pair(pretoken_vocab[pretoken_id], to_merge, new_token_id))
        for pretoken_id in affected_pretoken_ids
    }
    for id, (old, new) in pretoken_vocab_diff.items():
        assert pretoken_vocab[id] == old
        pretoken_vocab[id] = new

    return pretoken_vocab_diff


def token_pair_counts_delta(
    pretoken_repr_old_to_new: dict[PretokenId, tuple[Word, Word]], pretoken_id_counts: PretokenIdCounts
) -> TokenPairCounts:
    old: Counter[TokenIdPair] = Counter()
    new: Counter[TokenIdPair] = Counter()

    for pretoken_id, (repr_old, repr_new) in pretoken_repr_old_to_new.items():
        if len(repr_old) >= 2:
            for pair in zip(repr_old[:-1], repr_old[1:]):
                old[pair] += pretoken_id_counts[pretoken_id]
        if len(repr_new) >= 2:
            for pair in zip(repr_new[:-1], repr_new[1:]):
                new[pair] += pretoken_id_counts[pretoken_id]

    new.subtract(old)
    return new


def update_token_pair_counts(
    token_pair_counts: TokenPairCounts,
    pretoken_id_counts: PretokenIdCounts,
    pretoken_vocab: PretokenVocab,
    pretoken_vocab_diff: dict[PretokenId, tuple[Word, Word]],
):
    delta = token_pair_counts_delta(pretoken_vocab_diff, pretoken_id_counts)
    token_pair_counts += delta


def token_pair_from_delta(
    pretoken_repr_old_to_new: dict[PretokenId, tuple[Word, Word]],
) -> tuple[TokenPairInPretokens, TokenPairInPretokens]:
    old: TokenPairInPretokens = defaultdict(set)
    new: TokenPairInPretokens = defaultdict(set)

    for pretoken_id, (repr_old, repr_new) in pretoken_repr_old_to_new.items():
        if len(repr_old) >= 2:
            for pair in zip(repr_old[:-1], repr_old[1:]):
                old[pair].add(pretoken_id)
        if len(repr_new) >= 2:
            for pair in zip(repr_new[:-1], repr_new[1:]):
                new[pair].add(pretoken_id)

    assert all(new[k] <= old[k] for k in old.keys())

    remove = {k: old[k] - new[k] for k in old.keys()}
    add = {k: new[k] - old[k] for k in new.keys()}
    return remove, add


def update_token_pair_from(
    token_pair_from: TokenPairInPretokens,
    pretoken_vocab: PretokenVocab,
    delta_pretoken_vocab: dict[PretokenId, tuple[Word, Word]],
):
    remove, add = token_pair_from_delta(delta_pretoken_vocab)
    for pair, pretoken_ids in remove.items():
        token_pair_from[pair] -= pretoken_ids
    for pair, pretoken_ids in add.items():
        token_pair_from[pair].update(pretoken_ids)


# TODO: make this use ID's internally. Then use a global delta-based approach
# for the pair counts.
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    with open(input_path, "rb") as file:
        raw = file.read().decode("utf-8", errors="ignore")

    pretoken_vocab, pretoken_id_counts = pretokenize(raw, special_tokens)
    assert len(pretoken_vocab) == len(pretoken_id_counts)
    assert len(pretoken_vocab) > 0

    token_vocab, token_pair_counts, token_pair_from = init_stats(pretoken_vocab, pretoken_id_counts)
    assert all(bytes([k]) == v for k, v in token_vocab.items())
    assert len(token_vocab) < vocab_size

    merges: list[tuple[bytes, bytes]] = []
    while len(token_vocab) < vocab_size - len(special_tokens):
        to_merge, _count = max(
            token_pair_counts.items(), key=lambda item: (item[1], (token_vocab[item[0][0]], token_vocab[item[0][1]]))
        )
        merges.append((token_vocab[to_merge[0]], token_vocab[to_merge[1]]))

        new_token_id = len(token_vocab)
        token_vocab[new_token_id] = token_vocab[to_merge[0]] + token_vocab[to_merge[1]]

        affected_pretoken_ids = token_pair_from[to_merge]

        # Apply updates
        pretoken_vocab_diff = update_pretoken_vocab(pretoken_vocab, affected_pretoken_ids, to_merge, new_token_id)

        update_token_pair_counts(token_pair_counts, pretoken_id_counts, pretoken_vocab, pretoken_vocab_diff)

        update_token_pair_from(token_pair_from, pretoken_vocab, pretoken_vocab_diff)

    for special_token in special_tokens:
        token_vocab[len(token_vocab)] = special_token.encode("utf-8")

    return token_vocab, merges


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
