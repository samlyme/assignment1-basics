import argparse
from collections import Counter
from datetime import datetime
from multiprocessing import Pool
import os
from collections.abc import Iterable
import pickle
from typing import BinaryIO

import regex

type TokenId = int
type PretokenId = int
TokenIdPair = tuple[TokenId, TokenId]
Word = tuple[TokenId, ...]
type PretokenIdCounts = Counter[PretokenId]
PairCounts = dict[tuple[TokenId, TokenId], int]


PretokenVocab = dict[PretokenId, Word]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_pretokens(text: Iterable[str]):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for str in text:
        yield from regex.finditer(PAT, str)


def pretokenize(
    input_path: str | os.PathLike, special_tokens: list[str], range: tuple[int, int] | None = None
) -> Counter[bytes]:
    with open(input_path, "rb") as file:
        if range is None:
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)

            range = (0, file_size)

        start, end = range
        file.seek(start)
        raw = file.read(end - start).decode("utf-8", errors="ignore")

    docs = regex.splititer("|".join(map(regex.escape, special_tokens)), raw)

    pretokens = split_pretokens(docs)

    counts = Counter(map(lambda x: x.group().encode("utf-8", errors="ignore"), pretokens))
    return counts


def index_pretokens(counts: Counter[bytes]) -> tuple[PretokenVocab, PretokenIdCounts]:
    pretoken_items = list(counts.items())
    pretoken_items.sort()

    pretoken_vocab = {i: tuple(map(int, bytes)) for i, (bytes, count) in enumerate(pretoken_items)}
    pretoken_id_counts = Counter({i: count for i, (bytes, count) in enumerate(pretoken_items)})

    return pretoken_vocab, pretoken_id_counts


def parallel_pretokenize(input_path: str | os.PathLike, special_tokens: list[str], workers: int) -> Counter[bytes]:
    assert workers > 0

    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, workers, special_tokens[0].encode("utf-8"))
        if len(boundaries) >= 2:
            chunks = zip(boundaries[:-1], boundaries[1:])

            with Pool(workers) as pool:
                tasks = [(input_path, special_tokens, chunk) for chunk in chunks]
                partial_counts = pool.starmap(pretokenize, tasks)

            total: Counter[bytes] = Counter()
            for partial in partial_counts:
                total.update(partial)
        else:
            total = pretokenize(input_path, special_tokens)

    return total


def main():
    parser = argparse.ArgumentParser(description="Read a file as raw bytes.")
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the file to read",
    )
    parser.add_argument("--workers", type=int, default=0, help="Number of worker threads")
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default=f"out/pretokens_{datetime.now()}.pkl",
        help="Where to output trained params.",
    )
    args = parser.parse_args()  # assume we only need "<|endoftext|>" for now

    if args.workers == 0:
        results = pretokenize(args.input_path, ["<|endoftext|>"])
    else:
        results = parallel_pretokenize(args.input_path, ["<|endoftext|>"], args.workers)

    with open(args.output_path, "wb") as out:
        pickle.dump(results, out)


if __name__ == "__main__":
    main()
