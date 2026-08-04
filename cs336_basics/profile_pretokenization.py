import argparse
import time

from cs336_basics.pretokenizer import parallel_pretokenize, pretokenize


def main():
    parser = argparse.ArgumentParser(description="Read a file as raw bytes.")
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the file to read",
    )

    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]

    start = time.perf_counter()
    pretokenize(args.input_path, special_tokens)
    end = time.perf_counter()
    print(f"Serial:\t{end - start:.4f}")

    start = time.perf_counter()
    parallel_pretokenize(args.input_path, special_tokens, 2)
    end = time.perf_counter()
    print(f"Parallel 2:\t{end - start:.4f}")

    start = time.perf_counter()
    parallel_pretokenize(args.input_path, special_tokens, 4)
    end = time.perf_counter()
    print(f"Parallel 4:\t{end - start:.4f}")

    start = time.perf_counter()
    parallel_pretokenize(args.input_path, special_tokens, 8)
    end = time.perf_counter()
    print(f"Parallel 8:\t{end - start:.4f}")


if __name__ == "__main__":
    main()
