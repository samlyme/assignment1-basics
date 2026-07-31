import argparse

from cs336_basics.train_tokenizer import pretokenize


def main():
    parser = argparse.ArgumentParser(description="Read a file as raw bytes.")
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the file to read",
    )

    args = parser.parse_args()

    with open(args.input_path, "rb") as file:
        raw = file.read().decode("utf-8", errors="ignore")
    special_tokens = ["<|endoftext|>"]

    pretoken_vocab, pretoken_id_counts = pretokenize(raw, special_tokens)


if __name__ == "__main__":
    main()
