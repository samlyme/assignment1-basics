from argparse import ArgumentParser

import torch

from cs336_basics.bpe_tokenizer import Tokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import softmax
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import load_checkpoint


def generate_text(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 1.0,
    device: torch.device | None = None,
):
    tokens = tokenizer.encode(prompt)

    # assume that we dont have batched inputs yet
    for i in range(max_tokens):
        probs = softmax(
            model(torch.tensor(tokens, dtype=torch.long, device=device)), dim=1
        )

        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens.append(int(next_token))

    return tokenizer.decode(tokens)


def main():
    pass


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("prompt", type=str, default="<|endoftext|>")

    # load from training checkpoint
    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("vocab_filepath", type=str)  # tokenizer configs
    parser.add_argument("merges_filepath", type=str)

    parser.add_argument("--device", type=str)

    args = parser.parse_args()

    device = torch.device(args.device)

    tokenizer = Tokenizer.from_files(
        args.vocab_filepath, args.merges_filepath, ["<|endoftext|>"]
    )
    model = TransformerLM(
        vocab_size=10000,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        context_length=256,
        rope_theta=10000,
    ).to(device)

    # TODO: separate model state dict
    # dummy optimizer to fit api LOL.
    optimizer = AdamW(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0
    )
    start_iter = load_checkpoint(args.checkpoint, model, optimizer)

    generate_text(model, tokenizer, args.prompt, device=device)
