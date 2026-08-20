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
    k: int | None = None,
    device: torch.device | None = None,
) -> str:
    tokens = tokenizer.encode(prompt)

    if k is not None and k < 1:
        raise ValueError("k must be at least 1")

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(
                torch.tensor(tokens, dtype=torch.long, device=device)
            )[-1]

            if temperature > 0:
                scaled_logits = logits / temperature

                if k is not None and k < scaled_logits.size(-1):
                    # Restrict sampling to the k most likely tokens.
                    top_k_logits, top_k_indices = torch.topk(
                        scaled_logits,
                        k=k,
                        dim=-1,
                    )
                    probabilities = softmax(top_k_logits, dim=-1)

                    sampled_index = torch.multinomial(
                        probabilities,
                        num_samples=1,
                    )
                    next_token = top_k_indices[sampled_index]
                else:
                    probabilities = softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(
                        probabilities,
                        num_samples=1,
                    )
            else:
                # Temperature <= 0 means deterministic greedy decoding.
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            tokens.append(int(next_token.item()))

            if tokenizer.decode(tokens[-1:]) == "<|endoftext|>":
                break

    return tokenizer.decode(tokens)


def main():
    pass


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("prompt", type=str)

    # load from training checkpoint
    parser.add_argument("--checkpoint", type=str)

    # tokenizer configs
    parser.add_argument(
        "--vocab-filepath",
        type=str,
        default="out/bpe_params_ts-train/vocab.pkl",
    )
    parser.add_argument(
        "--merges-filepath",
        type=str,
        default="out/bpe_params_ts-train/merges.pkl",
    )

    parser.add_argument("--temperature", type=float, default=1.0)

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

    print(
        generate_text(
            model,
            tokenizer,
            args.prompt,
            temperature=args.temperature,
            device=device,
        )
    )
