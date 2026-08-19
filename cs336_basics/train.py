from pathlib import Path
import torch

import numpy as np
import argparse

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import get_batch, load_checkpoint, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--train",
        type=str,
        help="Path to tokenized training data",
    )
    parser.add_argument(
        "--val",
        type=str,
        help="Path to tokenized validation data",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=256)

    parser.add_argument("--steps", type=int, default=40000)

    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("--device", type=str)

    parser.add_argument("--out-dir", type=str)

    args = parser.parse_args()

    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = np.load(args.train, mmap_mode="r")
    val_dataset = np.load(args.val, mmap_mode="r")

    # TODO: load model config from arg
    model = TransformerLM(
        vocab_size=10000,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        context_length=256,
        rope_theta=10000,
    ).to(device)

    # TODO: load optimizer
    optimizer = AdamW(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0
    )

    start_iter = 0
    if args.checkpoint is not None:
        start_iter = load_checkpoint(args.checkpoint, model, optimizer)

    # overfit test.
    for iteration in range(start_iter, args.steps):
        x, y = get_batch(
            dataset, args.batch_size, args.context_length, args.device
        )

        x = x.long()
        y = y.long()

        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if iteration % 1000 == 0:
            train_loss = loss.item()
            x_val, y_val = get_batch(
                val_dataset, args.batch_size, args.context_length, args.device
            )
            x_val = x_val.long()
            y_val = y_val.long()
            val_loss = cross_entropy(model(x_val), y_val).item()
            print(f"{iteration=}, {train_loss=:.3f}, {val_loss=:.3f}")

            save_checkpoint(
                model, optimizer, iteration, out_dir / f"{iteration}.pt"
            )


if __name__ == "__main__":
    main()
