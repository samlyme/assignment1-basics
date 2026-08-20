from datetime import datetime
from pathlib import Path
import torch

import numpy as np
import argparse

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import get_batch, load_checkpoint, save_checkpoint

import sys


# Define some hparams
MODEL_CONFIGS = {
    "toy": {
        "vocab_size": 10000,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "context_length": 256,
        "rope_theta": 10000,
    },
    "toy-long-context": {
        "vocab_size": 10000,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "context_length": 512,
        "rope_theta": 10000,
    },
    "small": {
        "vocab_size": 10000,
        "d_model": 768,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 2048,
        "context_length": 256,
        "rope_theta": 10000,
    },
    "small-long-context": {
        "vocab_size": 10000,
        "d_model": 768,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 2048,
        "context_length": 512,
        "rope_theta": 10000,
    },
}

OPTIM_CONFIGS = {
    "slower": {
        "lr": 1.5e-4,
        "betas": (0.9, 0.999),
        "weight_decay": 0,
    },
    "slow": {
        "lr": 3e-4,
        "betas": (0.9, 0.999),
        "weight_decay": 0,
    },
    "fast": {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 0,
    },
    "faster": {  # most of the time unstable
        "lr": 2e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 0,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--train",
        type=str,
        help="Path to tokenized training data",
        default="data/ts-train-ts-10000.npy",
    )
    parser.add_argument(
        "--val",
        type=str,
        help="Path to tokenized validation data",
        default="data/ts-valid-ts-10000.npy",
    )

    parser.add_argument("--model-config", type=str, default="toy")
    parser.add_argument("--optim-config", type=str, default="slow")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--total-tokens", type=int)

    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--saves", type=int, default=5)

    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("--device", type=str)

    parser.add_argument("--out-dir", type=str)

    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(
            f"out/{args.model_config}-{args.optim_config}-{args.batch_size}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save models and logs to: {out_dir}")

    # NOTE: this is cursed. Implicitly redirecting everything into that log.
    log = open(out_dir / "train.log", "a", buffering=1)
    sys.stdout = log
    sys.stderr = log

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        # pytorch warning said this is good :D
        torch.set_float32_matmul_precision("high")
    elif torch.mps.is_available():
        device = torch.device("mps")
        # DO NOT USE `torch.set_float32_matmul_precision("high")` here.
        # assignment claims this causes silent failures.
    else:
        device = torch.device("cpu")

    print(f"USING DEVICE: {device}")
    # use mmap_mode="c" because it means "copy on write".
    # This way, we garuantee that the dataset is unharmed, but we get rid of
    # that warning.
    dataset = np.load(args.train, mmap_mode="c")
    val_dataset = np.load(args.val, mmap_mode="c")
    print(f"Training dataset: {args.train}")
    print(f"Val dataset: {args.val}")

    model_config = MODEL_CONFIGS[args.model_config]
    model = TransformerLM(**model_config).to(device)
    model.compile()
    print(f"Model configs: {model_config}")

    optim_config = OPTIM_CONFIGS[args.optim_config]
    optim = AdamW(model.parameters(), **optim_config)
    print(f"Optim configs: {optim_config}")

    context_length = model_config["context_length"]
    train_configs = {"batch_size": args.batch_size, "steps": args.steps}
    if args.total_tokens:
        train_configs["steps"] = args.total_tokens // (
            train_configs["batch_size"] * context_length
        )
        print(
            "INFO: Using total tokens processed to override number of training steps"
        )
    print(f"Train configs: {train_configs}")

    LOG_FREQ = args.log_freq
    SAVE_FREQ = train_configs["steps"] // args.saves
    print(f"{LOG_FREQ=}, {SAVE_FREQ=}")
    print("~" * 32)

    start_iter = 0
    if args.checkpoint is not None:
        start_iter = load_checkpoint(args.checkpoint, model, optim)
    for iteration in range(start_iter, args.steps):
        x, y = get_batch(dataset, args.batch_size, context_length, device)

        x = x.long()
        y = y.long()

        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()

        optim.step()
        optim.zero_grad()

        if iteration % LOG_FREQ == 0:
            train_loss = loss.item()
            x_val, y_val = get_batch(
                val_dataset,
                args.batch_size,
                context_length,
                device,
            )
            x_val = x_val.long()
            y_val = y_val.long()
            val_loss = cross_entropy(model(x_val), y_val).item()
            time = datetime.now()
            print(
                f"{time.strftime('%H:%M:%S')}, {iteration=}, {train_loss=:.3f}, {val_loss=:.3f}"
            )

        if iteration % SAVE_FREQ == 0:
            # TODO: add full val set loss here.
            save_checkpoint(
                model, optim, iteration, out_dir / f"{iteration}.pt"
            )

    train_loss = loss.item()
    x_val, y_val = get_batch(
        val_dataset, args.batch_size, context_length, device
    )
    x_val = x_val.long()
    y_val = y_val.long()
    val_loss = cross_entropy(model(x_val), y_val).item()
    print(
        f"{time.strftime('%H:%M:%S')}, {iteration=}, {train_loss=:.3f}, {val_loss=:.3f}, final"
    )
    save_checkpoint(model, optim, iteration, out_dir / f"{iteration}-final.pt")


if __name__ == "__main__":
    main()
