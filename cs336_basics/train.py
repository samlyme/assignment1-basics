from pathlib import Path
import torch
import wandb

import argparse

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.utils import (
    MODEL_CONFIGS,
    OPTIM_CONFIGS,
    DatasetConfig,
    TrainConfig,
    dataset_from_config,
    model_from_config,
    load_checkpoint,
    optimizer_from_config,
    save_checkpoint,
)


def train(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    train_config: TrainConfig,
    out_dir: Path,
    log_freq: int = 100,
    save_freq: int = 1000,
    checkpoint: Path | None = None,
    device: torch.Device = None,
    wandb_run: wandb.Run | None = None,
):
    if checkpoint is not None:
        start_iter = load_checkpoint(checkpoint, model, optim)
    else:
        start_iter = 0

    dataset_train = dataset_from_config(train_config.train)
    load_train = iter(
        torch.utils.data.DataLoader(
            dataset_train, batch_size=train_config.batch_size
        )
    )

    dataset_val = dataset_from_config(train_config.val)
    load_val = iter(
        torch.utils.data.DataLoader(
            dataset_val, batch_size=train_config.batch_size
        )
    )

    for iteration in range(start_iter, train_config.steps):
        x_val, y_val = next(load_train)
        x_val = x_val.to(device, dtype=torch.long)
        y_val = y_val.to(device, dtype=torch.long)

        logits = model(x_val)
        loss = cross_entropy(logits, y_val)
        loss.backward()

        optim.step()
        optim.zero_grad()

        if iteration % log_freq == 0:
            train_loss = loss.item()
            x_val, y_val = next(load_val)
            x_val = x_val.to(device, dtype=torch.long)
            y_val = y_val.to(device, dtype=torch.long)
            val_loss = cross_entropy(model(x_val), y_val).item()
            if wandb_run:
                wandb_run.log(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    step=iteration,
                )

        if iteration % save_freq == 0:
            # TODO: add full val set loss here.
            save_checkpoint(
                model, optim, iteration, out_dir / f"{iteration}.pt"
            )

    train_loss = loss.item()
    x_val, y_val = next(load_val)
    x_val = x_val.to(device, dtype=torch.long)
    y_val = y_val.to(device, dtype=torch.long)
    val_loss = cross_entropy(model(x_val), y_val).item()
    if wandb_run:
        wandb_run.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            step=iteration,
        )
    save_checkpoint(model, optim, iteration, out_dir / f"{iteration}-final.pt")
    if wandb_run:
        wandb_run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--train",
        type=Path,
        help="Path to tokenized training data",
    )
    parser.add_argument(
        "--val",
        type=Path,
        help="Path to tokenized validation data",
    )

    parser.add_argument("--model-config", type=str, default="toy")
    parser.add_argument("--optim-config", type=str, default="slow")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--total-tokens", type=int)

    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--saves", type=int, default=5)
    parser.add_argument("--out-dir", type=Path)

    parser.add_argument("--checkpoint", type=Path)

    parser.add_argument("--device", type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.out_dir is None:
        out_dir = Path(
            f"/data/models/{args.model_config}-{args.optim_config}-{args.batch_size}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save models and logs to: {out_dir}")

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

    model_config = MODEL_CONFIGS[args.model_config]
    model = model_from_config(model_config).to(device)
    model.compile()
    print(f"Model configs: {model_config}")

    optim_config = OPTIM_CONFIGS[args.optim_config]
    optim = optimizer_from_config(model, optim_config)
    print(f"Optim configs: {optim_config}")

    context_length = model_config.context_length
    train_config = TrainConfig(
        train=DatasetConfig(
            path=args.train, context_length=model_config.context_length
        ),
        val=DatasetConfig(
            path=args.val, context_length=model_config.context_length
        ),
        batch_size=args.batch_size,
        steps=args.steps,
    )
    if args.total_tokens:
        train_config.steps = args.total_tokens // (
            train_config.batch_size * context_length
        )
        print(
            "INFO: Using total tokens processed to override number of training steps"
        )
    print(f"Train configs: {train_config}")

    LOG_FREQ = args.log_freq
    SAVE_FREQ = train_config.steps // args.saves
    print(f"{LOG_FREQ=}, {SAVE_FREQ=}")
    print("~" * 32)

    checkpoint: Path | None = args.checkpoint

    run = wandb.init(
        entity="canofspam-cal-poly-pomona",
        project="my-awesome-project",
        config={
            "model": model_config,
            "train": train_config,
            "optim": optim_config,
        },
    )

    train(
        model,
        optim,
        train_config,
        out_dir,
        LOG_FREQ,
        SAVE_FREQ,
        checkpoint,
        device,
        run,
    )


if __name__ == "__main__":
    main()
