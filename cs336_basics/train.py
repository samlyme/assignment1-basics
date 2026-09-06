from dataclasses import asdict
from pathlib import Path
import torch
import wandb

import argparse

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.utils import (
    MODEL_CONFIGS,
    OPTIM_CONFIGS,
    DatasetConfig,
    RunConfig,
    TrainConfig,
    make_dataloader,
    model_from_config,
    load_checkpoint,
    optimizer_from_config,
    parse_run_config,
    save_checkpoint,
)


def run_training(
    config: RunConfig,
    out_dir: Path,
    log_freq: int = 100,
    save_freq: int = 1000,
    checkpoint: Path | None = None,
    device: torch.types.Device = None,
    wandb_run: wandb.Run | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    model = model_from_config(config.model).to(device)
    model.train()
    model.compile()

    optim = optimizer_from_config(model, config.optim)

    if checkpoint is not None:
        start_iter = load_checkpoint(checkpoint, model, optim)
    else:
        start_iter = 0

    load_train = iter(
        make_dataloader(config.train.train, config.train.batch_size)
    )

    load_val = iter(make_dataloader(config.train.val, config.train.batch_size))

    for iteration in range(start_iter, config.train.steps):
        x_train, y_train = next(load_train)
        x_train = x_train.to(device, dtype=torch.long)
        y_train = y_train.to(device, dtype=torch.long)

        logits = model(x_train)
        loss = cross_entropy(logits, y_train)
        loss.backward()

        optim.step()
        optim.zero_grad()

        if iteration % log_freq == 0:
            model.eval()

            with torch.inference_mode():
                # recompute train loss after optim step.
                train_loss = cross_entropy(model(x_train), y_train).item()

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

            model.train()

        if iteration % save_freq == 0:
            # TODO: pass info to wandb
            save_checkpoint(
                model, optim, iteration, out_dir / f"{iteration}.pt"
            )

    train_loss = loss.item()
    x_train, y_train = next(load_val)
    x_train = x_train.to(device, dtype=torch.long)
    y_train = y_train.to(device, dtype=torch.long)
    val_loss = cross_entropy(model(x_train), y_train).item()
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

    parser.add_argument("--config", type=str)

    return parser.parse_args()


def parse_config(args: argparse.Namespace) -> RunConfig:
    model_config = MODEL_CONFIGS[args.model_config]
    # model = model_from_config(model_config).to(device)
    # model.compile()
    # print(f"Model configs: {model_config}")

    optim_config = OPTIM_CONFIGS[args.optim_config]
    # optim = optimizer_from_config(model, optim_config)
    # print(f"Optim configs: {optim_config}")

    context_length = model_config.context_length
    train_config = TrainConfig(
        train=DatasetConfig(
            path=args.train,
            seq_len=context_length,
            random_sample=True,
        ),
        val=DatasetConfig(
            path=args.val,
            seq_len=context_length,
            random_sample=True,
        ),
        batch_size=args.batch_size,
        steps=args.steps,
    )
    if args.total_tokens:
        train_config.steps = args.total_tokens // (
            train_config.batch_size * context_length
        )

    config = RunConfig(
        model=model_config,
        optim=optim_config,
        train=train_config,
    )

    return config


def main():
    args = parse_args()

    if args.config:
        print("Overriding all other args with '--config'")
        config = parse_run_config(args.config)
    else:
        config = parse_config(args)

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

    run = wandb.init(
        entity="canofspam-cal-poly-pomona",
        project="cs336-a1",
        config=asdict(config),
    )
    if args.out_dir is None:
        out_dir = Path(f"/data/models/{run.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save models and logs to: {out_dir}")

    LOG_FREQ = args.log_freq
    SAVE_FREQ = config.train.steps // args.saves

    checkpoint: Path | None = args.checkpoint

    run_training(
        config,
        out_dir,
        LOG_FREQ,
        SAVE_FREQ,
        checkpoint,
        device,
        run,
    )


if __name__ == "__main__":
    main()
