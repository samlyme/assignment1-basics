from collections.abc import Iterable
import itertools
from pathlib import Path
import torch
import wandb

import argparse

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.utils import (
    RunConfig,
    make_dataloader,
    model_from_config,
    load_checkpoint,
    optimizer_from_config,
    save_checkpoint,
)


def run_training(
    config: RunConfig,
    out_dir: Path,
    log_train_freq: int = 100,
    log_val_freq: int = 500,
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

    loader_train_iter = iter(
        make_dataloader(config.train.train, config.train.batch_size)
    )

    loader_val = make_dataloader(config.train.val, config.train.batch_size)
    # we want to take ~100k tokens to evaluate.
    num_val_batches = int(
        100_000 / config.train.batch_size / config.model.context_length
    )
    val_batches = list(itertools.islice(loader_val, num_val_batches))

    for iteration in range(start_iter, config.train.steps):
        x_train, y_train = next(loader_train_iter)
        x_train = x_train.to(device, dtype=torch.long)
        y_train = y_train.to(device, dtype=torch.long)

        logits = model(x_train)
        loss = cross_entropy(logits, y_train)
        loss.backward()

        optim.step()
        optim.zero_grad()

        if iteration % log_train_freq == 0:
            # recompute train loss after optim step.
            if wandb_run:
                wandb_run.log(
                    {
                        "train_loss": evaluate_model(
                            model, [(x_train, y_train)], device
                        )
                    },
                    step=iteration,
                )

        if iteration % log_val_freq == 0:
            if wandb_run:
                wandb_run.log(
                    {"val_loss": evaluate_model(model, val_batches, device)},
                    step=iteration,
                )

        if iteration % save_freq == 0:
            # TODO: pass info to wandb
            save_checkpoint(
                model, optim, iteration, out_dir / f"{iteration}.pt"
            )

    save_checkpoint(model, optim, iteration, out_dir / f"{iteration}.pt")
    if wandb_run:
        wandb_run.finish()


def evaluate_model(
    model: torch.nn.Module,
    val_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.types.Device = None,
):
    was_training = model.training
    model.eval()

    with torch.inference_mode():
        val_loss_total = 0
        num_batches = 0
        for x_val, y_val in val_batches:
            x_val = x_val.to(device, dtype=torch.long)
            y_val = y_val.to(device, dtype=torch.long)
            val_loss_total += cross_entropy(model(x_val), y_val).item()
            num_batches += 1
        val_loss = val_loss_total / num_batches

    model.train(was_training)
    return val_loss


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=RunConfig.model_validate_json)
    parser.add_argument("--device", type=torch.device, default="cuda")

    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--saves", type=int, default=5)

    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    config: RunConfig = args.config

    device: torch.device = args.device
    if device.type.find("cuda") != -1:
        torch.set_float32_matmul_precision("high")

    with wandb.init(
        entity="canofspam-cal-poly-pomona",
        project="cs336-a1",
        config=config.model_dump(),
    ) as run:
        out_dir = Path(f"/data/models/{run.name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Save models and logs to: {out_dir}")

        log_loss_freq = args.log_freq
        log_val_freq = 500
        save_freq = config.train.steps // args.saves

        checkpoint: Path | None = args.checkpoint

        run_training(
            config,
            out_dir,
            log_loss_freq,
            log_val_freq,
            save_freq,
            checkpoint,
            device,
            run,
        )


if __name__ == "__main__":
    main()
