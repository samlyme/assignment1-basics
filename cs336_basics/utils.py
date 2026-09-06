import os
from pathlib import Path
import typing

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
import numpy as np
import torch
import numpy.typing as npt
from cs336_basics.nn_utils import clip_gradient, cross_entropy

from pydantic import BaseModel


class TransformerLMConfig(BaseModel):
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    context_length: int
    rope_theta: float


ModelConfig = TransformerLMConfig


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "toy": TransformerLMConfig(
        vocab_size=10000,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        context_length=256,
        rope_theta=10000,
    ),
    "small": TransformerLMConfig(
        vocab_size=10000,
        d_model=768,
        num_layers=4,
        num_heads=16,
        d_ff=2048,
        context_length=256,
        rope_theta=10000,
    ),
}


class AdamWConfig(BaseModel):
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0


OptimizerConfig = AdamWConfig

OPTIM_CONFIGS: dict[str, OptimizerConfig] = {
    "slower": AdamWConfig(
        lr=1.5e-4,
        betas=(0.9, 0.999),
        weight_decay=0,
    ),
    "slow": AdamWConfig(
        lr=3e-4,
        betas=(0.9, 0.999),
        weight_decay=0,
    ),
    "fast": AdamWConfig(
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0,
    ),
}


class DatasetConfig(BaseModel):
    path: Path
    seq_len: int
    random_sample: bool = True


class TrainConfig(BaseModel):
    train: DatasetConfig
    val: DatasetConfig
    batch_size: int = 32
    steps: int = 40_000


class RunConfig(BaseModel):
    model: ModelConfig
    optim: OptimizerConfig
    train: TrainConfig


def parse_run_config(str: str) -> RunConfig:
    return RunConfig.model_validate_json(str)


class NextTokenDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: npt.NDArray[np.integer],
        context_length: int,
    ) -> None:
        if data.ndim != 1:
            raise ValueError("data must be one-dimensional")
        if len(data) <= context_length:
            raise ValueError("data must be longer than context_length")

        self.data = torch.from_numpy(data).unfold(0, context_length + 1, 1)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.data[index]
        return tokens[:-1], tokens[1:]


def make_dataloader(
    dataset_config: DatasetConfig,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> torch.utils.data.DataLoader:

    dataset = NextTokenDataset(
        # first 'train' is the train config, second is the train set.
        data=np.load(dataset_config.path, mmap_mode="c"),
        context_length=dataset_config.seq_len,
    )
    sampler = (
        torch.utils.data.RandomSampler(
            dataset, replacement=True, generator=generator
        )
        if dataset_config.random_sample
        else None
    )

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, sampler=sampler
    )


def model_from_config(config: ModelConfig) -> torch.nn.Module:
    match config:
        case TransformerLMConfig():
            return TransformerLM(**config.model_dump())


def optimizer_from_config(
    model: torch.nn.Module, config: OptimizerConfig
) -> torch.optim.Optimizer:
    match config:
        case AdamWConfig():
            return AdamW(model.parameters(), **config.model_dump())


def lr_sweep(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    low: float = 1e-5,
    high: float = 1e-2,
    steps: int = 128,
    device: torch.types.Device = None,
) -> tuple[list, list]:
    for group in optimizer.param_groups:
        group["lr"] = low

    model = model.to(device)
    model.train()
    data = iter(dataloader)

    lrs = []
    losses = []
    for step in range(steps):
        lr = low * (high / low) ** (step / (steps - 1))

        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = next(data)
        x: torch.Tensor
        y: torch.Tensor

        x = x.to(device, dtype=torch.long)
        y = y.to(device, dtype=torch.long)

        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()

        clip_gradient(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()

        lrs.append(lr)
        losses.append(loss.item())

    return lrs, losses


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]
