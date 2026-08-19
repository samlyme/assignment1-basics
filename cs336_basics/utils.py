from dataclasses import dataclass
import os
import typing

import torch
import numpy.typing as npt


@dataclass(frozen=True)
class RotaryPositionalEmbeddingConfig(typing.TypedDict):
    theta: float
    d_k: int
    max_seq_len: int


@dataclass(frozen=True)
class TransformerLMConfig(typing.TypedDict):
    vocab_size: int
    d_model: int
    num_heads: int
    d_ff: int
    context_length: int
    num_layers: int
    rope_config: RotaryPositionalEmbeddingConfig


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dataset.ndim == 1

    # .from_numpy shares memory with numpy, thus it does not mess with mmap.
    data = torch.from_numpy(dataset).unfold(0, context_length + 1, 1)
    start_indices = torch.randint(0, data.shape[0], (batch_size,))

    batches = data[start_indices]

    return batches[:, :-1].to(device), batches[:, 1:].to(device)


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
