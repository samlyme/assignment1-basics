# %%
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import subprocess

from cs336_basics.optimizer import AdamW
from cs336_basics.utils import (
    MODEL_CONFIGS,
    DatasetConfig,
    NextTokenDataset,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
    lr_sweep,
    model_from_config,
)
from matplotlib import pyplot as plt
import numpy as np
import sympy as sp
import torch
from tqdm import tqdm


# %%
# Approximate max batch size.
train_path = Path("/data/tokenized/ts-train-10k_ts.npy")
valid_path = Path("/data/tokenized/ts-valid-10k_ts.npy")
model_config = MODEL_CONFIGS["toy"]
model_config_dict = asdict(model_config)
keys = list(model_config_dict.keys())
model_symbols = sp.symbols(keys)


def subs_model(exp):
    return exp.subs(
        {s: model_config_dict[k] for s, k in zip(model_symbols, keys)}
    )


(
    vocab_size,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    context_length,
    rope_theta,
) = model_symbols

batch_size, steps = sp.symbols("B S")
symbols_to_keys = {s: k for s, k in zip(model_symbols, keys)}


def param_count():
    input_emb = vocab_size * d_model
    rms = d_model

    mha = 4 * d_model**2
    swiglu = 3 * d_model * d_ff

    output_proj = d_model * vocab_size

    return (
        input_emb + num_layers * (mha + rms + swiglu + rms) + rms + output_proj
    )


def activation_count():
    rms = batch_size * context_length * d_model
    mha = (
        5 * batch_size * context_length * d_model
        + 2 * batch_size * num_heads * context_length * context_length
    )
    swiglu = (
        4 * batch_size * context_length * d_ff
        + batch_size * context_length * d_model
    )

    output_proj = batch_size * context_length * vocab_size

    ce = (
        2 * batch_size * context_length * vocab_size
        + 3 * batch_size * vocab_size
    )

    return num_layers * (mha + rms + swiglu + rms) + output_proj + ce


P = param_count()

A = activation_count()

G = P

O = 2 * P  # noqa: E741

VRAM = 24  # 24 GB for 3090
peak_mem = (P + A + G + O) * 4 * 1e-9  # use fp32 = 4 bytes, 1e-9 is gb.
# find upper bound for batch size for a give VRAM

max_batch_size = int(subs_model(sp.solve(sp.Eq(peak_mem, VRAM), batch_size)[0]))
max_batch_size


# %%
# Approximate model memory usage for each batch size
def batch_sizes_range(lo: int, hi: int) -> list[int]:
    out: list[int] = []

    i = 1
    while i < hi:
        out.append(i)
        i *= 2

    return out


batch_sizes = batch_sizes_range(1, max_batch_size)
batch_sizes.reverse()  # for my setup it is better to run large batch sizes first.
{b: subs_model(peak_mem.subs({batch_size: b})) for b in batch_sizes}

# %%
# Batch size experiment with fixed total tokens processed
total_tokens_processed = 327_680_000
train_configs: list[TrainConfig] = []
for batch_size in batch_sizes:
    steps = int(
        subs_model(total_tokens_processed / batch_size / context_length)
    )
    train_configs.append(
        TrainConfig(
            train=DatasetConfig(train_path, model_config.context_length),
            val=DatasetConfig(valid_path, model_config.context_length),
            batch_size=batch_size,
            steps=steps,
        )
    )
train_configs

# %%
# LR sweep by batch size. Previously showed no real dependence.
res = {}
data = np.load(
    "/data/tokenized/ts-valid-10k_ts.npy",
    mmap_mode="c",
)
min_lr = 1e-5
max_lr = 1e-2
for batch_size in tqdm(batch_sizes):
    model = model_from_config(model_config)
    lr = min_lr
    optimizer = AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0
    )

    dataloader = torch.utils.data.DataLoader(
        NextTokenDataset(data, model_config.context_length),
        batch_size,
        shuffle=True,
    )
    lrs, losses = lr_sweep(model, optimizer, dataloader, device="cuda")

    res[batch_size] = {"lrs": lrs, "losses": losses}

for batch_size, result in res.items():
    plt.plot(
        result["lrs"],
        result["losses"],
        label=f"BS={batch_size}",
    )

plt.xscale("log")
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.title("Learning Rate Sweep by Batch Size")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# seems like 1.5e-4 is very stable, with 1e-3 being relatively aggro for all batch sizes.

# %%
# TODO: sweep other hparams.
# TODO: sweep scheduling.
optim_config = OptimizerConfig(lr=1.5e-4)

# %%
# Define function to submit jobs.


def add_pueue(config: RunConfig, desc: str | None = None) -> None:
    def handle_path(value):
        if isinstance(value, Path):
            return str(value)
        raise TypeError(
            f"Object of type {type(value)} is not JSON serializable"
        )

    payload = json.dumps(asdict(config), default=handle_path)

    command = shlex.join(
        [
            ".venv/bin/python",
            "cs336_basics/train.py",
            "--config",
            payload,
        ]
    )

    res = subprocess.run(
        [
            "pueue",
            "add",
            "--stashed",
            *(["--label", desc] if desc is not None else []),
            "--",
            command,
        ],
        check=True,
        cwd="/home/sam/code/stanford-cs336/assignment1-basics",
        capture_output=True,
    )
    print(res)


# %%
add_pueue(
    RunConfig(
        model=model_config,
        optim=optim_config,
        train=TrainConfig(
            train=DatasetConfig(
                train_path, model_config.context_length, shuffle=False
            ),
            val=DatasetConfig(
                valid_path, model_config.context_length, shuffle=False
            ),
            batch_size=32,
            steps=1000,
        ),
    ),
    "Test new dataloader util.",
)

# %%
# Submit batch size sweep.
for train_config in train_configs:
    config = RunConfig(
        model=model_config,
        optim=optim_config,
        train=train_config,
    )
    add_pueue(config)
