# %%
from cs336_basics.model import TransformerLM
import numpy as np
import torch

from cs336_basics.optimizer import AdamW
from cs336_basics.utils import (
    model_from_config,
    lr_sweep,
    NextTokenDataset,
    MODEL_CONFIGS,
)

import matplotlib.pyplot as plt

# %%

data = np.load(
    "/data/tokenized/ts-valid-10k_ts.npy",
    mmap_mode="c",
)

# %%

min_lr = 1e-5
max_lr = 1e-2
steps = 100
batch_sizes = [32]

res = {}

for batch_size in batch_sizes:
    model: TransformerLM = model_from_config(MODEL_CONFIGS["toy"])  # type: ignore
    lr = min_lr
    optimizer = AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0
    )

    dataloader = torch.utils.data.DataLoader(
        NextTokenDataset(data, 256), batch_size
    )
    lrs, losses = lr_sweep(model, optimizer, dataloader, device="cuda")

    res[batch_size] = {"lrs": lrs, "losses": losses}

# %%
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


# %%
candidate_lrs_toy = [
    1.5e-4,
    3e-4,
    1e-3,
    2e-3,
]
