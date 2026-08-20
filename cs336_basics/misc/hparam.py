# %%
import numpy as np
import torch
from tqdm import tqdm

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import clip_gradient, cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import get_batch
from cs336_basics.train import MODEL_CONFIGS

import matplotlib.pyplot as plt

# %%

device = torch.device("cuda")

dataset = np.load(
    "../../data/ts-valid-ts-10000.npy",
    mmap_mode="c",
)

# %%

min_lr = 1e-5
max_lr = 1e-2
steps = 100
batch_sizes = [32, 64, 128]

res = {}

for batch_size in batch_sizes:
    print("using batch_size", batch_size)
    model = TransformerLM(**MODEL_CONFIGS["toy"]).to(device)

    lr = min_lr
    optimizer = AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0
    )

    lrs = []
    losses = []

    for it in tqdm(range(steps)):
        lr = min_lr * (max_lr / min_lr) ** (it / (steps - 1))

        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(dataset, batch_size, 256, "cuda:0")

        x = x.long()
        y = y.long()

        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()

        clip_gradient(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()

        lrs.append(lr)
        losses.append(loss.item())

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
