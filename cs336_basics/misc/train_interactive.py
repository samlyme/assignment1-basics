# %%
import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import get_batch

# %%
dataset = np.load(
    "/Users/sam/Documents/code/stanford-cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid-ts.npy",
    mmap_mode="r",
)

# TODO: load model config from arg
model = TransformerLM(10000, 64, 4, 192, 128, 2, 10000.0).to("mps")

# TODO: load optimizer
optimizer = AdamW(model.parameters())

# %%

for i in range(10):
    x, y = get_batch(dataset, 32, 128, "mps")
    x = x.long()
    y = y.long()

    pred = model(x)

    loss = cross_entropy(pred, y)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

print(f"{loss.item()=:.3f}")
