# %%
import torch
import numpy as np
from einops import einsum, rearrange

BATCH = 32
SEQ = 16
VOCAB = 8

logits = torch.randn((BATCH, SEQ, VOCAB))

expected = torch.randint(0, VOCAB, (BATCH, SEQ, 1))

print(logits.shape, expected.shape)
# %%

logits.gather(-1, expected).shape


# %%
dataset = np.random.randint(0, VOCAB, (64,))

num_tokens = dataset.shape[0]
batch_size = BATCH
context_length = SEQ

data = torch.from_numpy(dataset).unfold(0, context_length + 1, 1)
data.shape
