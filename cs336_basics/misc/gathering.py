# %%
import torch

BATCH = 32
SEQ = 16
VOCAB = 8

logits = torch.randn((BATCH, SEQ, VOCAB))

expected = torch.randint(0, VOCAB, (BATCH, SEQ, 1))

print(logits.shape, expected.shape)
# %%

logits.gather(-1, expected).shape
