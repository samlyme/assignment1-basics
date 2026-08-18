from collections.abc import Iterable
import math

import torch
from einops import reduce
from jaxtyping import Float, Int


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max = torch.max(x, dim=dim, keepdim=True).values
    x = x - max

    x = torch.exp(x)
    sum = torch.sum(x, dim=dim, keepdim=True)

    x = x / sum
    return x


def cross_entropy(
    logits: Float[torch.Tensor, "... vocab"],
    target: Int[torch.Tensor, "... 1"],
) -> Float[torch.Tensor, "1"]:
    # Subtract max for numerical stability
    maxes = reduce(logits, "... vocab -> ... 1", "max")
    logits = logits - maxes

    sum_exp = reduce(logits.exp(), "... vocab -> ... 1", "sum")

    target_logits = logits.gather(dim=-1, index=target.unsqueeze(-1))

    return -reduce(target_logits - sum_exp.log(), "... -> 1", "mean")


def clip_gradient(
    params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6
) -> None:
    # TODO: could be optimized i think.
    grads = [p.grad for p in params if p.grad is not None]
    l2_norm = 0.0
    for grad in grads:
        l2_norm += (grad**2).sum().item()
    l2_norm = math.sqrt(l2_norm)

    if l2_norm >= max_l2_norm:
        factor = max_l2_norm / (l2_norm + eps)
        for grad in grads:
            grad *= factor
