from collections.abc import Iterable
import math
from typing import Any
from collections.abc import Callable

from einops import reduce
from torch import Tensor
from jaxtyping import Float, Int
import torch


def cross_entropy(
    logits: Float[Tensor, "... vocab"],
    target: Int[Tensor, "... 1"],
) -> Float[Tensor, "... vocab"]:
    # Subtract max for numerical stability
    maxes = reduce(logits, "... vocab -> ... 1", "max")
    logits = logits - maxes

    sum_exp = reduce(logits.exp(), "... vocab -> ... 1", "sum")

    log_p = logits - sum_exp.log()

    return -log_p.gather(dim=-1, index=target.unsqueeze(-1)).mean()


class SGD(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor]
        | Iterable[dict[str, Any]]
        | Iterable[tuple[str, Tensor]],
        lr=1e-3,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        with torch.enable_grad():
            loss = closure() if closure else None

        for group in self.param_groups:
            lr: float = group["lr"]  # just a pytorch idiom
            for p in group["params"]:  # another pytorch idiom
                p: torch.Tensor
                if p.grad is None:
                    continue

                state = self.state[p]  # each parameter has its own state
                t = state.get("t", 0)

                alpha = lr / math.sqrt(t + 1)
                p.add_(-p.grad, alpha=alpha)
                state["t"] = t + 1

        return loss
