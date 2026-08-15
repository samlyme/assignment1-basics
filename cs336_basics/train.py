from collections.abc import Iterable
import math
from typing import Any, TypedDict
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


class Adam(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor]
        | Iterable[dict[str, Any]]
        | Iterable[tuple[str, Tensor]],
        lr: float,
        beta_1: float,
        beta_2: float,
        eps: float,
    ) -> None:
        assert 0 <= beta_1 < 1 and 0 <= beta_2 < 1

        defaults = {
            "lr": lr,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        with torch.enable_grad():
            loss = closure() if closure else None

        for group in self.param_groups:
            alpha: float = group["lr"]
            beta_1: float = group["beta_1"]
            beta_2: float = group["beta_2"]
            eps: float = group["eps"]

            for p in group["params"]:
                p: torch.Tensor

                if p.grad is None:
                    continue
                g = p.grad

                if not self.state[p]:
                    # initialize the moments.
                    state: AdamParamState = {
                        "m": torch.zeros_like(p),
                        "v": torch.zeros_like(p),
                        "t": 0,
                    }
                    self.state[p] = state

                state: AdamParamState = self.state[p]

                # update state
                state["t"] += 1

                state["m"] *= beta_1
                state["m"] += (1 - beta_1) * g

                state["v"] *= beta_2
                state["v"] += (1 - beta_2) * (g * g)

                # compute bias corrected
                m_hat = state["m"] / (1 - beta_1 ** state["t"])
                v_hat = state["v"] / (1 - beta_2 ** state["t"])

                # apply update
                p -= alpha * (m_hat / (v_hat.sqrt() + eps))

        return loss


class AdamParamState(TypedDict):
    m: torch.Tensor
    v: torch.Tensor
    t: int
