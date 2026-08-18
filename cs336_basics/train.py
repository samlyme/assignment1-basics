from collections.abc import Iterable
import math
from typing import Any, TypedDict
from collections.abc import Callable

from einops import reduce
from torch import Tensor
from jaxtyping import Float, Int
import torch
import numpy.typing as npt


def cross_entropy(
    logits: Float[Tensor, "... vocab"],
    target: Int[Tensor, "... 1"],
) -> Float[Tensor, "1"]:
    # Subtract max for numerical stability
    maxes = reduce(logits, "... vocab -> ... 1", "max")
    logits = logits - maxes

    sum_exp = reduce(logits.exp(), "... vocab -> ... 1", "sum")

    target_logits = logits.gather(dim=-1, index=target.unsqueeze(-1))

    return -reduce(target_logits - sum_exp.log(), "... -> 1", "mean")


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
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        with torch.enable_grad():
            loss = closure() if closure else None

        for group in self.param_groups:
            alpha: float = group["lr"]
            betas = group["betas"]
            beta_1, beta_2 = betas
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                p: torch.Tensor

                if p.grad is None:
                    continue
                g = p.grad

                if weight_decay != 0.0:
                    g += weight_decay * p

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


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor]
        | Iterable[dict[str, Any]]
        | Iterable[tuple[str, Tensor]],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        with torch.enable_grad():
            loss = closure() if closure else None

        for group in self.param_groups:
            alpha: float = group["lr"]
            betas = group["betas"]
            beta_1, beta_2 = betas
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

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

                alpha_t = (
                    alpha
                    * math.sqrt(1 - beta_2 ** state["t"])
                    / (1 - beta_1 ** state["t"])
                )

                # apply update
                p -= alpha * weight_decay * p
                p -= alpha_t * state["m"] / (state["v"].sqrt() + eps)

        return loss


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    if it > cosine_cycle_iters:
        return min_learning_rate

    return min_learning_rate + 0.5 * (
        1
        + math.cos(
            (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
        )
    ) * (max_learning_rate - min_learning_rate)


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


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dataset.ndim == 1

    # .from_numpy shares memory with numpy, thus it does not mess with mmap.
    data = torch.from_numpy(dataset).unfold(0, context_length + 1, 1)
    start_indices = torch.randint(0, data.shape[0], (batch_size,))

    batches = data[start_indices]

    return batches[:, :-1].to(device), batches[:, 1:].to(device)
