from einops import reduce, einsum
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(
    logits: Float[Tensor, "... vocab"],
    target: Int[Tensor, "... 1"],
) -> Float[Tensor, "... vocab"]:
    # Subtract max for numerical stability
    maxes = reduce(logits, "... vocab -> ... 1", "max")
    logits = logits - maxes
    logits.shape
    target.shape

    sum_exp = reduce(logits.exp(), "... vocab -> ... 1", "sum")
    sum_exp.shape

    log_p = logits - sum_exp.log()
    log_p.shape

    return -log_p.gather(dim=-1, index=target.unsqueeze(-1)).mean()
