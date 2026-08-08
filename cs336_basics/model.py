from einops import einsum, reduce
import torch


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        std = (2 / (in_features + out_features)) ** 0.5
        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(
                    (out_features, in_features), dtype=dtype, device=device
                ),
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        std = 1
        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(
                    (num_embeddings, embedding_dim), dtype=dtype, device=device
                ),
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.g = torch.nn.Parameter(
            torch.ones(d_model, dtype=dtype, device=device)
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_square = reduce(x * x, "... d -> ... 1", "mean")
        rms = torch.sqrt(mean_square + self.eps)

        result = (x / rms) * self.g

        return result.to(in_dtype)
