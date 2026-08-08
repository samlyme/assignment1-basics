from einops import einsum
import torch


class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        std = (2 / (in_features + out_features)) ** 0.5
        self.W = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty((out_features, in_features), dtype=dtype, device=device), std=std, a=-3 * std, b=3 * std
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
                torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device), std=std, a=-3 * std, b=3 * std
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]
