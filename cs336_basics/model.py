from math import ceil
import math

from einops import einsum, rearrange, reduce
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


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        if d_ff is None:
            x = 8 / 3 * d_model
            d_ff = ceil(x / 64) * 64
        self.d_ff = d_ff

        self.W_1 = Linear(d_model, d_ff, device, dtype)
        self.W_2 = Linear(d_ff, d_model, device, dtype)
        self.W_3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_2(silu(self.W_1(x)) * self.W_3(x))


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.big_theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0
        K = d_k // 2
        buf = torch.empty((max_seq_len, K, 2, 2))

        def get_theta(i: int, k: int):
            p = k / K
            denom = self.big_theta**p
            return i / denom

        for position in range(max_seq_len):
            for ki in range(K):
                theta_ik = get_theta(position, ki)
                buf[position, ki] = torch.tensor(
                    [
                        [math.cos(theta_ik), -math.sin(theta_ik)],
                        [math.sin(theta_ik), math.cos(theta_ik)],
                    ]
                )

        self.register_buffer("rotations", buf.to(device), persistent=False)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor
    ) -> torch.Tensor:
        assert torch.all(0 <= token_positions)
        assert torch.all(token_positions < self.max_seq_len)

        token_positions = token_positions.broadcast_to(x.shape[:-1])
        x = rearrange(x, "... seq (k pair) -> ... seq k pair", pair=2)

        Rs = self._get_rotation_tensor(token_positions)

        z = einsum(Rs, x, "... seq k row col, ... seq k col -> ... seq k row")

        return rearrange(z, "... seq k pair -> ... seq (k pair)")

    def _get_rotation_tensor(self, token_positions) -> torch.Tensor:
        # TODO: represent rotation as tensor: ... k (2, 2)
        # where ... is from token_positions.

        rotations: torch.Tensor = self.rotations  # type: ignore
        return rotations[token_positions]
