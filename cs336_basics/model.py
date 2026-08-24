from math import ceil, sqrt
import math

from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.nn_utils import softmax

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
        self.weight = torch.nn.Parameter(
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
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


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
        self.weight = torch.nn.Parameter(
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
        return self.weight[token_ids]


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
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, dtype=dtype, device=device)
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_square = reduce(x * x, "... d -> ... 1", "mean")
        rms = torch.sqrt(mean_square + self.eps)

        result = (x / rms) * self.weight

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

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


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

        # TODO: vectorize this.
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

        x = rearrange(x, "... seq (k pair) -> ... seq k pair", pair=2)

        Rs = self._get_rotation_tensor(token_positions)

        # NOTE: a standard matmul in einops is "m n, n k -> m k".
        # Thus, a matvec is "row col, col 1 -> row 1"
        z = einsum(Rs, x, "... seq k row col, ... seq k col -> ... seq k row")

        return rearrange(z, "... seq k pair -> ... seq (k pair)")

    def _get_rotation_tensor(self, token_positions) -> torch.Tensor:
        # Returns a tensor of shape ... k 2 2.
        rotations: torch.Tensor = self.rotations  # type: ignore
        return rotations[token_positions]


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... n d_k"],
    K: Float[Tensor, " ... m d_k"],
    V: Float[Tensor, " ... m d_v"],
    mask: Bool[Tensor, " ... n m"] | None = None,
) -> Float[Tensor, " ... n d_v"]:
    # NOTE: trust the jaxtyping. No need for extensive assert's
    d_k = Q.shape[-1]

    a = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / sqrt(d_k)
    a.shape

    if mask is not None:
        a = a.masked_fill(~mask, float("-inf"))

    a = softmax(a, -1)

    return einsum(a, V, "... sq sk, ... sk d_v -> ... sq d_v")


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_embedding: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0

        # following paper
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # same as d_model * d_model
        # wrote them using "semantics" in mind lol.
        self.q_proj = Linear(d_model, self.d_k * num_heads)
        self.k_proj = Linear(d_model, self.d_k * num_heads)
        self.v_proj = Linear(d_model, self.d_v * num_heads)
        self.output_proj = Linear(self.d_v * num_heads, d_model)

        self.positional_embedding = positional_embedding

    def forward(
        self,
        x: Float[Tensor, " ... n d_model"],
        token_positions: Int[Tensor, " ... n"] | None = None,
    ) -> Float[Tensor, "... n d_v"]:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q_i = rearrange(
            Q, "... n (head d_k) -> ... head n d_k", head=self.num_heads
        )
        K_i = rearrange(
            K, "... n (head d_k) -> ... head n d_k", head=self.num_heads
        )
        if (
            token_positions is not None
            and self.positional_embedding is not None
        ):
            Q_i = self.positional_embedding(Q_i, token_positions)
            K_i = self.positional_embedding(K_i, token_positions)

        V_i = rearrange(
            V, "... n (head d_v) -> ... head n d_v", head=self.num_heads
        )

        n = x.shape[-2]
        causal_mask = torch.tril(
            torch.ones(n, n, dtype=torch.bool, device=x.device),
            diagonal=0,
        )

        out = scaled_dot_product_attention(Q_i, K_i, V_i, causal_mask)
        out = rearrange(out, "... head n d_v -> ... n (head d_v)")
        return self.output_proj(out)


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_embedding: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(
            d_model, num_heads, positional_embedding
        )

        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self, x: Float[Tensor, " ... n d_model"]
    ) -> Float[Tensor, "... n d_model"]:
        n = x.shape[-2]
        token_positions = torch.arange(n, dtype=torch.long, device=x.device)

        z_1 = x + self.attn(self.ln1(x), token_positions)

        return z_1 + self.ffn(self.ln2(z_1))


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        num_layers: int,
        rope_theta: float,
    ) -> None:
        if d_model % num_heads != 0:
            raise ValueError(
                f"Invalid hyperparameter: {d_model=} must be divisible by {num_heads=}"
            )

        super().__init__()

        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model)

        # NOTE: assume that positional embedding has no params, and can share
        # instance across layers.
        d_k = d_model // num_heads
        rope = RotaryPositionalEmbedding(rope_theta, d_k, context_length)
        self.layers = torch.nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, rope)
            for i in range(num_layers)
        )

        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self, x: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        if x.shape[-1] > self.context_length:
            raise ValueError(
                "Input sequence does not fit within model's context window."
            )

        emb = self.token_embeddings(x)

        z = emb
        for layer in self.layers:
            z = layer(z)

        z = self.ln_final(z)
        z = self.lm_head(z)
        return z
