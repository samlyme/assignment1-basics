# %%

import sympy as sp


sp.init_printing()
batch_size, vocab_size, context_length, num_layers, num_heads, d_m, d_ff = (
    sp.symbols(
        "B V C L H d_m d_ff",
        positive=True,
        integer=True,
    )
)
d_k = d_m / num_heads
gpt_2_xl = {
    batch_size: 1024,
    vocab_size: 50257,
    context_length: 1024,
    num_layers: 48,
    d_m: 1600,
    num_heads: 25,
    d_ff: 4288,
}
toy = {
    vocab_size: 10000,
    context_length: 256,
    num_layers: 4,
    d_m: 512,
    num_heads: 16,
    d_ff: 1344,
}
small = {
    vocab_size: 10000,
    context_length: 256,
    num_layers: 4,
    d_m: 768,
    num_heads: 16,
    d_ff: 2048,
}

# %%
# memory usage by params and activations. Solving for max batch size.G


def param_count():
    input_emb = vocab_size * d_m
    rms = d_m

    mha = 4 * d_m**2
    swiglu = 3 * d_m * d_ff

    output_proj = d_m * vocab_size

    return (
        input_emb + num_layers * (mha + rms + swiglu + rms) + rms + output_proj
    )


def activation_count():
    rms = batch_size * context_length * d_m
    mha = (
        5 * batch_size * context_length * d_m
        + 2 * batch_size * num_heads * context_length * context_length
    )
    swiglu = (
        4 * batch_size * context_length * d_ff
        + batch_size * context_length * d_m
    )

    output_proj = batch_size * context_length * vocab_size

    ce = (
        2 * batch_size * context_length * vocab_size
        + 3 * batch_size * vocab_size
    )

    return num_layers * (mha + rms + swiglu + rms) + output_proj + ce


hbm_3090_gb = 23.5

P = param_count()

A = activation_count()

G = P

O = 2 * P  # noqa: E741

peak_mem = (P + A + G + O) * 4 * 1e-9  # use fp32 = 4 bytes, 1e-9 is gb.
peak_mem.subs(toy), peak_mem.subs(small)
# %%
sp.solve(sp.Eq(peak_mem, hbm_3090_gb), batch_size)[0].subs(small)

# %%

# for each param, we do an update for 2 moments u, v:
update_m = 3 * P
update_v = 4 * P

# we then apply updates for the params
weight_decay = 2 * P
apply_moment = 5 * P

adamw_step = update_m + update_v + weight_decay + apply_moment


def matmul(M, K, N):
    return 2 * M * K * N


def mha():
    q = matmul(context_length, d_m, d_m)
    k = matmul(context_length, d_m, d_m)
    v = matmul(context_length, d_m, d_m)
    # ignore RoPE, it is just matvec

    attn = num_heads * matmul(context_length, d_k, context_length)

    weighted_sum = matmul(context_length, context_length, d_m)

    o = matmul(context_length, d_m, d_m)
    return q + k + v + attn + weighted_sum + o


def swiglu():
    return 3 * matmul(
        context_length, d_m, d_ff
    )  # not the same shapes, but same ops


forward_pass = batch_size * (
    num_layers * (mha() + swiglu()) + matmul(context_length, d_m, vocab_size)
)
backward_pass = 2 * forward_pass

total_flops = (forward_pass + backward_pass + adamw_step) * 400_000
print(
    f"{total_flops.subs(gpt_2_xl).subs({batch_size: 1024}) * 1e-12} teraFLOPs"
)

peak_flop_per_s = 495 * 1e12
mfu = 0.5

effective_flops_per_s = peak_flop_per_s * mfu

seconds = total_flops / effective_flops_per_s

print((seconds / 60 / 60).subs(gpt_2_xl).round(), "hours")
