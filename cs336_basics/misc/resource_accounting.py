# %%

import sympy as sp


sp.init_printing()
B, V, C, L, H, d_m, d_ff = sp.symbols(
    "B V C L H d_m d_ff",
    positive=True,
    integer=True,
)
d_k = d_m / H
gpt_2_xl = {B: 1024, V: 50257, C: 1024, L: 48, d_m: 1600, H: 25, d_ff: 4288}
toy = {V: 10000, C: 256, L: 4, d_m: 512, H: 16, d_ff: 1344}

# %%
# memory usage by params and activations. Solving for max batch size.G


def param_count():
    input_emb = V * d_m
    rms = d_m

    mha = 4 * d_m**2
    swiglu = 3 * d_m * d_ff

    output_proj = d_m * V

    return input_emb + L * (mha + rms + swiglu + rms) + rms + output_proj


def activation_count():
    rms = B * C * d_m
    mha = 5 * B * C * d_m + 2 * B * H * C * C
    swiglu = 4 * B * C * d_ff + B * C * d_m

    output_proj = B * C * V

    ce = 2 * B * C * V + 3 * B * V

    return L * (mha + rms + swiglu + rms) + output_proj + ce


max_mem = 80

P = param_count()

A = activation_count()

G = P

O = 2 * P  # noqa: E741

peak_mem = (P + A + G + O) * 4 * 1e-9  # use fp32 = 4 bytes, 1e-9 is gb.
peak_mem.subs(toy)
# %%
sp.solve(sp.Eq(peak_mem, max_mem), B)[0].subs(toy)

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
    q = matmul(C, d_m, d_m)
    k = matmul(C, d_m, d_m)
    v = matmul(C, d_m, d_m)
    # ignore RoPE, it is just matvec

    attn = H * matmul(C, d_k, C)

    weighted_sum = matmul(C, C, d_m)

    o = matmul(C, d_m, d_m)
    return q + k + v + attn + weighted_sum + o


def swiglu():
    return 3 * matmul(C, d_m, d_ff)  # not the same shapes, but same ops


forward_pass = B * (L * (mha() + swiglu()) + matmul(C, d_m, V))
backward_pass = 2 * forward_pass

total_flops = (forward_pass + backward_pass + adamw_step) * 400_000
print(f"{total_flops.subs(gpt_2_xl).subs({B: 1024}) * 1e-12} teraFLOPs")

peak_flop_per_s = 495 * 1e12
mfu = 0.5

effective_flops_per_s = peak_flop_per_s * mfu

seconds = total_flops / effective_flops_per_s

print((seconds / 60 / 60).subs(gpt_2_xl).round(), "hours")
