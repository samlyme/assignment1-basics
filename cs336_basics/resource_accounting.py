# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def get_flops(
    vocab_size=50257,
    context_length=1024,
    num_layers=48,
    d_model=1600,
    num_heads=25,
):
    d_ff = int(8 / 3 * d_model)

    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    # Q, K, V, and output projections
    kqvo_proj = 4 * 2 * context_length * d_model * d_model

    # QK^T and attention @ V
    dot_atten = (
        num_heads
        * 2  # QK^T and attention @ V
        * 2  # 2 FLOPs per multiply-add
        * context_length
        * d_k
        * context_length
    )

    mha_flops = num_layers * (kqvo_proj + dot_atten)

    # SwiGLU: gate, up, and down projections
    ffn_flops = num_layers * 3 * 2 * context_length * d_model * d_ff

    output_proj_flops = 2 * context_length * d_model * vocab_size

    total_flops = mha_flops + ffn_flops + output_proj_flops

    return total_flops, mha_flops, ffn_flops, output_proj_flops


# %%

models = {
    "Small": (12, 768, 12),
    "Medium": (24, 1024, 16),
    "Large": (36, 1280, 20),
    "XL": (48, 1600, 25),
}

attention_pct = []
ffn_pct = []
output_pct = []

for name, (layers, d_model, heads) in models.items():
    total, mha, ffn, output = get_flops(
        num_layers=layers,
        d_model=d_model,
        num_heads=heads,
    )

    attention_pct.append(100 * mha / total)
    ffn_pct.append(100 * ffn / total)
    output_pct.append(100 * output / total)


x = np.arange(len(models))

plt.figure(figsize=(8, 5))

plt.bar(x, attention_pct, label="Attention")
plt.bar(x, ffn_pct, bottom=attention_pct, label="SwiGLU FFN")

bottom = np.array(attention_pct) + np.array(ffn_pct)
plt.bar(x, output_pct, bottom=bottom, label="Output projection")

plt.xticks(x, models.keys())
plt.ylabel("Fraction of forward-pass matmul FLOPs (%)")
plt.xlabel("GPT-2 model")
plt.title("Distribution of Matmul FLOPs as GPT-2 Scales")
plt.ylim(0, 100)
plt.legend()

plt.tight_layout()
plt.show()

# %%
models = {
    "Small": (12, 768, 12),
    "Medium": (24, 1024, 16),
    "Large": (36, 1280, 20),
    "XL": (48, 1600, 25),
}

attention_pct = []
ffn_pct = []
output_pct = []

for name, (layers, d_model, heads) in models.items():
    total, mha, ffn, output = get_flops(
        num_layers=layers,
        d_model=d_model,
        num_heads=heads,
        context_length=16384,
    )

    attention_pct.append(100 * mha / total)
    ffn_pct.append(100 * ffn / total)
    output_pct.append(100 * output / total)


x = np.arange(len(models))

plt.figure(figsize=(8, 5))

plt.bar(x, attention_pct, label="Attention")
plt.bar(x, ffn_pct, bottom=attention_pct, label="SwiGLU FFN")

bottom = np.array(attention_pct) + np.array(ffn_pct)
plt.bar(x, output_pct, bottom=bottom, label="Output projection")

plt.xticks(x, models.keys())
plt.ylabel("Fraction of forward-pass matmul FLOPs (%)")
plt.xlabel("GPT-2 model")
plt.title("Distribution of Matmul FLOPs as GPT-2 Scales with long context")
plt.ylim(0, 100)
plt.legend()

plt.tight_layout()
plt.show()
