# %%
import random
import time


from cs336_basics.bpe_tokenizer import Tokenizer

# %%

tiny_tokenizer = Tokenizer.from_files(
    "../out/bpe_params_ts-train/vocab.pkl", "../out/bpe_params_ts-train/merges.pkl", ["<|endoftext|>"]
)

owt_tokenizer = Tokenizer.from_files(
    "../out/bpe_params_owt-train-long/vocab.pkl", "../out/bpe_params_owt-train-long/merges.pkl", ["<|endoftext|>"]
)

# %%

tiny_docs: list[str] = []
with open("../data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
    # get first 50, I can sample later if needed.
    for i in range(50):
        doc = []
        while True:
            line = f.readline()
            if line.find(b"<|endoftext|>") != -1:
                break
            doc.append(line)

        tiny_docs.append(b"".join(doc).decode())


owt_docs: list[str] = []
with open("../data/owt_valid.txt", "rb") as f:
    # get first 50, I can sample later if needed.
    for i in range(50):
        doc = []
        while True:
            line = f.readline()
            if line.find(b"<|endoftext|>") != -1:
                break
            doc.append(line)

        owt_docs.append(b"".join(doc).decode())

# %%

samples = {"tiny": random.choices(tiny_docs, k=10), "owt": random.choices(owt_docs, k=10)}

tiny_tokenizer_res = {
    "tiny": list(map(tiny_tokenizer.encode, samples["tiny"])),
    "owt": list(map(tiny_tokenizer.encode, samples["owt"])),
}

owt_tokenizer_res = {
    "tiny": list(map(owt_tokenizer.encode, samples["tiny"])),
    "owt": list(map(owt_tokenizer.encode, samples["owt"])),
}

tiny_tokenizer_ratio = {
    "tiny": sum(len(sample) for sample in samples["tiny"]) / sum(len(tokens) for tokens in tiny_tokenizer_res["tiny"]),
    "owt": sum(len(sample) for sample in samples["owt"]) / sum(len(tokens) for tokens in tiny_tokenizer_res["owt"]),
}

owt_tokenizer_ratio = {
    "tiny": sum(len(sample) for sample in samples["tiny"]) / sum(len(tokens) for tokens in owt_tokenizer_res["tiny"]),
    "owt": sum(len(sample) for sample in samples["owt"]) / sum(len(tokens) for tokens in owt_tokenizer_res["owt"]),
}

print("tiny_tokenizer", tiny_tokenizer_ratio)
print("owt_tokenizer", owt_tokenizer_ratio)

# %%
num_bytes = int(1e6)
with open("../data/owt_valid.txt", "rb") as f:
    throughput_sample = f.read(num_bytes).decode(errors="ignore")

start = time.perf_counter()
tiny_tokenizer.encode(throughput_sample)
end = time.perf_counter()
elapsed = end - start

throughput = num_bytes / elapsed
print(f"{elapsed=:.4f}, {throughput=:.0f}")
