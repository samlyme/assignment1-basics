# %%
import subprocess


def submit_pueue(train_path, val_path):
    python = ".venv/bin/python"
    src = "cs336_basics/train.py"
    subprocess.run(
        f"pueue add '{python} {src} --train {train_path} --val {val_path}'",
        shell=True,
    )
