# %%
import matplotlib.pyplot as plt

# %%

logs = []
for run in ("toy-slower", "toy-slow", "toy-fast", "toy-faster"):
    with open(f"../../out/{run}/train.log") as f:
        logs.append(f.read())

log_bodies = [log.split("~" * 32)[-1] for log in logs]
# %%
for log_body in log_bodies:
    iterations = []
    train_losses = []
    val_losses = []

    for line in log_body.strip().splitlines():
        parts = line.split(", ")

        iteration = int(parts[1].split("=")[1])
        train_loss = float(parts[2].split("=")[1])
        val_loss = float(parts[3].split("=")[1])

        iterations.append(iteration)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, train_losses, label="Train loss")
    plt.plot(iterations, val_losses, label="Validation loss")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
