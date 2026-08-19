# %%
import matplotlib.pyplot as plt

# %%

logs = """USING DEVICE: cuda
Save models to: out/train-compiled
LOG_FREQ=100, SAVE_FREQ=8000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
13:25:43, iteration=33100, train_loss=1.263, val_loss=1.449
13:25:50, iteration=33200, train_loss=1.352, val_loss=1.322
13:25:57, iteration=33300, train_loss=1.375, val_loss=1.443
13:26:04, iteration=33400, train_loss=1.382, val_loss=1.515
13:26:11, iteration=33500, train_loss=1.409, val_loss=1.428
13:26:18, iteration=33600, train_loss=1.353, val_loss=1.376
13:26:25, iteration=33700, train_loss=1.345, val_loss=1.409
13:26:31, iteration=33800, train_loss=1.376, val_loss=1.451
13:26:38, iteration=33900, train_loss=1.378, val_loss=1.276
13:26:45, iteration=34000, train_loss=1.488, val_loss=1.432
13:26:52, iteration=34100, train_loss=1.360, val_loss=1.416
13:26:59, iteration=34200, train_loss=1.331, val_loss=1.422
13:27:06, iteration=34300, train_loss=1.359, val_loss=1.439
13:27:13, iteration=34400, train_loss=1.436, val_loss=1.458
13:27:20, iteration=34500, train_loss=1.362, val_loss=1.397
13:27:27, iteration=34600, train_loss=1.417, val_loss=1.247
13:27:34, iteration=34700, train_loss=1.313, val_loss=1.365
13:27:41, iteration=34800, train_loss=1.370, val_loss=1.375
13:27:47, iteration=34900, train_loss=1.470, val_loss=1.384
13:27:54, iteration=35000, train_loss=1.344, val_loss=1.417
"""

log_body = logs.split("~" * 32)[-1]
# %%
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
plt.plot(iterations, train_losses, marker="o", label="Train loss")
plt.plot(iterations, val_losses, marker="o", label="Validation loss")

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
