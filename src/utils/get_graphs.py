import re
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r'^\s*(?P<star>⭐)?\s*Epoch\s+(?P<epoch>\d+)/(?:\s*)?(?P<epochs>\d+)\s*'
    r'\(\s*(?P<pct>[\d.]+)%\)\s*\|\s*'
    r'train loss\s+(?P<tr_loss>[\d.]+)\s+acc\s+(?P<tr_acc>[\d.]+)%\s*\|\s*'
    r'test loss\s+(?P<te_loss>[\d.]+)\s+acc\s+(?P<te_acc>[\d.]+)%\s*\|\s*'
    r'best\s+(?P<best>[\d.]+)%\s*\|\s*'
    r'lr\s+(?P<lr>[\deE+\-\.]+)\s*\|\s*'
    r'time\s+(?P<time>[\d.]+)s\s*$'
)


def parse_training_log(text: str) -> Dict[int, Dict[str, Any]]:
    by_epoch: Dict[int, Dict[str, Any]] = {}

    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue

        m = LINE_RE.match(line)
        if not m:
            continue

        epoch = int(m["epoch"])
        by_epoch[epoch] = {
            "is_best": bool(m["star"]),
            "epoch": epoch,
            "epochs_total": int(m["epochs"]),
            "progress_pct": float(m["pct"]),
            "train_loss": float(m["tr_loss"]),
            "train_acc": float(m["tr_acc"]) / 100.0,   # 0..1
            "test_loss": float(m["te_loss"]),
            "test_acc": float(m["te_acc"]) / 100.0,    # 0..1
            "best_acc": float(m["best"]) / 100.0,      # 0..1
            "lr": float(m["lr"]),
            "time_s": float(m["time"]),
        }

    return by_epoch


def to_series(data: Dict[int, Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # sort by epoch
    epochs_sorted = np.array(sorted(data.keys()), dtype=int)

    train_acc = np.array([data[e]["train_acc"] for e in epochs_sorted], dtype=float)
    test_acc  = np.array([data[e]["test_acc"]  for e in epochs_sorted], dtype=float)
    train_loss = np.array([data[e]["train_loss"] for e in epochs_sorted], dtype=float)
    test_loss  = np.array([data[e]["test_loss"]  for e in epochs_sorted], dtype=float)

    return epochs_sorted, train_acc, test_acc, train_loss, test_loss


def add_best_ticks(ax, best_epoch: int, best_values: List[float], xlim: Tuple[int, int] | None = None, ylim: Tuple[float, float] | None = None):
    # X ticks: keep existing + best_epoch
    xt = ax.get_xticks()
    xt = [x for x in xt if np.isfinite(x)]
    xt = sorted(set([int(round(x)) for x in xt] + [best_epoch]))
    if xlim:
        xt = [x for x in xt if xlim[0] <= x <= xlim[1]]
    ax.set_xticks(xt)

    # Y ticks: keep existing + best values
    yt = ax.get_yticks()
    yt = [y for y in yt if np.isfinite(y)]
    yt = sorted(set(list(yt) + best_values))
    if ylim:
        yt = [y for y in yt if ylim[0] <= y <= ylim[1]]
    ax.set_yticks(yt)


def main():
    with open("logs.txt", "r", encoding="utf-8") as f:
        log_text = f.read()

    data = parse_training_log(log_text)
    if not data:
        raise RuntimeError("No parse. Check the log format.")

    epochs, train_acc, test_acc, train_loss, test_loss = to_series(data)

    # best by TEST accuracy
    best_i = int(np.argmax(test_acc))
    best_epoch = int(epochs[best_i])
    best_test_acc = float(test_acc[best_i])
    best_train_acc = float(train_acc[best_i])

    # ---- Accuracy plot ----
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.plot(epochs, train_acc, label="Train")
    ax.plot(epochs, test_acc, label="Test")

    add_best_ticks(
        ax,
        best_epoch=best_epoch,
        best_values=[best_test_acc, best_train_acc],
        xlim=(int(epochs.min()), int(epochs.max())),
        ylim=(0.0, 1.0)
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train/Test Accuracy over Epochs")
    ax.grid(True)
    ax.legend()
    plt.show()

    # ---- Loss plot (ticks at loss values from best TEST-acc epoch) ----
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.plot(epochs, train_loss, label="Train")
    ax.plot(epochs, test_loss, label="Test")

    add_best_ticks(
        ax,
        best_epoch=best_epoch,
        best_values=[float(train_loss[best_i]), float(test_loss[best_i])],
        xlim=(int(epochs.min()), int(epochs.max())),
        ylim=(0.0, float(max(train_loss.max(), test_loss.max())))
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train/Test Loss over Epochs")
    ax.grid(True)
    ax.legend()
    plt.show()

    print(f"Best test accuracy = {best_test_acc*100:.2f}% at epoch {best_epoch}")


if __name__ == "__main__":
    main()
