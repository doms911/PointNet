import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.model.pointnet import PointNetBaseline, PointNetCls
from src.dataset.modelnet_dataset import ModelNetPointCloud


def pick_device(cpu: bool) -> torch.device:
    if cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_cm(cm: np.ndarray, class_names: list[str], normalize: bool, out_path: Path, title: str):
    if normalize:
        cm_plot = cm.astype(np.float64)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, np.maximum(row_sums, 1.0))
    else:
        cm_plot = cm

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_plot)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_plot[i, j]
            txt = f"{val:.2f}" if normalize else str(int(val))
            plt.text(j, i, txt, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def eval_confusion(
    baseline: torch.nn.Module,
    cls: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[np.ndarray, float]:
    y_true_chunks = []
    y_pred_chunks = []

    baseline.eval()
    cls.eval()

    for bi, (pts, labels) in enumerate(loader):
        pts = pts.to(device, non_blocking=True)       # [B,3,N]
        labels = labels.to(device, non_blocking=True) # [B]

        x64, _ = baseline(pts)
        logits, _ = cls(x64)                          # [B,C]
        preds = torch.argmax(logits, dim=1)           # [B]

        # --- robust conversion to numpy (avoids MPS weirdness) ---
        labels_np = labels.detach().to("cpu", dtype=torch.int64).contiguous().numpy()
        preds_np  = preds.detach().to("cpu", dtype=torch.int64).contiguous().numpy()

        # --- sanity checks (fail fast with info) ---
        if (labels_np < 0).any() or (labels_np >= num_classes).any():
            print(f"[BAD LABELS] batch={bi}, dtype={labels_np.dtype}, min={labels_np.min()}, max={labels_np.max()}")
            raise RuntimeError("Found labels outside [0, num_classes-1]. Dataset mapping is wrong.")

        if (preds_np < 0).any() or (preds_np >= num_classes).any():
            print(f"[BAD PREDS] batch={bi}, dtype={preds_np.dtype}, min={preds_np.min()}, max={preds_np.max()}")
            # extra debug: show first few logits shapes
            print("logits shape:", tuple(logits.shape))
            raise RuntimeError(
                "Model produced predictions outside [0, num_classes-1]. "
                "This often happens due to device (MPS) conversion issues."
            )

        y_true_chunks.append(labels_np)
        y_pred_chunks.append(preds_np)

    y_true = np.concatenate(y_true_chunks).astype(int, copy=False)
    y_pred = np.concatenate(y_pred_chunks).astype(int, copy=False)

    cm = build_confusion(y_true, y_pred, num_classes=num_classes)
    acc = float((y_true == y_pred).mean())
    return cm, acc


def main():
    ap = argparse.ArgumentParser(description="Compute confusion matrix on ModelNet10/40 using the Dataset pipeline.")
    ap.add_argument("--ckpt", type=str, default="results/pointnet_modelnet10.pt")
    ap.add_argument("--data_root", type=str, required=True,
                    help="Path to ModelNet root, e.g. data/raw/ModelNet10")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--npoints", type=int, default=None, help="Override npoints (else from ckpt args, fallback 1024)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cache_dir", type=str, default=None, help="Optional cache dir for sampled point clouds")
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    device = pick_device(args.cpu)

    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {})

    bn = bool(ckpt_args.get("bn", True))
    num_classes = int(ckpt_args.get("num_classes", 10))
    npoints = int(args.npoints or ckpt_args.get("npoints", 1024))

    baseline = PointNetBaseline(bn=bn).to(device)
    cls = PointNetCls(num_classes=num_classes, bn=bn).to(device)
    baseline.load_state_dict(ckpt["baseline_state"])
    cls.load_state_dict(ckpt["cls_state"])

    # IMPORTANT: augment=False for evaluation
    ds = ModelNetPointCloud(
        root=args.data_root,
        subset=args.split,
        npoints=npoints,
        augment=False,
        cache_dir=args.cache_dir,
    )

    # use ds.classes so labels match the dataset ordering
    class_names = ds.classes
    if len(class_names) != num_classes:
        print(f"WARNING: dataset has {len(class_names)} classes but model expects {num_classes}.")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type != "cpu"),
    )

    cm, acc = eval_confusion(baseline, cls, loader, device, num_classes=num_classes)

    out_dir = Path(args.out_dir)
    plot_cm(cm, class_names, normalize=False,
            out_path=out_dir / f"confusion_matrix_{args.split}.png",
            title=f"Confusion Matrix ({args.split})")

    plot_cm(cm, class_names, normalize=True,
            out_path=out_dir / f"confusion_matrix_{args.split}_normalized.png",
            title=f"Confusion Matrix ({args.split}, normalized)")

    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Saved -> {out_dir / f'confusion_matrix_{args.split}.png'}")
    print(f"Saved -> {out_dir / f'confusion_matrix_{args.split}_normalized.png'}")


if __name__ == "__main__":
    main()
