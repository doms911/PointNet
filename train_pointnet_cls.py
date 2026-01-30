import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.utils import feature_transform_regularizer
from src.model.pointnet import PointNetBaseline, PointNetCls
from src.dataset.modelnet_dataset import ModelNetPointCloud
import time
from tqdm import tqdm


@torch.no_grad()
def evaluate(baseline: nn.Module, cls: nn.Module, loader: DataLoader, device: torch.device, lambda_reg: float):
    baseline.eval()
    cls.eval()
    correct, total = 0, 0
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()

    for points, y in loader:
        points, y = points.to(device), y.to(device)

        x64, A = baseline(points)
        scores, _ = cls(x64)
        loss = ce(scores, y) + lambda_reg * feature_transform_regularizer(A)

        total_loss += float(loss.item()) * points.size(0)
        preds = scores.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += points.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    baseline = PointNetBaseline(bn=args.bn).to(device)
    cls = PointNetCls(num_classes=args.num_classes, bn=args.bn).to(device)

    params = list(baseline.parameters()) + list(cls.parameters())

    # TF default: Adam, lr=0.001, no weight decay
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    # TF: exponential_decay(... staircase=True)
    # decay_step is in "examples": (batch_index * batch_size)
    # PyTorch StepLR steps per optimizer step -> divide by batch_size
    decay_steps = max(1, args.decay_step // args.batch_size)  # 200000//32=6250
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_steps, gamma=args.decay_rate)

    ce = nn.CrossEntropyLoss()

    train_set = ModelNetPointCloud(
        root=args.data_root, subset="train", npoints=args.npoints,
        augment=True, cache_dir=args.cache_dir
    )
    test_set = ModelNetPointCloud(
        root=args.data_root, subset="test", npoints=args.npoints,
        augment=False, cache_dir=args.cache_dir
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False
    )

    best_acc = 0.0
    os.makedirs(args.out_dir, exist_ok=True)

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        baseline.train()
        cls.train()

        t0 = time.time()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        steps_in_epoch = len(train_loader)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{args.epochs:03d}",
            leave=False,
            dynamic_ncols=True,
        )

        for points, y in pbar:
            points, y = points.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)

            x64, A = baseline(points)
            scores, _ = cls(x64)

            loss = ce(scores, y) + args.lambda_reg * feature_transform_regularizer(A)
            loss.backward()
            opt.step()

            global_step += 1
            scheduler.step()

            # TF clips lr to >= 1e-5
            for pg in opt.param_groups:
                pg["lr"] = max(pg["lr"], args.lr_clip)

            bs = points.size(0)
            running_loss += float(loss.item()) * bs
            preds = scores.argmax(dim=1)
            running_correct += int((preds == y).sum().item())
            running_total += bs

            avg_loss = running_loss / max(running_total, 1)
            avg_acc = running_correct / max(running_total, 1)
            epoch_progress = 100.0 * (pbar.n / max(steps_in_epoch, 1))

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc%": f"{avg_acc*100:.2f}",
                "ep%": f"{epoch_progress:5.1f}",
                "lr": f"{opt.param_groups[0]['lr']:.1e}",
            })

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        test_loss, test_acc = evaluate(baseline, cls, test_loader, device, args.lambda_reg)

        improved = test_acc > best_acc
        if improved:
            best_acc = test_acc
            ckpt = {
                "epoch": epoch,
                "baseline_state": baseline.state_dict(),
                "cls_state": cls.state_dict(),
                "opt_state": opt.state_dict(),
                "best_acc": best_acc,
                "args": vars(args),
            }
            torch.save(ckpt, Path(args.out_dir) / "best.pt")

        elapsed = time.time() - t0
        progress_pct = 100.0 * epoch / args.epochs
        flag = "⭐" if improved else "  "

        print(
            f"{flag} Epoch {epoch:03d}/{args.epochs:03d} ({progress_pct:5.1f}%) | "
            f"train loss {train_loss:.4f} acc {train_acc*100:6.2f}% | "
            f"test loss {test_loss:.4f} acc {test_acc*100:6.2f}% | "
            f"best {best_acc*100:6.2f}% | "
            f"lr {opt.param_groups[0]['lr']:.1e} | "
            f"time {elapsed:5.1f}s"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--num_classes", type=int, required=True)

    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="results")

    # TF defaults
    ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lambda_reg", type=float, default=0.001)

    # TF lr decay defaults
    ap.add_argument("--decay_step", type=int, default=200000)
    ap.add_argument("--decay_rate", type=float, default=0.7)
    ap.add_argument("--lr_clip", type=float, default=1e-5)

    ap.add_argument("--bn", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    train(args)


if __name__ == "__main__":
    main()
