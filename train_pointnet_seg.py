import argparse
from pathlib import Path
import time
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model.pointnet import PointNetBaseline, PointNetCls, PointNetSeg
from src.dataset.shapenetpart_dataset import ShapeNetPart


def pick_device(cpu: bool) -> torch.device:
    if cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --- cache za eye matricu (da se ne radi torch.eye svaki batch) ---
_EYE_CACHE = {}  # key: (device_type, device_index, K, dtype) -> tensor [K,K]


def _get_eye(K: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device.type, device.index, K, dtype)
    I = _EYE_CACHE.get(key)
    if I is None:
        I = torch.eye(K, device=device, dtype=dtype)
        _EYE_CACHE[key] = I
    return I


def feature_transform_regularizer(A: torch.Tensor) -> torch.Tensor:
    # A: [B, 64, 64], encourage orthogonality
    B, K, _ = A.shape
    I = _get_eye(K, A.device, A.dtype).unsqueeze(0)  # [1,K,K]
    return ((torch.bmm(A, A.transpose(1, 2)) - I.expand(B, -1, -1)) ** 2).mean()


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # labels: [B] long -> [B, num_classes] float (same device as labels)
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).to(dtype=torch.float32)


def seg_loss(seg_logits: torch.Tensor, seg_lbl: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    # seg_logits: [B,C,N], seg_lbl: [B,N]
    B, C, N = seg_logits.shape
    logits = seg_logits.transpose(1, 2).reshape(B * N, C)  # [B*N, C]
    labels = seg_lbl.reshape(B * N)                        # [B*N]
    return criterion(logits, labels)


@torch.no_grad()
def update_confmat(confmat: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    """
    confmat: [C,C] long (CPU recommended)
    pred/target: [B,N] long
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    mask = (target >= 0) & (target < num_classes)
    pred = pred[mask]
    target = target[mask]

    idx = target * num_classes + pred
    confmat += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


@torch.no_grad()
def mean_iou_from_confmat(confmat: torch.Tensor, eps: float = 1e-6) -> float:
    conf = confmat.to(torch.float32)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    iou = tp / (tp + fp + fn + eps)

    # ignoriraj klase koje se nikad ne pojave u GT (stabilnije)
    valid = conf.sum(dim=1) > 0
    if valid.any():
        return float(iou[valid].mean().item())
    return float(iou.mean().item())


@torch.no_grad()
def mean_acc_from_confmat(confmat: torch.Tensor, eps: float = 1e-12) -> float:
    conf = confmat.to(torch.float32)
    correct = torch.diag(conf).sum()
    total = conf.sum()
    return float((correct / (total + eps)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Path to shapenetcore_partanno_segmentation_benchmark_v0(_normal)")
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--cache_dir", type=str, default=None)

    ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--bn", action="store_true", default=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--workers", type=int, default=0)

    ap.add_argument("--num_parts", type=int, default=50, help="ShapeNetPart has 50 part labels total")
    ap.add_argument("--num_shape_classes", type=int, default=16, help="ShapeNetPart has 16 object categories")
    ap.add_argument("--lambda_reg", type=float, default=0.001, help="feature transform regularizer weight")
    ap.add_argument("--use_cls_onehot", action="store_true", default=True,
                    help="Concatenate shape-class one-hot (recommended for part seg)")

    args = ap.parse_args()

    device = pick_device(args.cpu)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # datasets
    train_ds = ShapeNetPart(
        root=args.data_root,
        split="train",
        npoints=args.npoints,
        augment=True,
        cache_dir=args.cache_dir,
    )
    val_ds = ShapeNetPart(
        root=args.data_root,
        split="val",
        npoints=args.npoints,
        augment=False,
        cache_dir=args.cache_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
    )

    # models
    baseline = PointNetBaseline(bn=args.bn).to(device)

    # Only used to get global_feat (1024). Logits are not used.
    global_head = PointNetCls(num_classes=args.num_shape_classes, bn=args.bn).to(device)

    extra = args.num_shape_classes if args.use_cls_onehot else 0
    seg_head = PointNetSeg(num_classes=args.num_parts, bn=args.bn, extra_channels=extra).to(device)

    params = list(baseline.parameters()) + list(global_head.parameters()) + list(seg_head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    best_miou = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ---- train ----
        baseline.train()
        global_head.train()
        seg_head.train()

        tr_loss_sum = 0.0
        n_batches = 0

        # metrike računamo NA KRAJU epohe (preko confmat)
        train_conf = torch.zeros(args.num_parts, args.num_parts, dtype=torch.int64, device="cpu")

        train_pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", leave=False)
        for pts, cls_lbl, seg_lbl in train_pbar:
            pts = pts.to(device, non_blocking=True)         # [B,3,N]
            cls_lbl = cls_lbl.to(device, non_blocking=True) # [B]
            seg_lbl = seg_lbl.to(device, non_blocking=True) # [B,N]

            opt.zero_grad(set_to_none=True)

            x64, A = baseline(pts)                # [B,64,N]
            _, global_feat = global_head(x64)     # [B,1024]

            cls_oh = one_hot(cls_lbl, args.num_shape_classes) if args.use_cls_onehot else None
            seg_logits = seg_head(x64, global_feat, cls_onehot=cls_oh)  # [B, num_parts, N]

            loss = seg_loss(seg_logits, seg_lbl, criterion)
            loss = loss + args.lambda_reg * feature_transform_regularizer(A)

            loss.backward()
            opt.step()

            tr_loss_sum += float(loss.item())
            n_batches += 1

            # samo loss između iteracija (brzo)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            # predikcije skupljamo, ali metriku računamo tek na kraju epohe
            with torch.no_grad():
                pred = torch.argmax(seg_logits, dim=1)  # [B,N]
                update_confmat(
                    train_conf,
                    pred.detach().cpu(),
                    seg_lbl.detach().cpu(),
                    args.num_parts,
                )

        tr_loss = tr_loss_sum / max(n_batches, 1)
        tr_acc = mean_acc_from_confmat(train_conf)
        tr_miou = mean_iou_from_confmat(train_conf)

        # ---- val ----
        baseline.eval()
        global_head.eval()
        seg_head.eval()

        va_loss_sum = 0.0
        n_batches = 0
        val_conf = torch.zeros(args.num_parts, args.num_parts, dtype=torch.int64, device="cpu")

        val_pbar = tqdm(val_loader, desc=f"Val   {epoch}/{args.epochs}", leave=False)
        with torch.no_grad():
            for pts, cls_lbl, seg_lbl in val_pbar:
                pts = pts.to(device, non_blocking=True)
                cls_lbl = cls_lbl.to(device, non_blocking=True)
                seg_lbl = seg_lbl.to(device, non_blocking=True)

                x64, A = baseline(pts)
                _, global_feat = global_head(x64)

                cls_oh = one_hot(cls_lbl, args.num_shape_classes) if args.use_cls_onehot else None
                seg_logits = seg_head(x64, global_feat, cls_onehot=cls_oh)

                loss = seg_loss(seg_logits, seg_lbl, criterion)
                loss = loss + args.lambda_reg * feature_transform_regularizer(A)

                va_loss_sum += float(loss.item())
                n_batches += 1

                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

                pred = torch.argmax(seg_logits, dim=1)
                update_confmat(
                    val_conf,
                    pred.detach().cpu(),
                    seg_lbl.detach().cpu(),
                    args.num_parts,
                )

        va_loss = va_loss_sum / max(n_batches, 1)
        va_acc = mean_acc_from_confmat(val_conf)
        va_miou = mean_iou_from_confmat(val_conf)

        dt = time.time() - t0
        is_best = va_miou > best_miou

        if is_best:
            best_miou = va_miou
            ckpt = {
                "args": vars(args),
                "baseline_state": baseline.state_dict(),
                "global_state": global_head.state_dict(),
                "seg_state": seg_head.state_dict(),
                "best_miou": best_miou,
                "epoch": epoch,
            }
            torch.save(ckpt, out_dir / "pointnet_shapenetpart_best.pt")

        tqdm.write(
            f"{'⭐' if is_best else ' '} Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc*100:5.2f}% mIoU {tr_miou:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc*100:5.2f}% mIoU {va_miou:.4f} | "
            f"best {best_miou:.4f} | time {dt:5.1f}s"
        )

    print(f"\nBest val mIoU: {best_miou:.4f}")
    print(f"Saved -> {out_dir / 'pointnet_shapenetpart_best.pt'}")


if __name__ == "__main__":
    main()
