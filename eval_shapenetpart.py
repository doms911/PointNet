#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model.pointnet import PointNetBaseline, PointNetCls, PointNetSeg
from src.dataset.shapenetpart_dataset import ShapeNetPart, CATEGORY_TO_PARTS


SHAPENET_CLASSES = {
    0: "airplane", 1: "bag", 2: "cap", 3: "car", 4: "chair",
    5: "earphone", 6: "guitar", 7: "knife", 8: "lamp",
    9: "laptop", 10: "motorbike", 11: "mug", 12: "pistol",
    13: "rocket", 14: "skateboard", 15: "table",
}


def pick_device(cpu: bool) -> torch.device:
    if cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()


@torch.no_grad()
def iou_for_parts(pred: np.ndarray, gt: np.ndarray, part_ids: list) -> float:
    """mean IoU over given part_ids for ONE shape"""
    ious = []
    for pid in part_ids:
        pred_mask = pred == pid
        gt_mask = gt == pid
        if not gt_mask.any():
            continue
        inter = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else np.nan


def map_pred_to_global(pred: np.ndarray, part_ids: list) -> np.ndarray:
    """
    If predictions are local (per-category) labels, map to global part ids.
    Handles:
      - already global (subset of part_ids)
      - local 1-based (1..len(parts))
      - local 0-based (0..len(parts)-1)
    """
    pred = pred.reshape(-1)
    part_set = set(part_ids)
    uniq = set(np.unique(pred).tolist())

    if uniq.issubset(part_set):
        return pred

    max_local = len(part_ids)
    if pred.min() >= 1 and pred.max() <= max_local:
        return np.array([part_ids[i - 1] for i in pred], dtype=np.int64)
    if pred.min() >= 0 and pred.max() < max_local:
        return np.array([part_ids[i] for i in pred], dtype=np.int64)

    return pred


def compute_part_ce_loss(
    logits: torch.Tensor,  # [N, 50]
    seg_lbl: torch.Tensor,  # [N] global labels
    part_ids: list,
) -> torch.Tensor:
    """
    Compute CE loss using only the valid part channels for this category.
    This makes loss meaningful even if labels/preds are local vs global.
    """
    # logits_sub: [N, K]
    logits_sub = logits[:, part_ids]

    # map global -> local index 0..K-1
    mapping = torch.full((50,), -1, dtype=torch.long, device=seg_lbl.device)
    mapping[torch.tensor(part_ids, device=seg_lbl.device)] = torch.arange(
        len(part_ids), device=seg_lbl.device
    )
    seg_local = mapping[seg_lbl]

    # guard: if anything unmapped, ignore those points
    valid = seg_local >= 0
    if not valid.any():
        return torch.tensor(0.0, device=seg_lbl.device)

    return torch.nn.functional.cross_entropy(logits_sub[valid], seg_local[valid])


def build_models(ckpt: dict, device: torch.device):
    args = ckpt["args"]
    baseline = PointNetBaseline(bn=args["bn"]).to(device)
    global_head = PointNetCls(args["num_shape_classes"], bn=args["bn"]).to(device)
    seg_head = PointNetSeg(
        num_classes=args["num_parts"],
        bn=args["bn"],
        extra_channels=args["num_shape_classes"] if args["use_cls_onehot"] else 0,
    ).to(device)

    baseline.load_state_dict(ckpt["baseline_state"])
    global_head.load_state_dict(ckpt["global_state"])
    seg_head.load_state_dict(ckpt["seg_state"])

    baseline.eval()
    global_head.eval()
    seg_head.eval()
    return baseline, global_head, seg_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.cpu)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt["args"]

    baseline, global_head, seg_head = build_models(ckpt, device)

    ds = ShapeNetPart(
        root=args.data_root,
        split=args.split,
        npoints=ckpt_args["npoints"],
        augment=False,
        cache_dir=args.cache_dir,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    per_class_ious = defaultdict(list)
    loss_sum = 0.0
    acc_sum = 0.0
    count_sum = 0

    for pts, cls_lbl, seg_lbl in loader:
        pts = pts.to(device)
        cls_lbl = cls_lbl.to(device)
        seg_lbl = seg_lbl.to(device)

        with torch.no_grad():
            x64, _ = baseline(pts)
            _, gf = global_head(x64)
            cls_oh = one_hot(cls_lbl, ckpt_args["num_shape_classes"]).to(device)
            logits = seg_head(x64, gf, cls_oh)
            pred = logits.argmax(dim=1)  # [B,N]

        # accuracy in global label space (per-point)
        for b in range(pred.size(0)):
            cid = int(cls_lbl[b])
            cname = SHAPENET_CLASSES[cid]
            part_ids = CATEGORY_TO_PARTS[cname]

            pred_np = pred[b].cpu().numpy()
            pred_np = map_pred_to_global(pred_np, part_ids)
            pred_g = torch.from_numpy(pred_np).to(seg_lbl.device)
            acc_sum += (pred_g == seg_lbl[b]).sum().item()
            count_sum += seg_lbl[b].numel()

            # loss over valid part channels for this category
            logits_b = logits[b].transpose(0, 1).contiguous()  # [N, 50]
            loss_sum += compute_part_ce_loss(logits_b, seg_lbl[b], part_ids).item()

        for b in range(pred.size(0)):
            cid = int(cls_lbl[b])
            cname = SHAPENET_CLASSES[cid]
            part_ids = CATEGORY_TO_PARTS[cname]

            pred_np = pred[b].cpu().numpy()
            pred_np = map_pred_to_global(pred_np, part_ids)
            miou = iou_for_parts(pred_np, seg_lbl[b].cpu().numpy(), part_ids)
            if not np.isnan(miou):
                per_class_ious[cname].append(miou)

    print("\nPer-class mIoU:")
    class_mious = []
    for cname in sorted(per_class_ious.keys()):
        miou = float(np.mean(per_class_ious[cname]))
        class_mious.append(miou)
        print(f"{cname:12s}: {miou:.4f}")

    print("-" * 40)
    print(f"Class mIoU (mean over 16): {np.mean(class_mious):.4f}")
    if count_sum > 0:
        print(f"Point Accuracy: {acc_sum / count_sum:.4f}")
    if len(loader) > 0:
        print(f"Part CE Loss: {loss_sum / len(loader):.4f}")


if __name__ == "__main__":
    main()
