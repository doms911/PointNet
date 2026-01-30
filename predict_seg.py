#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch

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


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).to(dtype=torch.float32)


def make_palette(num_classes: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # lijepe, stabilne boje (0..1)
    pal = rng.uniform(0.1, 0.95, size=(num_classes, 3)).astype(np.float32)
    return pal


def ensure_pts_shape(pts: torch.Tensor) -> torch.Tensor:
    """
    Želimo [3, N] float tensor.
    Dataset često vraća [3,N], ali nekad [N,3] - handleamo oba.
    """
    if pts.ndim != 2:
        raise ValueError(f"Expected pts 2D, got shape {tuple(pts.shape)}")
    if pts.shape[0] == 3:
        return pts
    if pts.shape[1] == 3:
        return pts.transpose(0, 1)
    raise ValueError(f"Can't infer point shape. Got {tuple(pts.shape)}")


def build_models(ckpt: dict, device: torch.device):
    ckpt_args = ckpt.get("args", {})
    bn = bool(ckpt_args.get("bn", True))
    num_shape_classes = int(ckpt_args.get("num_shape_classes", 16))
    num_parts = int(ckpt_args.get("num_parts", 50))
    use_cls_onehot = bool(ckpt_args.get("use_cls_onehot", True))
    extra = num_shape_classes if use_cls_onehot else 0

    baseline = PointNetBaseline(bn=bn).to(device)
    global_head = PointNetCls(num_classes=num_shape_classes, bn=bn).to(device)
    seg_head = PointNetSeg(num_classes=num_parts, bn=bn, extra_channels=extra).to(device)

    baseline.load_state_dict(ckpt["baseline_state"], strict=True)
    global_head.load_state_dict(ckpt["global_state"], strict=True)
    seg_head.load_state_dict(ckpt["seg_state"], strict=True)

    baseline.eval()
    global_head.eval()
    seg_head.eval()

    return baseline, global_head, seg_head, num_shape_classes, num_parts, use_cls_onehot


def viz_open3d(points_xyz: np.ndarray, labels: np.ndarray, num_classes: int, title: str, palette: np.ndarray):
    import open3d as o3d

    pts = points_xyz.astype(np.float64)
    colors = palette[labels.clip(0, num_classes - 1)].astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Open3D nema "title" po prozoru svugdje konzistentno, ali ok.
    return pcd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to pointnet_shapenetpart_best.pt")
    ap.add_argument("--data_root", type=str, required=True, help="Path to shapenetcore_partanno... folder")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--index", type=int, default=0, help="Sample index in the chosen split")
    ap.add_argument("--npoints", type=int, default=None, help="Override npoints (otherwise from ckpt args)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--show_gt", action="store_true", help="Show GT next to prediction (if available)")
    args = ap.parse_args()

    device = pick_device(args.cpu)

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    baseline, global_head, seg_head, num_shape_classes, num_parts, use_cls_onehot = build_models(ckpt, device)

    ckpt_args = ckpt.get("args", {})
    npoints = args.npoints if args.npoints is not None else int(ckpt_args.get("npoints", 1024))

    ds = ShapeNetPart(
        root=args.data_root,
        split=args.split,
        npoints=npoints,
        augment=False,
        cache_dir=args.cache_dir,
    )

    if not (0 <= args.index < len(ds)):
        raise IndexError(f"index {args.index} out of range (0..{len(ds)-1})")

    pts, cls_lbl, seg_lbl = ds[args.index]

    # pts: [3,N] or [N,3]
    pts = ensure_pts_shape(pts).float()
    cls_lbl = torch.as_tensor(cls_lbl).long()
    seg_lbl = torch.as_tensor(seg_lbl).long()

    # batchify
    pts_b = pts.unsqueeze(0).to(device)          # [1,3,N]
    cls_b = cls_lbl.unsqueeze(0).to(device)      # [1]
    seg_b = seg_lbl.unsqueeze(0).to(device)      # [1,N]

    with torch.no_grad():
        x64, _A = baseline(pts_b)                # [1,64,N]
        _, global_feat = global_head(x64)        # [1,1024]

        cls_oh = one_hot(cls_b, num_shape_classes).to(device) if use_cls_onehot else None
        seg_logits = seg_head(x64, global_feat, cls_onehot=cls_oh)  # [1,num_parts,N]
        pred = torch.argmax(seg_logits, dim=1)   # [1,N]

    pred_np = pred.squeeze(0).cpu().numpy().astype(np.int64)
    gt_np = seg_b.squeeze(0).cpu().numpy().astype(np.int64)

    pts_np = pts.transpose(0, 1).cpu().numpy()   # [N,3]

    palette = make_palette(num_parts, seed=123)

    # --- visualize ---
    try:
        import open3d as o3d

        pred_pcd = viz_open3d(pts_np, pred_np, num_parts, "PRED", palette)
        geoms = [pred_pcd]

        if args.show_gt:
            gt_pcd = viz_open3d(pts_np, gt_np, num_parts, "GT", palette)
            # pomakni GT u stranu da se vide oba
            gt_pcd.translate((1.5, 0.0, 0.0))
            geoms = [pred_pcd, gt_pcd]

        print(f"Device: {device}")
        print(f"Split: {args.split} | index: {args.index} | npoints: {npoints}")
        print(f"class label (shape): {int(cls_lbl)}")
        print("Showing:", "PRED + GT" if args.show_gt else "PRED")

        o3d.visualization.draw_geometries(geoms)

    except ImportError:
        raise SystemExit(
            "Open3D nije instaliran. Instaliraj ga pa probaj opet:\n"
            "  pip install open3d\n"
        )


if __name__ == "__main__":
    main()
