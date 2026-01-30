#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import trimesh

from src.model.pointnet import PointNetBaseline, PointNetCls

MODELNET10_CLASSES = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",  # nekad piše "nightstand"
    "sofa",
    "table",
    "toilet",
]


def pick_device(cpu: bool) -> torch.device:
    if cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_points(pts: np.ndarray) -> np.ndarray:
    """Center + scale to unit sphere."""
    pts = pts.astype(np.float32)
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def load_mesh_sample_points(path: Path, npoints: int) -> np.ndarray:
    mesh = trimesh.load(str(path), force="mesh")

    # Sometimes trimesh returns a Scene
    if not isinstance(mesh, trimesh.Trimesh):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception as e:
            raise ValueError(f"Cannot convert {path} to Trimesh. Got {type(mesh)}") from e

    pts, _ = trimesh.sample.sample_surface(mesh, npoints)  # [N,3]
    return normalize_points(pts)


def idx_to_name(idx: int, num_classes: int) -> str:
    if num_classes == 10 and 0 <= idx < len(MODELNET10_CLASSES):
        return MODELNET10_CLASSES[idx]
    return "class"


@torch.no_grad()
def predict_one(
    baseline: torch.nn.Module,
    cls: torch.nn.Module,
    device: torch.device,
    mesh_path: Path,
    npoints: int,
) -> tuple[int, float, list[tuple[int, float]]]:
    pts = load_mesh_sample_points(mesh_path, npoints)        # [N,3]
    x = torch.from_numpy(pts).T.contiguous().unsqueeze(0)    # [1,3,N]
    x = x.to(device)

    x64, _ = baseline(x)
    logits, _ = cls(x64)                                     # [1,C]
    probs = torch.softmax(logits, dim=1).squeeze(0)          # [C]

    k = min(10, probs.numel())
    vals, idxs = torch.topk(probs, k=k)

    pred_idx = int(idxs[0].item())
    pred_prob = float(vals[0].item())
    top10 = [(int(i.item()), float(v.item())) for v, i in zip(vals, idxs)]
    return pred_idx, pred_prob, top10


def main():
    ap = argparse.ArgumentParser(description="PointNet inference on a single mesh (ModelNet10).")
    ap.add_argument("--ckpt", type=str, default="results/pointnet_modelnet10.pt",
                    help="Path to checkpoint (.pt)")
    ap.add_argument("--input", type=str, required=True,
                    help="Path to a single mesh file (.off/.ply/.obj/.stl/...)")
    ap.add_argument("--npoints", type=int, default=1024,
                    help="Override npoints")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    args = ap.parse_args()

    device = pick_device(args.cpu)

    ckpt_path = Path(args.ckpt)
    mesh_path = Path(args.input)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    ckpt_args = ckpt.get("args", {})
    
    print("ckpt args:", ckpt.get("args", {}))

    bn = bool(ckpt_args.get("bn", True))
    num_classes = int(ckpt_args.get("num_classes", 10))
    npoints = int(args.npoints or ckpt_args.get("npoints", 1024))

    baseline = PointNetBaseline(bn=bn).to(device)
    cls = PointNetCls(num_classes=num_classes, bn=bn).to(device)

    baseline.load_state_dict(ckpt["baseline_state"])
    cls.load_state_dict(ckpt["cls_state"])

    baseline.eval()
    cls.eval()

    pred_idx, pred_prob, top10 = predict_one(baseline, cls, device, mesh_path, npoints)

    pred_name = idx_to_name(pred_idx, num_classes)

    print("=" * 72)
    print("PointNet Inference")
    print("-" * 72)
    print(f"ckpt   : {ckpt_path}")
    print(f"device : {device}")
    print(f"npoints: {npoints}")
    print(f"input  : {mesh_path}")
    print("-" * 72)
    print(f"PREDICTED: {pred_name} ({pred_idx})  prob={pred_prob:.4f}")
    print("-" * 72)
    print("Top-10 probabilities:")
    for rank, (idx, prob) in enumerate(top10, start=1):
        name = idx_to_name(idx, num_classes)
        print(f"{rank:>2}. {name:<12} ({idx})  {prob:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
