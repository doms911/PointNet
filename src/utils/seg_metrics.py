import numpy as np
import torch


@torch.no_grad()
def per_shape_iou(pred: torch.Tensor, target: torch.Tensor, num_parts: int) -> float:
    """
    pred:   [N] int64 predicted part labels (0..num_parts-1)
    target: [N] int64 GT part labels
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)


    ious = []
    for p in range(num_parts):
        I = torch.logical_and(pred == p, target == p).sum().item()
        U = torch.logical_or(pred == p, target == p).sum().item()
        if U == 0:
            # part not present in both pred and target; ignore
            continue
        ious.append(I / U)
    if not ious:
        return 1.0
    return float(np.mean(ious))


@torch.no_grad()
def batch_mean_iou(pred: torch.Tensor, target: torch.Tensor, num_parts: int) -> float:
    """
    pred:   [B,N]
    target: [B,N]
    """
    B = pred.shape[0]
    vals = [per_shape_iou(pred[b], target[b], num_parts) for b in range(B)]
    return float(np.mean(vals))
