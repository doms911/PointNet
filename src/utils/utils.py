import torch


def feature_transform_regularizer(A: torch.Tensor) -> torch.Tensor:
    # A: [B, 64, 64]
    B, K, _ = A.shape
    I = torch.eye(K, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(B, 1, 1)
    diff = torch.bmm(A, A.transpose(1, 2)) - I
    return (diff ** 2).mean()

