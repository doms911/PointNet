import torch
from torch import nn
from src.model.shared_mlp import shared_mlp

class TNet(nn.Module):
    def __init__(self, k: int, bn: bool = True):
        super().__init__()
        self.k = k

        # shared MLP (point-wise): [B, k, N] -> [B, 1024, N]
        self.mlp = shared_mlp([k, 64, 128, 1024], bn=bn)

        # FC head: 1024 -> 512 -> 256 -> k*k
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            (nn.BatchNorm1d(512) if bn else nn.Identity()),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            (nn.BatchNorm1d(256) if bn else nn.Identity()),
            nn.ReLU(inplace=True),

            nn.Linear(256, k * k),
        )

        # init last layer to produce ~0 so output starts near Identity after +I
        nn.init.constant_(self.fc[-1].weight, 0.0)
        nn.init.constant_(self.fc[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, k, N]
        returns: [B, k, k]
        """
        B = x.size(0)

        x = self.mlp(x)                 # [B, 1024, N]
        x = x.max(dim=2)[0]             # [B, 1024]
        x = self.fc(x)                  # [B, k*k]

        I = torch.eye(self.k, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)  # [B,k,k]
        x = x.reshape(B, self.k, self.k) + I
        return x
