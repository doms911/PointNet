import torch
from torch import nn
from src.model.tnet import TNet
from src.model.shared_mlp import shared_mlp


class PointNetBaseline(nn.Module):
    def __init__(self, bn: bool):
        super().__init__()
        self.input_transform = TNet(k=3, bn=bn)
        self.mlp1 = shared_mlp([3, 64, 64], bn=bn)
        self.feature_transform = TNet(k=64, bn=bn)

    def forward(self, x):
        T = self.input_transform(x) # [B, 3, 3]
        x = torch.bmm(T, x) # [B, 3, N]
        x = self.mlp1(x) # [B, 64, N]
        A = self.feature_transform(x) # [B, 64, 64] - 
        x = torch.bmm(A, x) # [B, 64, N]
        return x, A
    

class PointNetCls(nn.Module):
    def __init__(self, num_classes: int, bn: bool):
        super().__init__()
        self.mlp = shared_mlp([64, 128, 1024], bn=bn)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512) if bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256) if bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # [B, 64, N]
        x = self.mlp(x) # [B, 1024, N]
        global_feat = x.max(dim=2)[0] # [B, 1024]
        scores = self.fc(global_feat) # [B, num_classes]
        return scores, global_feat

    
class PointNetSeg(nn.Module):
    def __init__(self, num_classes: int, bn: bool, extra_channels: int = 0):
        super().__init__()
        in_ch = 1088 + extra_channels
        self.mlp = shared_mlp([in_ch, 512, 256, 128], bn=bn)
        self.conv_out = nn.Conv1d(128, num_classes, kernel_size=1)

    def forward(self, x, global_features, cls_onehot=None):
        B, _, N = x.shape
        if global_features.dim() == 2:
            global_features = global_features.unsqueeze(-1)  # [B,1024,1]
        global_rep = global_features.expand(-1, -1, N)       # [B,1024,N]

        feats = [x, global_rep]
        if cls_onehot is not None:
            if cls_onehot.dim() == 2:
                cls_onehot = cls_onehot.unsqueeze(-1)        # [B,16,1]
            cls_rep = cls_onehot.expand(-1, -1, N)           # [B,16,N]
            feats.append(cls_rep)

        x = torch.cat(feats, dim=1)
        x = self.mlp(x)
        return self.conv_out(x)

    