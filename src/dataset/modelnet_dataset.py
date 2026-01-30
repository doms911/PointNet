import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import trimesh
from typing import List, Tuple
from src.dataset.utils import normalize_points, random_z_rotate, jitter


class ModelNetPointCloud(Dataset):
    """
    Expects ModelNet style folder:
      root/
        ModelNet40/ (or ModelNet10/)
          chair/
            train/*.off
            test/*.off
          table/
            train/*.off
            test/*.off
          ...
    """
    def __init__(
        self,
        root: str,
        subset: str,          # "train" or "test"
        npoints: int = 1024,
        augment: bool = False,
        cache_dir: str | None = None,
    ):
        self.root = Path(root)
        self.subset = subset
        self.npoints = npoints
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.files: List[Tuple[Path, int]] = []
        for c in self.classes:
            off_dir = self.root / c / subset
            for f in sorted(off_dir.glob("*.off")):
                self.files.append((f, self.class_to_idx[c]))

        if len(self.files) == 0:
            raise RuntimeError(f"No .off files found under {self.root} (subset={subset}).")

    def __len__(self):
        return len(self.files)

    def _cache_path(self, off_path: Path) -> Path:
        # unique name based on relative path
        rel = off_path.relative_to(self.root).as_posix().replace("/", "__")
        return self.cache_dir / f"{rel}.npy"

    def _load_and_sample(self, off_path: Path) -> np.ndarray:
        mesh = trimesh.load(off_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            # sometimes returns Scene
            mesh = mesh.dump(concatenate=True)

        # sample points uniformly on surface
        pts, _ = trimesh.sample.sample_surface(mesh, self.npoints)
        pts = pts.astype(np.float32)  # [N,3]
        pts = normalize_points(pts)
        del mesh
        return pts

    def __getitem__(self, idx: int):
        off_path, label = self.files[idx]

        if self.cache_dir:
            cp = self._cache_path(off_path)
            if cp.exists():
                pts = np.load(cp).astype(np.float32)
            else:
                pts = self._load_and_sample(off_path)
                np.save(cp, pts)
        else:
            pts = self._load_and_sample(off_path)

        if self.augment:
            pts = random_z_rotate(pts)
            pts = jitter(pts)

        # return as [3,N] for your model
        pts = torch.from_numpy(pts).transpose(0, 1).contiguous()  # [3,N]
        label = torch.tensor(label, dtype=torch.long)
        return pts, label
    