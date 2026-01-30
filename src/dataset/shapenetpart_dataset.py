import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.utils import normalize_points, random_z_rotate, jitter
import re

_SYNSET_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")

# ShapeNetPart: global part labels 0..49 (standard benchmark)
CATEGORY_TO_PARTS: Dict[str, List[int]] = {
    "airplane": list(range(0, 4)),
    "bag": list(range(4, 6)),
    "cap": list(range(6, 8)),
    "car": list(range(8, 12)),
    "chair": list(range(12, 16)),
    "earphone": list(range(16, 19)),
    "guitar": list(range(19, 22)),
    "knife": list(range(22, 24)),
    "lamp": list(range(24, 28)),
    "laptop": list(range(28, 30)),
    "motorbike": list(range(30, 36)),
    "mug": list(range(36, 38)),
    "pistol": list(range(38, 41)),
    "rocket": list(range(41, 44)),
    "skateboard": list(range(44, 47)),
    "table": list(range(47, 50)),
}


def _read_synset_mapping(root: Path) -> Dict[str, str]:
    """
    Reads synsetoffset2category.txt
    lines: <category_name>\t<synset_id>
    Example: Airplane 02691156
    """
    p = root / "synsetoffset2category.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Is dataset extracted correctly?")

    def norm(name: str) -> str:
        return name.strip().lower().replace(" ", "_")

    mapping: Dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Bad line in {p}: {line!r}")
        raw_name, synset = parts
        mapping[norm(raw_name)] = synset.strip()
    return mapping


def _read_split_ids(root: Path, split: str) -> List[str]:
    split_dir = root / "train_test_split"
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing {split_dir}. Is dataset extracted correctly?")

    if split == "train":
        fn = split_dir / "shuffled_train_file_list.json"
    elif split == "val":
        fn = split_dir / "shuffled_val_file_list.json"
    elif split == "test":
        fn = split_dir / "shuffled_test_file_list.json"
    else:
        raise ValueError("split must be one of: train/val/test")

    return json.loads(fn.read_text())


def _parse_sid(sid: str) -> Optional[Tuple[str, str]]:
    s = str(sid).strip().replace("\\", "/")
    parts = [p for p in s.split("/") if p]
    if not parts:
        return None

    # synset: bilo gdje u stringu (8 znamenki)
    m = _SYNSET_RE.search(s)
    if not m:
        return None
    synset = m.group(1)

    # shape id: zadnji segment bez ekstenzije
    last = parts[-1]
    base = last.split(".")[0]

    return synset, base


def _find_existing_file(dirpath: Path, base: str, exts: List[str]) -> Optional[Path]:
    """
    Try exact name base+ext for given ext list. If not found, try glob base.* and pick first.
    """
    for ext in exts:
        p = dirpath / f"{base}{ext}"
        if p.exists():
            return p
    # fallback: any extension
    candidates = sorted(dirpath.glob(f"{base}.*"))
    return candidates[0] if candidates else None


def _map_seg_to_global(seg: np.ndarray, part_ids: List[int]) -> np.ndarray:
    """
    Map per-category labels to global part ids if needed.
    Handles:
      - already global (subset of part_ids)
      - local 1-based (1..len(parts))
      - local 0-based (0..len(parts)-1)
    """
    seg = seg.reshape(-1)
    part_set = set(part_ids)
    uniq = set(np.unique(seg).tolist())

    if uniq.issubset(part_set):
        return seg

    max_local = len(part_ids)
    if seg.min() >= 1 and seg.max() <= max_local:
        return np.array([part_ids[i - 1] for i in seg], dtype=np.int64)
    if seg.min() >= 0 and seg.max() < max_local:
        return np.array([part_ids[i] for i in seg], dtype=np.int64)

    return seg


class ShapeNetPart(Dataset):
    """
    Returns:
      points: [3, N] float32
      cls:    scalar long (0..15) category id
      seg:    [N] long (0..49) global part label id
    """

    def __init__(
        self,
        root: str,
        split: str,               # "train" | "val" | "test"
        npoints: int = 2048,
        augment: bool = False,
        use_normals: bool = False,   # dataset includes normals; we use xyz only
        cache_dir: str | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.npoints = npoints
        self.augment = augment
        self.use_normals = use_normals  # for future
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        synset_map = _read_synset_mapping(self.root)

        # Keep only categories we know (16)
        self.categories = list(CATEGORY_TO_PARTS.keys())
        missing = [c for c in self.categories if c not in synset_map]
        if missing:
            raise RuntimeError(f"Expected 16 categories, missing in synset file: {missing}")

        self.cat_to_synset = {c: synset_map[c] for c in self.categories}
        self.cat_to_idx = {c: i for i, c in enumerate(self.categories)}
        self.synset_to_cat = {v: k for k, v in self.cat_to_synset.items()}
        self.idx_to_cat = {i: c for c, i in self.cat_to_idx.items()}

        ids = _read_split_ids(self.root, split)

        self.items: List[Tuple[Path, Path, int]] = []
        for sid in ids:
            parsed = _parse_sid(sid)
            if parsed is None:
                continue
            synset, base = parsed

            if synset not in self.synset_to_cat:
                continue  # ignore categories outside our 16

            cat = self.synset_to_cat[synset]
            cls_idx = self.cat_to_idx[cat]

            points_dir = self.root / synset / "points"
            labels_dir = self.root / synset / "points_label"

            if not points_dir.exists() or not labels_dir.exists():
                continue

            # standard files are .pts and .seg, but some dumps use .txt, etc.
            pts_path = _find_existing_file(points_dir, base, exts=[".pts", ".txt"])
            seg_path = _find_existing_file(labels_dir, base, exts=[".seg", ".txt"])

            if pts_path is None or seg_path is None:
                continue

            self.items.append((pts_path, seg_path, cls_idx))

        if not self.items:
            # give a helpful hint of what's inside one synset
            example_synset = next((p for p in self.root.iterdir() if p.is_dir() and p.name.isdigit()), None)
            hint = ""
            if example_synset:
                pts = list((example_synset / "points").glob("*"))[:3]
                seg = list((example_synset / "points_label").glob("*"))[:3]
                hint = f"\nExample synset: {example_synset.name}\npoints examples: {pts}\npoints_label examples: {seg}"
            raise RuntimeError(
                f"No ShapeNetPart files found for split={split} under {self.root}.{hint}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def _cache_path(self, pts_path: Path) -> Path:
        rel = pts_path.relative_to(self.root).as_posix().replace("/", "__")
        return self.cache_dir / f"{rel}.npz"

    def _load_one(self, pts_path: Path, seg_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.loadtxt(pts_path).astype(np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Bad pts file: {pts_path} shape={pts.shape}")

        xyz = pts[:, :3]  # [P,3]

        seg = np.loadtxt(seg_path).astype(np.int64)
        seg = seg.reshape(-1)  # ensure 1D
        if seg.shape[0] != xyz.shape[0]:
            raise ValueError(f"Bad seg file: {seg_path} seg={seg.shape} xyz={xyz.shape}")

        P = xyz.shape[0]
        if P >= self.npoints:
            idx = np.random.choice(P, self.npoints, replace=False)
        else:
            idx = np.random.choice(P, self.npoints, replace=True)

        xyz = xyz[idx]
        seg = seg[idx]

        xyz = normalize_points(xyz)
        return xyz, seg

    def __getitem__(self, idx: int):
        pts_path, seg_path, cls_idx = self.items[idx]
        part_ids = CATEGORY_TO_PARTS[self.idx_to_cat[cls_idx]]

        if self.cache_dir:
            cp = self._cache_path(pts_path)
            if cp.exists():
                d = np.load(cp)
                xyz = d["xyz"].astype(np.float32)
                seg = d["seg"].astype(np.int64)
            else:
                xyz, seg = self._load_one(pts_path, seg_path)
                seg = _map_seg_to_global(seg, part_ids)
                np.savez(cp, xyz=xyz, seg=seg)
        else:
            xyz, seg = self._load_one(pts_path, seg_path)

        seg = _map_seg_to_global(seg, part_ids)

        if self.augment:
            xyz = random_z_rotate(xyz)
            xyz = jitter(xyz)

        pts = torch.from_numpy(xyz).transpose(0, 1).contiguous()  # [3,N]
        cls = torch.tensor(cls_idx, dtype=torch.long)
        seg = torch.from_numpy(seg).long()  # [N]
        return pts, cls, seg
