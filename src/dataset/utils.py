import numpy as np

def normalize_points(points: np.ndarray) -> np.ndarray:
    """points: [N,3] -> zero-mean + unit sphere"""
    points = points - points.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale
    return points


def random_z_rotate(points: np.ndarray) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float32)
    return points @ R.T


def jitter(points: np.ndarray, sigma=0.01, clip=0.02) -> np.ndarray:
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip).astype(np.float32)
    return points + noise