from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import open3d as o3d

from .models import PointCloud
from .pointcloud import to_open3d_point_cloud


def read_xyzrgb_txt(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an S3DIS text point cloud, tolerating control characters/bad lines.

    Well-formed files use NumPy's fast bulk parser.  The slower line-by-line
    parser is only used when a file contains malformed rows or control
    characters.
    """
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            data = np.loadtxt(handle, dtype=np.float32, ndmin=2)
        if data.shape[0] == 0 or data.shape[1] < 6:
            raise ValueError(f"no valid x y z r g b rows found in {path}")
        data = data[:, :6]
        return data[:, :3], data[:, 3:6]
    except ValueError:
        pass

    rows: list[np.ndarray] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            clean = "".join(ch for ch in line if ch in "\t\r\n" or ord(ch) >= 32).strip()
            if not clean:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                values = np.fromstring(clean, sep=" ")
            if values.size < 6:
                continue
            rows.append(values[:6])
    if not rows:
        raise ValueError(f"no valid x y z r g b rows found in {path}")
    data = np.vstack(rows).astype(np.float32, copy=False)
    return data[:, :3], data[:, 3:6]


def write_ply(path: str | Path, cloud: PointCloud, *, binary: bool = False) -> Path:
    """Write a point cloud with Open3D."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(
        str(path), to_open3d_point_cloud(cloud), write_ascii=not binary
    ):
        raise RuntimeError(f"failed to save point cloud: {path}")
    return path


def save_npz(path: str | Path, cloud: PointCloud) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"xyz": cloud.xyz}
    for key in ("rgb", "semantic_labels", "instance_labels"):
        value = getattr(cloud, key)
        if value is not None:
            payload[key] = value
    np.savez_compressed(path, **payload)
    return path
