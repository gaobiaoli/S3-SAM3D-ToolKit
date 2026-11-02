"""Pose tables use metres, local-to-BIM transforms, and xyzw quaternions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from ..config import slabim_registration_dir


def load_pose_table(path):
    """Return timestamps (N,) and local-to-parent matrices (N, 4, 4)."""
    values = np.loadtxt(path, ndmin=2)
    if values.shape[1] != 8 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Invalid pose table: {path}")
    if not np.all(np.diff(values[:, 0]) > 0):
        raise ValueError(f"Pose timestamps must increase: {path}")
    if not np.allclose(np.linalg.norm(values[:, 4:], axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError(f"Invalid pose quaternion: {path}")
    transforms = np.repeat(np.eye(4)[None], len(values), axis=0)
    transforms[:, :3, 3] = values[:, 1:4]
    transforms[:, :3, :3] = Rotation.from_quat(values[:, 4:]).as_matrix()
    return values[:, 0], transforms


def save_pose_table(path, times, transforms):
    """Write timestamp tx ty tz qx qy qz qw without rounding the source times."""
    quaternions = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()
    np.savetxt(
        path,
        np.column_stack((times, transforms[:, :3, 3], quaternions)),
        fmt="%.18e",
    )


def load_camera_poses(region, root=None, *, version="v1"):
    """Return image filenames, timestamps, and camera-to-BIM matrices.

    ``root`` is a version directory, defaulting to ``dataset/slabim/<version>``.
    The default remains v1; pass ``version="v2"`` for whole-region correction.
    Filenames are relative to ``sensor_data/<region>/images/data``.
    """
    root = slabim_registration_dir(version) if root is None else Path(root).expanduser()
    directory = root / region
    names = (directory / "camera_frames.txt").read_text(encoding="utf-8").splitlines()
    times, transforms = load_pose_table(directory / "camera_pose_to_bim.txt")
    if len(names) != len(times) or len(set(names)) != len(names):
        raise ValueError(f"Image index does not match camera poses: {region}")
    return names, times, transforms


def map_to_bim(path, times):
    """Read the official constant SLAM-map-to-BIM transform.

    SLABIM repeats this transform per LiDAR frame; it is not a trajectory.
    """
    source_times, transforms = load_pose_table(path)
    if len(times) != len(source_times) or not np.allclose(times, source_times, atol=1e-4, rtol=0):
        raise ValueError("Map-to-BIM timestamps differ from LiDAR timestamps")
    translation_spread = np.linalg.norm(transforms[:, :3, 3] - transforms[0, :3, 3], axis=1)
    relative = transforms[:, :3, :3] @ transforms[0, :3, :3].T
    if translation_spread.max() > 1e-6 or Rotation.from_matrix(relative).magnitude().max() > 1e-6:
        raise ValueError("Expected a constant SLAM-map-to-BIM transform")
    return transforms[0]


def smooth_poses(transforms):
    """Frozen v1 smoothing: five-frame Savitzky–Golay, sign-continuous quaternions."""
    count = len(transforms)
    if count < 3:
        return transforms.copy()
    window = min(5, count if count % 2 else count - 1)
    degree = min(3, window - 1)
    result = transforms.copy()
    result[:, :3, 3] = savgol_filter(transforms[:, :3, 3], window, degree, axis=0)
    quaternions = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()
    for index in range(1, count):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1
    quaternions = savgol_filter(quaternions, window, degree, axis=0)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    result[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
    return result


def interpolate_camera_poses(times, poses, query, extrinsic):
    """Interpolate LiDAR-to-BIM, then right-multiply camera-to-LiDAR."""
    query = np.asarray(query, dtype=float)
    extrinsic = np.asarray(extrinsic, dtype=float)
    if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
        raise ValueError("Invalid camera extrinsic")
    rotation = extrinsic[:3, :3]
    if (
        not np.allclose(extrinsic[3], [0, 0, 0, 1])
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
    ):
        raise ValueError("Invalid camera extrinsic")
    if len(times) < 2 or not np.all(np.diff(times) > 0) or not np.isfinite(query).all():
        raise ValueError("Invalid interpolation timestamps")
    if len(query) and (query.min() < times[0] or query.max() > times[-1]):
        raise ValueError("Refusing trajectory extrapolation")
    result = np.repeat(np.eye(4)[None], len(query), axis=0)
    result[:, :3, :3] = Slerp(times - times[0], Rotation.from_matrix(poses[:, :3, :3]))(
        query - times[0]
    ).as_matrix()
    result[:, :3, 3] = np.column_stack(
        [np.interp(query, times, poses[:, axis, 3]) for axis in range(3)]
    )
    return result @ extrinsic
