from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PYTORCH3D_TO_OPENCV = np.diag([-1.0, -1.0, 1.0])
GLB_Y_UP_TO_Z_UP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
)


def quaternion_to_matrix(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass
class AssetPose:
    rotation: np.ndarray
    translation: np.ndarray
    scale: np.ndarray

    def __post_init__(self):
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        self.rotation /= np.linalg.norm(self.rotation)
        self.translation = np.broadcast_to(self.translation, (3,)).astype(float).copy()
        self.scale = np.broadcast_to(self.scale, (3,)).astype(float).copy()

    @classmethod
    def load(cls, path, object_key="object_0"):
        data = json.loads(Path(path).read_text("utf-8"))[object_key]
        return cls(data["rotation"], data["translation"], data.get("scale", 1))

    def transform(self):
        transform = np.eye(4)
        transform[:3, :3] = quaternion_to_matrix(self.rotation) @ np.diag(self.scale)
        transform[:3, 3] = self.translation
        return transform


def load_pose_entry(path, object_key="object_0"):
    pose = AssetPose.load(path, object_key)
    return pose.rotation, pose.translation, pose.scale


def build_transform(rotation, translation, scale, include_glb_axis_conversion=False):
    pose = AssetPose(rotation, translation, scale)
    transform = np.eye(4)
    transform[:3, :3] = (
        PYTORCH3D_TO_OPENCV
        @ quaternion_to_matrix(pose.rotation).T
        @ np.diag(pose.scale)
    )
    transform[:3, 3] = PYTORCH3D_TO_OPENCV @ pose.translation
    if include_glb_axis_conversion:
        axis_change = np.eye(4)
        axis_change[:3, :3] = GLB_Y_UP_TO_Z_UP
        transform = transform @ axis_change
    return transform


def load_asset_transform(path, pose_space="auto", object_key="object_0"):
    path = Path(path)
    pose = AssetPose.load(path, object_key)
    if pose_space == "auto":
        pose_space = "opencv" if path.stem.endswith("_optimized") else "pytorch3d"
    if pose_space == "opencv":
        return pose.transform()
    if pose_space == "pytorch3d":
        return build_transform(
            pose.rotation,
            pose.translation,
            pose.scale,
            include_glb_axis_conversion=True,
        )
    raise ValueError("pose_space must be auto, opencv or pytorch3d")
