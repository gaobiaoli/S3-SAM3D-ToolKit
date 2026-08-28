from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class BoundingBox3D:
    min_bound: np.ndarray
    max_bound: np.ndarray

    @property
    def center(self):
        return (self.min_bound + self.max_bound) / 2

    @property
    def extent(self):
        return self.max_bound - self.min_bound

    @classmethod
    def from_points(cls, points):
        points = np.asarray(points, dtype=np.float32)
        return cls(points.min(axis=0), points.max(axis=0))

    def as_dict(self):
        return {
            "min_bound": self.min_bound.tolist(),
            "max_bound": self.max_bound.tolist(),
            "center": self.center.tolist(),
            "extent": self.extent.tolist(),
        }

@dataclass
class Mesh:
    pass

@dataclass
class IFC:
    pass

@dataclass
class PointCloud:
    xyz: np.ndarray
    rgb: np.ndarray | None = None
    semantic_labels: np.ndarray | None = None
    instance_labels: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, dtype=np.float32)
        if self.rgb is not None:
            self.rgb = np.asarray(self.rgb, dtype=np.float32)
        if self.semantic_labels is not None:
            self.semantic_labels = np.asarray(self.semantic_labels, dtype=np.int32)
        if self.instance_labels is not None:
            self.instance_labels = np.asarray(self.instance_labels, dtype=np.int32)

    def select(self, indices):
        return PointCloud(
            self.xyz[indices],
            None if self.rgb is None else self.rgb[indices],
            None if self.semantic_labels is None else self.semantic_labels[indices],
            None if self.instance_labels is None else self.instance_labels[indices],
            dict(self.metadata),
        )
    def visualize(self,together_with=[],window_name="Point Cloud", show_axis=False, axis_size=0.5, **kwargs):
        from .pointcloud import visualize_point_clouds

        visualize_point_clouds(
            [self] + together_with,
            window_name=window_name,
            show_coordinate_frame=show_axis,
            coordinate_frame_size=axis_size,
            **kwargs
        )

class GLBMesh:
    """SAM3D GLB mesh backed by Open3D."""

    def __init__(
        self,
        glb_path,
        pose_path=None,
        pose_space="auto",
        object_key="object_0",
        camera_to_world=None,
    ):
        self.glb_path = Path(glb_path)
        self.mesh = o3d.io.read_triangle_mesh(
            str(self.glb_path), enable_post_processing=True
        )
        if self.mesh.is_empty():
            raise ValueError(f"empty mesh: {self.glb_path}")
        self.transform_matrix = np.eye(4)
        self.pose_path = None
        if pose_path is not None:
            self.apply_pose(
                pose_path,
                pose_space=pose_space,
                object_key=object_key,
                camera_to_world=camera_to_world,
            )

    @classmethod
    def load_posed(
        cls,
        glb_path,
        pose_path,
        pose_space="auto",
        object_key="object_0",
        camera_to_world=None,
    ):
        return cls(
            glb_path,
            pose_path=pose_path,
            pose_space=pose_space,
            object_key=object_key,
            camera_to_world=camera_to_world,
        )

    def get(
        self,
        pose_path=None,
        pose_space="auto",
        object_key="object_0",
        camera_to_world=None,
    ):
        """Return the underlying Open3D mesh, optionally applying a pose first."""
        if pose_path is not None:
            self.apply_pose(
                pose_path,
                pose_space=pose_space,
                object_key=object_key,
                camera_to_world=camera_to_world,
            )
        return self.mesh

    def copy(self):
        result = object.__new__(GLBMesh)
        result.glb_path = self.glb_path
        result.mesh = copy.deepcopy(self.mesh)
        result.transform_matrix = self.transform_matrix.copy()
        result.pose_path = self.pose_path
        return result

    def apply_transform(self, transform):
        transform = np.asarray(transform, dtype=np.float64)
        self.mesh.transform(transform)
        self.transform_matrix = transform @ self.transform_matrix
        return self

    def apply_pose(
        self,
        pose_path,
        pose_space="auto",
        object_key="object_0",
        camera_to_world=None,
    ):
        from .sam3d.pose import load_asset_transform

        self.pose_path = Path(pose_path)
        transform = load_asset_transform(
            self.pose_path, pose_space=pose_space, object_key=object_key
        )
        if camera_to_world is not None:
            transform = np.asarray(camera_to_world) @ transform
        return self.apply_transform(transform)

    def sample_points(
        self,
        point_count=20_000,
        method="uniform",
        voxel_size=None,
        seed=42,
    ):
        o3d.utility.random.seed(seed)
        if method == "uniform":
            cloud = self.mesh.sample_points_uniformly(point_count)
        elif method == "poisson":
            cloud = self.mesh.sample_points_poisson_disk(point_count)
        else:
            raise ValueError("method must be 'uniform' or 'poisson'")
        if voxel_size:
            cloud = cloud.voxel_down_sample(voxel_size)
        colors = np.asarray(cloud.colors) if cloud.has_colors() else None
        return PointCloud(
            np.asarray(cloud.points),
            colors,
            metadata={
                "source": str(self.glb_path),
                "transform_matrix": self.transform_matrix.copy(),
            },
        )

    def bounding_box(self):
        box = self.mesh.get_axis_aligned_bounding_box()
        return BoundingBox3D(
            np.asarray(box.min_bound, dtype=np.float32),
            np.asarray(box.max_bound, dtype=np.float32),
        )

    def export(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_triangle_mesh(str(output_path), self.mesh):
            raise RuntimeError(f"failed to save mesh: {output_path}")
        return output_path

    def visualize(self, window_name="SAM3D GLB", show_axis=True, axis_size=0.5,**kwargs):
        from .pointcloud import visualize_point_clouds

        visualize_point_clouds(
            [self],
            window_name=window_name,
            show_coordinate_frame=show_axis,
            coordinate_frame_size=axis_size,
            **kwargs
        )
