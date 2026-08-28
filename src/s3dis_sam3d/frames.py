"""Shared calibrated RGB-D frame behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image

from .models import PointCloud
from .pointcloud import transform_points


@dataclass(frozen=True)
class RGBDFrame(ABC):
    """Common geometry interface for calibrated RGB-D dataset frames.

    Subclasses own dataset-specific identifiers, image decoding, calibration,
    projection models, masks, and metadata. This base class owns operations
    whose meaning is identical once camera-space points are available.
    """

    rgb_path: Path
    depth_path: Path | None

    depth_scale: ClassVar[float] = 1.0
    invalid_depth_value: ClassVar[int | None] = None

    def _read_rgb(self):
        with Image.open(self.rgb_path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    @property
    def rgb(self):
        return self._read_rgb()

    @property
    def has_depth(self):
        return self.depth_path is not None

    def _read_depth(self):
        if self.depth_path is None:
            raise FileNotFoundError(f"depth is unavailable for {self.rgb_path.stem}")
        with Image.open(self.depth_path) as image:
            raw = np.asarray(image, dtype=np.uint16)
        depth = raw.astype(np.float32) / self.depth_scale
        if self.invalid_depth_value is not None:
            depth[raw == self.invalid_depth_value] = 0
        return depth

    @property
    def depth(self):
        return self._read_depth()

    @cached_property
    def world_to_camera(self):
        return np.linalg.inv(self.camera_to_world).astype(np.float32)

    @property
    def camera_position(self):
        return np.asarray(self.camera_to_world[:3, 3], dtype=np.float32).copy()

    @cached_property
    def image_shape(self):
        depth_path = getattr(self, "depth_path", None)
        path = depth_path if depth_path is not None else self.rgb_path
        with Image.open(path) as image:
            return image.height, image.width

    @abstractmethod
    def project_camera_points(self, points):
        """Project camera-space points and return image coordinates and depth."""

    def project_world_points(self, points):
        camera_points = transform_points(points, self.world_to_camera)
        return self.project_camera_points(camera_points)

    @abstractmethod
    def _backproject(self, stride=1, depth_min=0.1, depth_max=10.0, mask=None):
        """Return ``(points, rows, columns, depth)`` in camera coordinates."""

    @abstractmethod
    def _point_cloud_metadata(self, coordinate_frame):
        """Return dataset-specific metadata for a generated point cloud."""

    def point_map(self, world_coordinates=False, mask=None):
        """Return a dense ``(H, W, 3)`` camera- or world-space point map."""

        points, ys, xs, depth = self._backproject(
            stride=1,
            depth_min=None,
            depth_max=None,
            mask=mask,
        )
        if world_coordinates:
            points = transform_points(points, self.camera_to_world)

        result = np.zeros((*depth.shape, 3), dtype=np.float32)
        result[ys, xs] = points
        return result

    def point_cloud(
        self,
        *,
        stride=4,
        depth_min=0.1,
        depth_max=8.0,
        world_coordinates=True,
        mask=None,
    ):
        """Back-project this frame into the shared :class:`PointCloud` model."""

        points, ys, xs, _ = self._backproject(
            stride=stride,
            depth_min=depth_min,
            depth_max=depth_max,
            mask=mask,
        )
        if world_coordinates:
            points = transform_points(points, self.camera_to_world)
        coordinate_frame = "world" if world_coordinates else "camera"
        return PointCloud(
            xyz=points,
            rgb=self.rgb[ys, xs],
            metadata=self._point_cloud_metadata(coordinate_frame),
        )
