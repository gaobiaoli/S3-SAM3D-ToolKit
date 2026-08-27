from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image

from .config import s23dis_area
from .models import PointCloud
from .pointcloud import transform_points, visualize_point_clouds, voxel_downsample
from .utils import (
    backproject_pano,
    backproject_regular,
    camera_to_world_from_pose,
    points_from_global_xyz,
    project_pano_points,
    project_pinhole_points,
)

S23DIS_DEPTH_SCALE = 512.0
S23DIS_INVALID_DEPTH = 65535

FRAME_PATTERN = re.compile(
    r"^camera_(?P<uuid>[0-9a-fA-F]+)_(?P<room>.+?)_frame_"
    r"(?P<frame_id>\d+|equirectangular)(?P<suffix>.*)$"
)
ASSET_PATTERN = re.compile(
    r"_domain_rgb_(?P<class_name>[A-Za-z][A-Za-z0-9_]*?)_(?P<instance_id>\d+)$"
)


def _read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255


def _read_depth(path):
    with Image.open(path) as image:
        raw = np.asarray(image, dtype=np.uint16)
    depth = raw.astype(np.float32) / S23DIS_DEPTH_SCALE
    depth[raw == S23DIS_INVALID_DEPTH] = 0
    return depth


def _read_global_xyz(path):
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    xyz = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return xyz[:, :, :3][:, :, ::-1].astype(np.float32)


def parse_stem(stem):
    """Parse a 2D-3D-S frame or a derived SAM3D asset stem."""
    stem = Path(stem).stem
    for suffix in ("_optimized", "_depth"):
        stem = stem.removesuffix(suffix)
    match = FRAME_PATTERN.match(stem)
    if match is None:
        raise ValueError(f"invalid 2D-3D-S stem: {stem}")

    item = match.groupdict()
    item["room_name"] = item["room"]
    item["frame_id"] = 0 if item["frame_id"] == "equirectangular" else int(item["frame_id"])
    asset = ASSET_PATTERN.search(stem)
    if asset is not None:
        item.update(asset.groupdict())
        item["instance_id"] = int(item["instance_id"])
        item["instance_name"] = f"{item['class_name']}_{item['instance_id']}"
    item.pop("suffix")
    return item


def _mask_for_frame(mask, frame):
    if isinstance(mask, Mapping):
        return mask.get(frame.stem)
    if callable(mask):
        return mask(frame)
    return mask


@dataclass(frozen=True)
class Frame:
    stem: str
    room: str
    frame_id: int
    uuid: str
    pose_path: Path
    rgb_path: Path
    depth_path: Path | None = None
    xyz_path: Path | None = None
    projection_type: str = "regular"

    @property
    def rgb(self):
        return _read_rgb(self.rgb_path)

    @cached_property
    def pose(self):
        return json.loads(self.pose_path.read_text("utf-8"))

    @property
    def has_depth(self):
        return self.depth_path is not None

    @property
    def depth(self):
        if self.depth_path is None:
            raise FileNotFoundError(f"depth is unavailable for {self.stem}")
        return _read_depth(self.depth_path)

    @property
    def has_xyz(self):
        return self.xyz_path is not None

    @property
    def xyz(self):
        if self.xyz_path is None:
            raise FileNotFoundError(f"global_xyz is unavailable for {self.stem}")
        return _read_global_xyz(self.xyz_path)

    @cached_property
    def intrinsics(self):
        return np.asarray(self.pose["camera_k_matrix"], dtype=np.float32)

    @cached_property
    def camera_to_world(self):
        return camera_to_world_from_pose(self.pose)

    @cached_property
    def world_to_camera(self):
        return np.linalg.inv(self.camera_to_world)

    @cached_property
    def image_shape(self):
        with Image.open(self.rgb_path) as image:
            return image.height, image.width

    def project_camera_points(self, points):
        if self.projection_type == "pano":
            return project_pano_points(points, self.image_shape)
        return project_pinhole_points(points, self.intrinsics)

    def project_world_points(self, points):
        points = transform_points(points, self.world_to_camera)
        return self.project_camera_points(points)

    def _backproject(
        self,
        stride=1,
        depth_min=0.1,
        depth_max=10.0,
        mask=None,
    ):
        depth = self.depth
        mask = _mask_for_frame(mask, self)
        if self.projection_type == "pano":
            points, ys, xs = backproject_pano(
                depth,
                stride,
                depth_min,
                depth_max,
                mask,
            )
        else:
            points, ys, xs = backproject_regular(
                depth,
                self.intrinsics,
                stride,
                depth_min,
                depth_max,
                mask,
            )
        return points, ys, xs, depth

    def point_map(self, world_coordinates=False, mask=None):
        points, ys, xs, depth = self._backproject(
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
        stride=4,
        depth_min=0.1,
        depth_max=8.0,
        mask=None,
        world_coordinates=True,
        from_global_xyz=False,
    ):
        rgb = self.rgb
        if from_global_xyz:
            points, colors = points_from_global_xyz(
                self.xyz,
                rgb,
                stride,
                _mask_for_frame(mask, self),
            )
            if world_coordinates:
                coordinates = "world"
            else:
                points = transform_points(points, self.world_to_camera)
                coordinates = "camera"
        else:
            points, ys, xs, _ = self._backproject(
                stride,
                depth_min,
                depth_max,
                mask,
            )
            colors = rgb[ys, xs]
            if world_coordinates:
                points = transform_points(points, self.camera_to_world)
            coordinates = "world" if world_coordinates else "camera"

        return PointCloud(
            points,
            colors,
            metadata={"frame": self.stem, "coordinate_frame": coordinates},
        )


class S23Dataset:
    """Stanford 2D-3D-S frame reader and room reconstructor."""

    def __init__(
        self,
        area_path=None,
        projection_type="regular",
        area="Area_1",
        default_uuid="first",
    ):
        area_path = s23dis_area(area) if area_path is None else area_path
        self.area_path = Path(area_path)
        self.default_uuid = default_uuid
        self.data_dir = self.area_path / (
            "data" if projection_type == "regular" else "pano"
        )
        self.pose_dir = self.data_dir / "pose"
        self.rgb_dir = self.data_dir / "rgb"
        self.depth_dir = self.data_dir / "depth"
        self.xyz_dir = self.data_dir / "global_xyz"
        self.frames = self._index_frames(projection_type)
        self.rooms = {}
        for frame in self.frames:
            self.rooms.setdefault(frame.room, []).append(frame)
        for frames in self.rooms.values():
            frames.sort(key=lambda frame: (frame.frame_id, frame.uuid))

    def _index_frames(self, projection_type):
        frames = []
        for pose_path in sorted(self.pose_dir.glob("*_pose.json")):
            stem = pose_path.name.removesuffix("_pose.json")
            try:
                metadata = parse_stem(stem)
            except ValueError:
                continue
            rgb_path = self.rgb_dir / f"{stem}_rgb.png"
            depth_path = self.depth_dir / f"{stem}_depth.png"
            xyz_path = self.xyz_dir / f"{stem}_global_xyz.exr"
            if rgb_path.exists() and (depth_path.exists() or xyz_path.exists()):
                frames.append(
                    Frame(
                        stem=stem,
                        room=metadata["room"],
                        frame_id=metadata["frame_id"],
                        uuid=metadata["uuid"],
                        pose_path=pose_path,
                        rgb_path=rgb_path,
                        depth_path=depth_path if depth_path.exists() else None,
                        xyz_path=xyz_path if xyz_path.exists() else None,
                        projection_type=projection_type,
                    )
                )
        return frames

    def list_rooms(self):
        return [(room, len(frames)) for room, frames in sorted(self.rooms.items())]

    def room_frames(self, room):
        return list(self.rooms[room])

    def list_uuids(self, room=None, frame_id=None):
        """List UUIDs in the Area, one room, or one room/frame."""
        frames = self.frames if room is None else self.rooms[room]
        if frame_id is not None:
            frames = [frame for frame in frames if frame.frame_id == frame_id]
        return sorted({frame.uuid for frame in frames})

    def get_frame(self, room, frame_id, uuid=None):
        frames = [
            frame
            for frame in self.rooms[room]
            if frame.frame_id == frame_id
        ]
        uuid = self.default_uuid if uuid is None else uuid
        if uuid == "first":
            return frames[0]
        return next(frame for frame in frames if frame.uuid == uuid)

    def reconstruct(
        self,
        room,
        frame_id=None,
        uuid=None,
        stride=4,
        depth_min=0.1,
        depth_max=8.0,
        voxel_size=0.03,
        max_frames=None,
        mask=None,
        world_coordinates=True,
        from_global_xyz=False,
        progress=False,
    ):
        frames = self.room_frames(room)
        if frame_id is not None:
            frames = [self.get_frame(room, frame_id, uuid)]
        frames = (
            [frame for frame in frames if frame.has_xyz]
            if from_global_xyz
            else [frame for frame in frames if frame.has_depth]
        )
        if isinstance(mask, Mapping):
            frames = [frame for frame in frames if frame.stem in mask]
        if frame_id is None and max_frames is not None:
            frames = frames[:max_frames]

        frame_iterator = frames
        if progress:
            from tqdm import tqdm

            frame_iterator = tqdm(frames, desc=f"Reconstructing {room}")

        clouds = []
        for frame in frame_iterator:
            cloud = frame.point_cloud(
                stride=stride,
                depth_min=depth_min,
                depth_max=depth_max,
                mask=mask,
                world_coordinates=world_coordinates,
                from_global_xyz=from_global_xyz,
            )
            if len(cloud.xyz):
                clouds.append(cloud)
        if not clouds:
            raise ValueError(f"no usable frames for room: {room}")

        cloud = PointCloud(
            np.concatenate([cloud.xyz for cloud in clouds]),
            np.concatenate([cloud.rgb for cloud in clouds]),
            metadata={
                "room": room,
                "frame_count": len(clouds),
                "coordinate_frame": clouds[0].metadata["coordinate_frame"],
            },
        )
        return voxel_downsample(cloud, voxel_size)

    def visualize_room(
        self,
        room,
        *,
        meshes=(),
        point_size=2.0,
        window_name=None,
        show_coordinate_frame=True,
        **reconstruct_options,
    ):
        cloud = self.reconstruct(room, **reconstruct_options)
        visualize_point_clouds(
            [cloud, *meshes],
            window_name=window_name or f"2D-3D-S | {room}",
            point_size=point_size,
            show_coordinate_frame=show_coordinate_frame,
        )
        return cloud

    def save_room_ply(self, room, output_path, **reconstruct_options):
        from .io import write_ply

        return write_ply(output_path, self.reconstruct(room, **reconstruct_options))
