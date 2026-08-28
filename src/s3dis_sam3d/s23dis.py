from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from .config import s23dis_area
from .frames import RGBDFrame
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
class Frame(RGBDFrame):
    # TODO rename to S23Frame
    depth_scale = S23DIS_DEPTH_SCALE
    invalid_depth_value = S23DIS_INVALID_DEPTH

    stem: str
    room: str
    frame_id: int
    uuid: str
    pose_path: Path
    xyz_path: Path | None = None
    projection_type: str = "regular"

    @cached_property
    def pose(self):
        return json.loads(self.pose_path.read_text("utf-8"))

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

    def project_camera_points(self, points):
        if self.projection_type == "pano":
            return project_pano_points(points, self.image_shape)
        return project_pinhole_points(points, self.intrinsics)

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

    def point_cloud(
        self,
        stride=4,
        depth_min=0.1,
        depth_max=8.0,
        mask=None,
        world_coordinates=True,
        from_global_xyz=False,
    ):
        if not from_global_xyz:
            return super().point_cloud(
                stride=stride,
                depth_min=depth_min,
                depth_max=depth_max,
                mask=mask,
                world_coordinates=world_coordinates,
            )

        rgb = self.rgb
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

        return PointCloud(
            points,
            colors,
            metadata=self._point_cloud_metadata(coordinates),
        )

    def _point_cloud_metadata(self, coordinate_frame):
        return {"frame": self.stem, "coordinate_frame": coordinate_frame}


@dataclass(frozen=True)
class S23Room:
    """One 2D-3D-S room and its regular or panorama frames."""

    dataset: S23Dataset = field(repr=False, compare=False)
    area: str
    name: str
    frames: tuple[Frame, ...]

    @property
    def key(self):
        return f"{self.area}/{self.name}"

    @property
    def projection_type(self):
        return self.dataset.projection_type

    def list_uuids(self, frame_id=None):
        frames = self.frames
        if frame_id is not None:
            frames = tuple(frame for frame in frames if frame.frame_id == frame_id)
        return sorted({frame.uuid for frame in frames})

    def get_frame(self, frame_id, uuid=None):
        frames = tuple(frame for frame in self.frames if frame.frame_id == frame_id)
        if not frames:
            raise KeyError(f"frame {frame_id!r} is unavailable in {self.key}")
        uuid = self.dataset.default_uuid if uuid is None else uuid
        if uuid == "first":
            return frames[0]
        try:
            return next(frame for frame in frames if frame.uuid == uuid)
        except StopIteration as error:
            raise KeyError(
                f"UUID {uuid!r} is unavailable for frame {frame_id!r} in {self.key}"
            ) from error

    def reconstruct(
        self,
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
        frames = list(self.frames)
        if frame_id is not None:
            frames = [self.get_frame(frame_id, uuid)]
        frames = (
            [frame for frame in frames if frame.has_xyz]
            if from_global_xyz
            else [frame for frame in frames if frame.has_depth]
        )
        if isinstance(mask, Mapping):
            frames = [frame for frame in frames if frame.stem in mask]
        if frame_id is None and max_frames is not None:
            if max_frames < 1:
                raise ValueError("max_frames must be positive")
            frames = frames[:max_frames]

        frame_iterator = frames
        if progress:
            from tqdm import tqdm

            frame_iterator = tqdm(frames, desc=f"Reconstructing {self.key}")

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
            raise ValueError(f"no usable frames for room: {self.key}")

        cloud = PointCloud(
            np.concatenate([cloud.xyz for cloud in clouds]),
            np.concatenate([cloud.rgb for cloud in clouds]),
            metadata={
                "area": self.area,
                "room": self.name,
                "frame_count": len(clouds),
                "coordinate_frame": clouds[0].metadata["coordinate_frame"],
            },
        )
        return voxel_downsample(cloud, voxel_size)

    def visualize(
        self,
        *,
        meshes=(),
        point_size=2.0,
        window_name=None,
        show_coordinate_frame=True,
        **reconstruct_options,
    ):
        cloud = self.reconstruct(**reconstruct_options)
        visualize_point_clouds(
            [cloud, *meshes],
            window_name=window_name or f"2D-3D-S | {self.key}",
            point_size=point_size,
            show_coordinate_frame=show_coordinate_frame,
        )
        return cloud

    def save_ply(self, output_path, **reconstruct_options):
        from .io import write_ply

        return write_ply(output_path, self.reconstruct(**reconstruct_options))

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, index):
        if isinstance(index, str):
            for frame in self.frames:
                if frame.stem == index:
                    return frame
            raise KeyError(index)
        return self.frames[index]

    def __repr__(self):
        return f"S23Room(key={self.key!r}, frames={len(self)})"


class S23Dataset:
    """Discover 2D-3D-S rooms and index their camera frames."""

    def __init__(
        self,
        area_path=None,
        projection_type="regular",
        area="Area_1",
        default_uuid="first",
    ):
        using_default_path = area_path is None
        area_path = s23dis_area(area) if using_default_path else area_path
        self.area_path = Path(area_path)
        path_area = self.area_path.name
        self.area = str(area) if using_default_path else (
            f"Area_{path_area.split('_', 1)[1]}"
            if path_area.casefold().startswith("area_")
            else path_area
        )
        if projection_type not in {"regular", "pano"}:
            raise ValueError("projection_type must be 'regular' or 'pano'")
        self.projection_type = projection_type
        self.default_uuid = default_uuid
        self.data_dir = self.area_path / (
            "data" if projection_type == "regular" else "pano"
        )
        self.pose_dir = self.data_dir / "pose"
        self.rgb_dir = self.data_dir / "rgb"
        self.depth_dir = self.data_dir / "depth"
        self.xyz_dir = self.data_dir / "global_xyz"
        self.frames = tuple(self._index_frames(projection_type))
        grouped = {}
        for frame in self.frames:
            grouped.setdefault(frame.room, []).append(frame)
        for frames in grouped.values():
            frames.sort(key=lambda frame: (frame.frame_id, frame.uuid))
        self.rooms = tuple(
            S23Room(self, self.area, name, tuple(frames))
            for name, frames in sorted(grouped.items())
        )

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
        return [(room.name, len(room)) for room in self.rooms]

    def room(self, room):
        if isinstance(room, S23Room):
            if room.dataset is self:
                return room
            raise ValueError("room belongs to another S23Dataset")
        if isinstance(room, int):
            return self.rooms[room]
        query = str(room).replace("\\", "/").strip("/").casefold()
        exact = [item for item in self.rooms if item.key.casefold() == query]
        matches = exact or [item for item in self.rooms if item.name.casefold() == query]
        if len(matches) != 1:
            raise ValueError(f"room not found or ambiguous: {room}")
        return matches[0]

    def room_frames(self, room):
        return list(self.room(room).frames)

    def list_uuids(self, room=None, frame_id=None):
        """List UUIDs in the Area, one room, or one room/frame."""
        if room is not None:
            return self.room(room).list_uuids(frame_id)
        frames = self.frames
        if frame_id is not None:
            frames = tuple(frame for frame in frames if frame.frame_id == frame_id)
        return sorted({frame.uuid for frame in frames})

    def get_frame(self, room, frame_id, uuid=None):
        return self.room(room).get_frame(frame_id, uuid)

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
        warnings.warn(
            "S23Dataset.reconstruct is deprecated; use dataset.room(...).reconstruct",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.room(room).reconstruct(
            frame_id=frame_id,
            uuid=uuid,
            stride=stride,
            depth_min=depth_min,
            depth_max=depth_max,
            voxel_size=voxel_size,
            max_frames=max_frames,
            mask=mask,
            world_coordinates=world_coordinates,
            from_global_xyz=from_global_xyz,
            progress=progress,
        )

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
        warnings.warn(
            "S23Dataset.visualize_room is deprecated; use dataset.room(...).visualize",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.room(room).visualize(
            meshes=meshes,
            point_size=point_size,
            window_name=window_name,
            show_coordinate_frame=show_coordinate_frame,
            **reconstruct_options,
        )

    def save_room_ply(self, room, output_path, **reconstruct_options):
        warnings.warn(
            "S23Dataset.save_room_ply is deprecated; use dataset.room(...).save_ply",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.room(room).save_ply(output_path, **reconstruct_options)

    def __len__(self):
        return len(self.rooms)

    def __iter__(self):
        return iter(self.rooms)

    def __getitem__(self, index):
        return self.room(index)

    def __repr__(self):
        return (
            f"S23Dataset(area={self.area!r}, projection_type={self.projection_type!r}, "
            f"rooms={len(self)})"
        )
