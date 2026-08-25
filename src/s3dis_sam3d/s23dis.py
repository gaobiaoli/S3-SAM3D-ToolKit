from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import s23dis_area
from .models import PointCloud
from .pointcloud import transform_points, visualize_point_clouds, voxel_downsample

S23DIS_DEPTH_SCALE = 512.0
S23DIS_INVALID_DEPTH = 65535

FRAME_PATTERN = re.compile(
    r"^camera_(?P<uuid>[0-9a-fA-F]+)_(?P<room>.+?)_frame_"
    r"(?P<frame_id>\d+|equirectangular)(?P<suffix>.*)$"
)
ASSET_PATTERN = re.compile(
    r"_domain_rgb_(?P<class_name>[A-Za-z][A-Za-z0-9_]*?)_(?P<instance_id>\d+)$"
)


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


@dataclass(frozen=True)
class FrameInfo:
    stem: str
    room: str
    frame_id: int
    uuid: str
    pose_path: Path
    rgb_path: Path
    depth_path: Path | None
    xyz_path: Path | None


class S23Dataset:
    """Stanford 2D-3D-S frame reader and room reconstructor."""

    frame_pattern = FRAME_PATTERN
    parse_stem = staticmethod(parse_stem)

    def __init__(
        self,
        area_path=None,
        image_type="regular",
        area="Area_1",
        default_uuid="first",
    ):
        area_path = s23dis_area(area) if area_path is None else area_path
        self.area_path = Path(area_path)
        self.image_type = image_type
        self.default_uuid = default_uuid
        self.data_dir = self.area_path / ("data" if image_type == "regular" else "pano")
        self.pose_dir = self.data_dir / "pose"
        self.rgb_dir = self.data_dir / "rgb"
        self.depth_dir = self.data_dir / "depth"
        self.xyz_dir = self.data_dir / "global_xyz"
        self.frames = self._index_frames()
        self.rooms = {}
        self.pose_cache = {}
        for frame in self.frames:
            self.rooms.setdefault(frame.room, []).append(frame)
        for frames in self.rooms.values():
            frames.sort(key=lambda frame: (frame.frame_id, frame.uuid))

    def _index_frames(self):
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
                    FrameInfo(
                        stem,
                        metadata["room"],
                        metadata["frame_id"],
                        metadata["uuid"],
                        pose_path,
                        rgb_path,
                        depth_path if depth_path.exists() else None,
                        xyz_path if xyz_path.exists() else None,
                    )
                )
        return frames

    def list_rooms(self):
        return [(room, len(frames)) for room, frames in sorted(self.rooms.items())]

    def room_frames(self, room):
        return list(self.rooms[room])

    def get_room_frames(self, room):
        return self.room_frames(room)

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

    def load_pose(self, frame):
        if frame.pose_path not in self.pose_cache:
            self.pose_cache[frame.pose_path] = json.loads(frame.pose_path.read_text("utf-8"))
        return self.pose_cache[frame.pose_path]

    def get_depth_path(self, room, frame_id, uuid=None):
        frame = self.get_frame(room, frame_id, uuid)
        if frame.depth_path is None:
            raise FileNotFoundError(f"depth is unavailable for {frame.stem}")
        return frame.depth_path

    def get_depth(self, room, frame_id, uuid=None, backproject=False, world_coordinates=False):
        if backproject:
            return self.point_map(room, frame_id, uuid, world_coordinates)
        return self.load_depth(self.get_depth_path(room, frame_id, uuid))

    def get_k(self, room, frame_id, uuid=None):
        frame = self.get_frame(room, frame_id, uuid)
        return self.intrinsics(self.load_pose(frame))

    def get_xyz(self, room, frame_id, uuid=None):
        frame = self.get_frame(room, frame_id, uuid)
        if frame.xyz_path is None:
            raise FileNotFoundError(f"global_xyz is unavailable for {frame.stem}")
        return self.load_global_xyz(frame.xyz_path)

    def get_image(self, room, frame_id, uuid=None):
        return self.load_rgb(self.get_frame(room, frame_id, uuid).rgb_path)

    def get_image_path(self, room, frame_id, uuid=None):
        return self.get_frame(room, frame_id, uuid).rgb_path

    @staticmethod
    def load_rgb(path):
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255

    @staticmethod
    def load_depth(path, depth_scale=S23DIS_DEPTH_SCALE):
        raw = np.asarray(Image.open(path), dtype=np.uint16)
        depth = raw.astype(np.float32) / depth_scale
        depth[raw == S23DIS_INVALID_DEPTH] = 0
        return depth

    @staticmethod
    def load_global_xyz(path):
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2

        xyz = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return xyz[:, :, :3][:, :, ::-1].astype(np.float32)

    @staticmethod
    def intrinsics(pose):
        return np.asarray(pose["camera_k_matrix"], dtype=np.float32)

    @staticmethod
    def euler_xyz_to_matrix(euler):
        rx, ry, rz = np.asarray(euler)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        rx_matrix = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry_matrix = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz_matrix = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return rz_matrix @ ry_matrix @ rx_matrix

    @classmethod
    def camera_to_world_from_pose(cls, pose):
        rotation = np.asarray(pose["camera_rt_matrix"])[:3, :3]
        adjustment = cls.euler_xyz_to_matrix(pose["final_camera_rotation"])
        transform = np.eye(4)
        transform[:3, :3] = adjustment @ rotation @ adjustment
        transform[:3, 3] = pose["camera_location"]
        return transform

    def camera_to_world(self, room, frame_id, uuid=None):
        frame = self.get_frame(room, frame_id, uuid)
        return self.camera_to_world_from_pose(self.load_pose(frame))

    def get_camera2world_transform(self, room, frame_id, uuid=None):
        return self.camera_to_world(room, frame_id, uuid)

    def world_to_camera(self, room, frame_id, uuid=None):
        return np.linalg.inv(self.camera_to_world(room, frame_id, uuid))

    def project_camera_points(self, points, image_shape, intrinsics=None):
        """Project camera-space points and return pixel coordinates and depth."""
        points = np.asarray(points, dtype=np.float32)
        height, width = image_shape
        if self.image_type == "pano":
            depth = np.linalg.norm(points, axis=1)
            valid = np.isfinite(points).all(axis=1) & (depth > 1e-6)
            points, depth = points[valid], depth[valid]
            theta = np.arctan2(points[:, 0], points[:, 2])
            phi = np.arctan2(-points[:, 1], np.hypot(points[:, 0], points[:, 2]))
            u = np.mod((theta + np.pi) / (2 * np.pi) * width, width)
            v = (0.5 - phi / np.pi) * height
        else:
            depth = points[:, 2]
            valid = np.isfinite(points).all(axis=1) & (depth > 1e-6)
            points, depth = points[valid], depth[valid]
            intrinsics = np.asarray(intrinsics)
            u = intrinsics[0, 0] * points[:, 0] / depth + intrinsics[0, 2]
            v = intrinsics[1, 1] * points[:, 1] / depth + intrinsics[1, 2]
        return np.column_stack((u, v)).astype(np.float32), depth.astype(np.float32)

    def project_world_points(self, points, room, frame_id, uuid=None):
        """Project world-space points into one frame."""
        frame = self.get_frame(room, frame_id, uuid)
        with Image.open(frame.rgb_path) as image:
            image_shape = (image.height, image.width)
        pose = self.load_pose(frame)
        points = transform_points(points, self.world_to_camera(room, frame_id, uuid))
        return self.project_camera_points(points, image_shape, self.intrinsics(pose))

    @classmethod
    def get_camera_rotation_from_pose(cls, pose):
        return cls.camera_to_world_from_pose(pose)[:3, :3]

    @classmethod
    def transform_mesh(cls, mesh, pose):
        result = copy.deepcopy(mesh)
        transform = cls.camera_to_world_from_pose(pose)
        if hasattr(result, "apply_transform"):
            result.apply_transform(transform)
        else:
            result.transform(transform)
        return result

    @staticmethod
    def prepare_mask(mask, shape=None):
        if mask is None:
            return None
        if isinstance(mask, (str, Path)):
            mask = np.asarray(Image.open(mask))
        elif isinstance(mask, Image.Image):
            mask = np.asarray(mask)
        else:
            mask = np.asarray(mask)

        if mask.ndim == 3:
            rgb = np.any(mask[..., :3] > 0, axis=-1)
            mask = mask[..., 3] > 0 if mask.shape[2] == 4 and not rgb.any() else rgb
        else:
            mask = mask > 0
        if shape is not None and mask.shape != shape:
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8)).resize(
                    shape[::-1], Image.Resampling.NEAREST
                )
            ) > 0
        return mask

    @staticmethod
    def _mask_for_frame(mask, frame):
        if isinstance(mask, Mapping):
            return mask.get(frame.stem)
        if callable(mask):
            return mask(frame)
        return mask

    @staticmethod
    def backproject_regular(
        depth,
        intrinsics,
        stride=1,
        depth_min=0.1,
        depth_max=10.0,
        mask=None,
    ):
        height, width = depth.shape
        ys, xs = np.mgrid[0:height:stride, 0:width:stride]
        z = depth[ys, xs]
        valid = np.isfinite(z) & (z > 0)
        if depth_min is not None:
            valid &= z >= depth_min
        if depth_max is not None:
            valid &= z <= depth_max
        mask = S23Dataset.prepare_mask(mask, depth.shape)
        if mask is not None:
            valid &= mask[ys, xs]

        u, v, z = xs[valid], ys[valid], z[valid]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
        return points.astype(np.float32), ys[valid], xs[valid]

    @staticmethod
    def backproject_pano(depth, stride=1, depth_min=0.1, depth_max=10.0, mask=None):
        height, width = depth.shape
        ys, xs = np.mgrid[0:height:stride, 0:width:stride]
        radius = depth[ys, xs]
        valid = np.isfinite(radius) & (radius > 0)
        if depth_min is not None:
            valid &= radius >= depth_min
        if depth_max is not None:
            valid &= radius <= depth_max
        mask = S23Dataset.prepare_mask(mask, depth.shape)
        if mask is not None:
            valid &= mask[ys, xs]

        u, v, radius = xs[valid], ys[valid], radius[valid]
        theta = u / width * 2 * np.pi - np.pi
        phi = np.pi / 2 - v / height * np.pi
        points = np.column_stack(
            (
                radius * np.cos(phi) * np.sin(theta),
                -radius * np.sin(phi),
                radius * np.cos(phi) * np.cos(theta),
            )
        )
        return points.astype(np.float32), ys[valid], xs[valid]

    @staticmethod
    def points_from_global_xyz(xyz, rgb, stride=4, mask=None):
        ys, xs = np.mgrid[0 : xyz.shape[0] : stride, 0 : xyz.shape[1] : stride]
        points = xyz[ys, xs]
        colors = rgb[ys, xs]
        valid = np.isfinite(points).all(axis=-1) & ~np.all(np.abs(points) < 1e-8, axis=-1)
        mask = S23Dataset.prepare_mask(mask, xyz.shape[:2])
        if mask is not None:
            valid &= mask[ys, xs]
        return points[valid], colors[valid]

    def _backproject_frame(
        self,
        frame,
        stride=1,
        depth_min=0.1,
        depth_max=10.0,
        mask=None,
    ):
        pose = self.load_pose(frame)
        if frame.depth_path is None:
            raise FileNotFoundError(f"depth is unavailable for {frame.stem}")
        depth = self.load_depth(frame.depth_path)
        mask = self._mask_for_frame(mask, frame)
        if self.image_type == "pano":
            points, ys, xs = self.backproject_pano(
                depth, stride, depth_min, depth_max, mask
            )
        else:
            points, ys, xs = self.backproject_regular(
                depth,
                self.intrinsics(pose),
                stride,
                depth_min,
                depth_max,
                mask,
            )
        return points, ys, xs, pose, depth

    def frame_cloud(
        self,
        frame,
        stride=4,
        depth_min=0.1,
        depth_max=8.0,
        mask=None,
        world_coordinates=True,
        from_global_xyz=False,
    ):
        rgb = self.load_rgb(frame.rgb_path)
        if from_global_xyz:
            if frame.xyz_path is None:
                raise FileNotFoundError(f"global_xyz is unavailable for {frame.stem}")
            xyz = self.load_global_xyz(frame.xyz_path)
            frame_mask = self._mask_for_frame(mask, frame)
            points, colors = self.points_from_global_xyz(xyz, rgb, stride, frame_mask)
            if world_coordinates:
                coordinates = "world"
            else:
                pose = self.load_pose(frame)
                world_to_camera = np.linalg.inv(self.camera_to_world_from_pose(pose))
                points = transform_points(points, world_to_camera)
                coordinates = "camera"
        else:
            points, ys, xs, pose, _ = self._backproject_frame(
                frame, stride, depth_min, depth_max, mask
            )
            colors = rgb[ys, xs]
            if world_coordinates:
                points = transform_points(points, self.camera_to_world_from_pose(pose))
            coordinates = "world" if world_coordinates else "camera"

        return PointCloud(
            points,
            colors,
            metadata={"frame": frame.stem, "coordinate_frame": coordinates},
        )

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
        source = "xyz_path" if from_global_xyz else "depth_path"
        frames = [frame for frame in frames if getattr(frame, source) is not None]
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
            cloud = self.frame_cloud(
                frame,
                stride,
                depth_min,
                depth_max,
                mask,
                world_coordinates,
                from_global_xyz,
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

    def point_map(
        self,
        room,
        frame_id,
        uuid=None,
        world_coordinates=False,
        mask=None,
    ):
        frame = self.get_frame(room, frame_id, uuid)
        points, ys, xs, pose, depth = self._backproject_frame(
            frame, depth_min=None, depth_max=None, mask=mask
        )
        if world_coordinates:
            points = transform_points(points, self.camera_to_world_from_pose(pose))
        result = np.zeros((*depth.shape, 3), dtype=np.float32)
        result[ys, xs] = points
        return result

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
