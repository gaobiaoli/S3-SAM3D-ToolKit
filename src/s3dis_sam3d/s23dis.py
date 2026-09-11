from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image

from .config import S23DIS_SEMANTIC_LABELS_PATH, s23dis_area
from .frames import RGBDFrame
from .models import PointCloud
from .pointcloud import transform_points, visualize_point_clouds, voxel_downsample
from .s3dis import SEMANTIC_CLASSES
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
S23DIS_INVALID_SEMANTIC = 0x0D0D0D
S23DIS_SEMANTIC_CLASSES = SEMANTIC_CLASSES
S23DIS_CLASS_TO_ID = {
    name: class_id for class_id, name in enumerate(S23DIS_SEMANTIC_CLASSES)
}

FRAME_PATTERN = re.compile(
    r"^camera_(?P<uuid>[0-9a-fA-F]+)_(?P<scene>.+?)_frame_"
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
    item["scene_name"] = item["scene"]
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


def _concatenate_optional_labels(clouds, attribute):
    labels = [getattr(cloud, attribute) for cloud in clouds]
    if not any(label is not None for label in labels):
        return None
    return np.concatenate(
        [
            label
            if label is not None
            else np.full(len(cloud.xyz), -1, dtype=np.int32)
            for cloud, label in zip(clouds, labels)
        ]
    )


@dataclass(frozen=True)
class S23Frame(RGBDFrame):
    depth_scale = S23DIS_DEPTH_SCALE
    invalid_depth_value = S23DIS_INVALID_DEPTH

    stem: str
    scene: str
    frame_id: int
    uuid: str
    pose_path: Path
    xyz_path: Path | None = None
    semantic_path: Path | None = None
    projection_type: str = "regular"
    semantic_instance_names: tuple[str, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    semantic_class_lookup: tuple[int, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    @cached_property
    def pose(self):
        return json.loads(self.pose_path.read_text("utf-8"))

    @property
    def has_xyz(self):
        return self.xyz_path is not None

    @property
    def has_semantic(self):
        return self.semantic_path is not None

    @property
    def xyz(self):
        if self.xyz_path is None:
            raise FileNotFoundError(f"global_xyz is unavailable for {self.stem}")
        return _read_global_xyz(self.xyz_path)

    @property
    def instance_labels(self):
        """Return the per-pixel global instance indices encoded by 2D-3D-S."""
        if self.semantic_path is None:
            raise FileNotFoundError(f"semantic image is unavailable for {self.stem}")
        with Image.open(self.semantic_path) as image:
            encoded = np.asarray(image.convert("RGB"), dtype=np.int32)
        labels = (
            (encoded[..., 0] << 16)
            | (encoded[..., 1] << 8)
            | encoded[..., 2]
        )
        labels[labels == S23DIS_INVALID_SEMANTIC] = -1
        return labels

    @property
    def semantic(self):
        """Alias for the decoded instance-index image."""
        return self.instance_labels

    @property
    def semantic_labels(self):
        """Return per-pixel S3DIS class IDs, with unknown pixels set to ``-1``."""
        if not self.semantic_class_lookup:
            raise FileNotFoundError(
                "semantic_labels.json is required to convert instance indices "
                f"to semantic classes for {self.stem}"
            )
        return self._classes_from_instances(self.instance_labels)

    def _classes_from_instances(self, instances):
        classes = np.full(instances.shape, -1, dtype=np.int32)
        valid = (instances >= 0) & (instances < len(self.semantic_class_lookup))
        classes[valid] = np.asarray(self.semantic_class_lookup, dtype=np.int32)[
            instances[valid]
        ]
        return classes

    @property
    def semantic_categories(self):
        """Return the semantic category names visible in this frame."""
        class_ids = set(np.unique(self.semantic_labels).tolist())
        return tuple(
            name
            for class_id, name in enumerate(S23DIS_SEMANTIC_CLASSES)
            if class_id in class_ids
        )

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
            points, ys, xs, _ = self._backproject(
                stride=stride,
                depth_min=depth_min,
                depth_max=depth_max,
                mask=mask,
            )
        else:
            points, colors, ys, xs = points_from_global_xyz(
                self.xyz,
                self.rgb,
                stride,
                _mask_for_frame(mask, self),
                return_indices=True,
            )

        if from_global_xyz:
            if not world_coordinates:
                points = transform_points(points, self.world_to_camera)
        elif world_coordinates:
            points = transform_points(points, self.camera_to_world)

        if not from_global_xyz:
            colors = self.rgb[ys, xs]
        semantic_labels = None
        instance_labels = None
        if self.has_semantic:
            instance_labels = self.instance_labels[ys, xs]
            if self.semantic_class_lookup:
                semantic_labels = self._classes_from_instances(instance_labels)

        coordinates = "world" if world_coordinates else "camera"

        return PointCloud(
            points,
            colors,
            semantic_labels=semantic_labels,
            instance_labels=instance_labels,
            metadata=self._point_cloud_metadata(coordinates),
        )

    def _point_cloud_metadata(self, coordinate_frame):
        metadata = {"frame": self.stem, "coordinate_frame": coordinate_frame}
        if self.semantic_path is not None:
            metadata["semantic_path"] = str(self.semantic_path)
        if self.semantic_class_lookup:
            metadata["label_names"] = S23DIS_SEMANTIC_CLASSES
        if self.semantic_instance_names:
            metadata["instance_names"] = self.semantic_instance_names
        return metadata


@dataclass(frozen=True)
class S23Scene:
    """One 2D-3D-S scene and its regular or panorama frames."""

    dataset: S23Dataset = field(repr=False, compare=False)
    area: str
    name: str
    frames: tuple[S23Frame, ...]

    @property
    def key(self):
        return f"{self.area}/{self.name}"

    @property
    def scene_id(self):
        return self.name

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
            raise ValueError(f"no usable frames for scene: {self.key}")

        cloud = PointCloud(
            np.concatenate([cloud.xyz for cloud in clouds]),
            np.concatenate([cloud.rgb for cloud in clouds]),
            _concatenate_optional_labels(clouds, "semantic_labels"),
            _concatenate_optional_labels(clouds, "instance_labels"),
            metadata={
                "area": self.area,
                "scene": self.name,
                "frame_count": len(clouds),
                "coordinate_frame": clouds[0].metadata["coordinate_frame"],
                "label_names": S23DIS_SEMANTIC_CLASSES,
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
        return f"S23Scene(key={self.key!r}, frames={len(self)})"


class S23Dataset:
    """Discover 2D-3D-S scenes and index their camera frames."""

    def __init__(
        self,
        area_path=None,
        projection_type="regular",
        area="Area_1",
        default_uuid="first",
        semantic_labels_path=None,
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
        self.semantic_dir = self.data_dir / "semantic"
        self.semantic_labels_path = self._resolve_semantic_labels_path(
            semantic_labels_path
        )
        self.semantic_instance_names = self._load_semantic_instance_names()
        self.semantic_class_lookup = tuple(
            S23DIS_CLASS_TO_ID.get(name.split("_", 1)[0].casefold(), -1)
            for name in self.semantic_instance_names
        )
        self.frames = tuple(self._index_frames(projection_type))
        grouped = {}
        for frame in self.frames:
            grouped.setdefault(frame.scene, []).append(frame)
        for frames in grouped.values():
            frames.sort(key=lambda frame: (frame.frame_id, frame.uuid))
        self.scenes = tuple(
            S23Scene(self, self.area, name, tuple(frames))
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
            semantic_path = self.semantic_dir / f"{stem}_semantic.png"
            if rgb_path.exists() and (depth_path.exists() or xyz_path.exists()):
                frames.append(
                    S23Frame(
                        stem=stem,
                        scene=metadata["scene"],
                        frame_id=metadata["frame_id"],
                        uuid=metadata["uuid"],
                        pose_path=pose_path,
                        rgb_path=rgb_path,
                        depth_path=depth_path if depth_path.exists() else None,
                        xyz_path=xyz_path if xyz_path.exists() else None,
                        semantic_path=(
                            semantic_path if semantic_path.exists() else None
                        ),
                        projection_type=projection_type,
                        semantic_instance_names=self.semantic_instance_names,
                        semantic_class_lookup=self.semantic_class_lookup,
                    )
                )
        return frames

    def _resolve_semantic_labels_path(self, path):
        if path is not None:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"semantic label metadata not found: {path}")
            return path

        candidates = (
            self.area_path.parent / "assets" / "semantic_labels.json",
            self.area_path / "assets" / "semantic_labels.json",
            self.area_path.parent / "semantic_labels.json",
            self.area_path / "semantic_labels.json",
            S23DIS_SEMANTIC_LABELS_PATH,
        )
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    def _load_semantic_instance_names(self):
        if self.semantic_labels_path is None:
            return ()
        values = json.loads(self.semantic_labels_path.read_text("utf-8"))
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(
                "semantic_labels.json must contain a JSON list of label names"
            )
        return tuple(values)

    @property
    def scene_ids(self):
        return [scene.scene_id for scene in self.scenes]

    def list_scenes(self):
        return self.scene_ids

    def get_scene(self, scene_id):
        if isinstance(scene_id, S23Scene):
            if scene_id.dataset is self:
                return scene_id
            raise ValueError("scene belongs to another S23Dataset")
        if isinstance(scene_id, int):
            return self.scenes[scene_id]
        query = str(scene_id).replace("\\", "/").strip("/").casefold()
        exact = [item for item in self.scenes if item.key.casefold() == query]
        matches = exact or [
            item for item in self.scenes if item.scene_id.casefold() == query
        ]
        if not matches:
            raise KeyError(f"Unknown 2D-3D-S scene: {scene_id}")
        if len(matches) > 1:
            raise ValueError(f"scene is ambiguous: {scene_id}")
        return matches[0]

    def iter_scenes(self):
        return iter(self)

    def scene_frames(self, scene_id):
        return list(self.get_scene(scene_id).frames)

    def list_uuids(self, scene_id=None, frame_id=None):
        """List UUIDs in the Area, one scene, or one scene/frame."""
        if scene_id is not None:
            return self.get_scene(scene_id).list_uuids(frame_id)
        frames = self.frames
        if frame_id is not None:
            frames = tuple(frame for frame in frames if frame.frame_id == frame_id)
        return sorted({frame.uuid for frame in frames})

    def get_frame(self, scene_id=None, frame_id=None, uuid=None):
        if scene_id is None or frame_id is None:
            raise TypeError("scene_id and frame_id are required")
        return self.get_scene(scene_id).get_frame(frame_id, uuid)

    def __len__(self):
        return len(self.scenes)

    def __iter__(self):
        return iter(self.scenes)

    def __getitem__(self, index):
        return self.get_scene(index)

    def __repr__(self):
        return (
            f"S23Dataset(area={self.area!r}, projection_type={self.projection_type!r}, "
            f"scenes={len(self)})"
        )
