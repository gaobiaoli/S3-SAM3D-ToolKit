from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from .config import S3DIS_ROOT
from .io import read_xyzrgb_txt
from .models import BoundingBox3D, PointCloud
from .pointcloud import random_downsample, visualize_point_clouds

SEMANTIC_CLASSES = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
)

SEMANTIC_PALETTE = np.asarray(
    [
        [233, 229, 107],
        [95, 156, 196],
        [179, 116, 81],
        [241, 149, 131],
        [81, 163, 148],
        [77, 174, 84],
        [108, 135, 75],
        [79, 79, 76],
        [41, 49, 101],
        [223, 52, 52],
        [89, 47, 95],
        [81, 109, 114],
        [233, 233, 229],
    ],
    dtype=np.float32,
) / 255.0


def _class_from_instance(stem):
    head, separator, tail = stem.rpartition("_")
    return head.casefold() if separator and tail.isdigit() else stem.casefold()


def _semantic_colors(labels):
    labels = np.asarray(labels, dtype=np.int32)
    colors = np.full((len(labels), 3), 0.5, dtype=np.float32)
    valid = labels >= 0
    colors[valid] = SEMANTIC_PALETTE[labels[valid] % len(SEMANTIC_PALETTE)]
    return colors


def _instance_colors(labels, seed=42):
    labels = np.asarray(labels, dtype=np.int32)
    colors = np.full((len(labels), 3), 0.5, dtype=np.float32)
    valid = labels >= 0
    if valid.any():
        palette = np.random.default_rng(seed).random((int(labels[valid].max()) + 1, 3))
        colors[valid] = palette[labels[valid]]
    return colors


@dataclass(frozen=True)
class S3DISInstance:
    room: S3DISRoom = field(repr=False, compare=False)
    instance_id: int
    name: str
    class_name: str
    class_id: int
    path: Path

    @property
    def point_cloud(self):
        xyz, rgb = self.room.dataset._read_points(self.path)
        size = len(xyz)
        return PointCloud(
            xyz,
            np.clip(rgb / 255.0, 0.0, 1.0),
            semantic_labels=np.full(size, self.class_id, dtype=np.int32),
            instance_labels=np.full(size, self.instance_id, dtype=np.int32),
            metadata={
                "area": self.room.area,
                "room": self.room.name,
                "name": self.name,
                "class_name": self.class_name,
                "class_id": self.class_id,
                "instance_id": self.instance_id,
                "path": str(self.path),
            },
        )

    @cached_property
    def bbox(self):
        return BoundingBox3D.from_points(self.point_cloud.xyz)


@dataclass(frozen=True)
class S3DISRoom:
    dataset: S3DISDataset = field(repr=False, compare=False)
    area: str
    name: str
    path: Path
    point_path: Path
    annotation_dir: Path | None

    @property
    def key(self):
        return f"{self.area}/{self.name}"

    @cached_property
    def instances(self):
        if self.annotation_dir is None:
            return ()
        result = []
        for instance_id, path in enumerate(sorted(self.annotation_dir.glob("*.txt"))):
            class_name = _class_from_instance(path.stem)
            result.append(
                S3DISInstance(
                    room=self,
                    instance_id=instance_id,
                    name=path.stem,
                    class_name=class_name,
                    class_id=self.dataset.class_to_id.get(
                        class_name,
                        self.dataset.class_to_id["clutter"],
                    ),
                    path=path,
                )
            )
        return tuple(result)

    def instance(self, target):
        if isinstance(target, int) or str(target).isdigit():
            target = int(target)
            return next(item for item in self.instances if item.instance_id == target)
        query = str(target).casefold()
        return next(item for item in self.instances if item.name.casefold() == query)

    def _resolve_instances(self, values, ignore_missing=False):
        result = []
        for value in values:
            if isinstance(value, int) or str(value).isdigit():
                matches = [
                    item for item in self.instances if item.instance_id == int(value)
                ]
            else:
                query = str(value).casefold()
                matches = [
                    item
                    for item in self.instances
                    if item.name.casefold() == query
                    or item.class_name.casefold() == query
                ]
            if not matches and not ignore_missing:
                raise ValueError(f"instance or class not found: {value}")
            for item in matches:
                if item not in result:
                    result.append(item)
        return result

    def _full_point_cloud(self, labels=True):
        if labels and self.instances:
            clouds = [instance.point_cloud for instance in self.instances]
            return PointCloud(
                np.concatenate([cloud.xyz for cloud in clouds]),
                np.concatenate([cloud.rgb for cloud in clouds]),
                np.concatenate([cloud.semantic_labels for cloud in clouds]),
                np.concatenate([cloud.instance_labels for cloud in clouds]),
                metadata={
                    "area": self.area,
                    "room": self.name,
                    "instances": {
                        item.instance_id: {
                            "name": item.name,
                            "class_name": item.class_name,
                            "class_id": item.class_id,
                            "path": str(item.path),
                        }
                        for item in self.instances
                    },
                },
            )

        xyz, rgb = self.dataset._read_points(self.point_path)
        missing = np.full(len(xyz), -1, dtype=np.int32) if labels else None
        return PointCloud(
            xyz,
            np.clip(rgb / 255.0, 0.0, 1.0),
            semantic_labels=missing,
            instance_labels=None if missing is None else missing.copy(),
            metadata={"area": self.area, "room": self.name, "instances": {}},
        )

    def point_cloud(
        self,
        *,
        labels=True,
        include_classes: Iterable[str | int] | None = None,
        exclude_classes: Iterable[str | int] | None = None,
        exclude_instances: Iterable[str | int] | None = None,
        color_mode="rgb",
        ignore_missing_instances=False,
        max_points=None,
        seed=42,
    ):
        cloud = self._full_point_cloud(labels)
        visible = None
        if include_classes is not None:
            class_ids = [self.dataset._class_id(value) for value in include_classes]
            visible = np.isin(cloud.semantic_labels, class_ids)
        if exclude_classes is not None:
            class_ids = [self.dataset._class_id(value) for value in exclude_classes]
            selected = ~np.isin(cloud.semantic_labels, class_ids)
            visible = selected if visible is None else visible & selected
        if exclude_instances is not None:
            instance_ids = [
                item.instance_id
                for item in self._resolve_instances(
                    exclude_instances,
                    ignore_missing=ignore_missing_instances,
                )
            ]
            selected = ~np.isin(cloud.instance_labels, instance_ids)
            visible = selected if visible is None else visible & selected

        result = cloud if visible is None else cloud.select(visible)
        if color_mode == "semantic":
            result.rgb = _semantic_colors(result.semantic_labels)
        elif color_mode == "instance":
            result.rgb = _instance_colors(result.instance_labels, seed)
        elif color_mode != "rgb":
            raise ValueError("color_mode must be rgb, semantic, or instance")
        if max_points is not None:
            result = random_downsample(result, max_points, seed=seed)
        return result

    def visualize(
        self,
        *,
        color_mode="rgb",
        hidden_classes=None,
        hidden_instances=None,
        bbox_instances=None,
        meshes=(),
        ignore_missing_instances=True,
        max_points=None,
        seed=42,
        point_size=2.0,
        window_name=None,
        width=1280,
        height=800,
        show_coordinate_frame=False,
        get_parameters=False,
        set_parameters=None,
    ):
        cloud = self.point_cloud(
            exclude_classes=hidden_classes,
            exclude_instances=hidden_instances,
            color_mode=color_mode,
            ignore_missing_instances=ignore_missing_instances,
            max_points=max_points,
            seed=seed,
        )
        boxes = [
            instance.bbox
            for target in bbox_instances or ()
            for instance in self._resolve_instances(
                [target],
                ignore_missing=ignore_missing_instances,
            )
        ]
        visualize_point_clouds(
            [cloud, *meshes],
            bounding_boxes=boxes,
            window_name=window_name or f"S3DIS | {self.key} | {color_mode}",
            width=width,
            height=height,
            point_size=point_size,
            show_coordinate_frame=show_coordinate_frame,
            get_parameters=get_parameters,
            set_parameters=set_parameters,
        )
        return cloud


class S3DISDataset:
    """Discover S3DIS rooms and cache their source text files."""

    semantic_classes = SEMANTIC_CLASSES

    def __init__(self, root=None):
        root = S3DIS_ROOT if root is None else root
        self.root = Path(root).expanduser().resolve()
        self.class_to_id = {
            name: index for index, name in enumerate(self.semantic_classes)
        }
        self._txt_cache = {}
        self.rooms = self._scan_rooms()
        if not self.rooms:
            raise ValueError(
                f"no S3DIS rooms found under {self.root}; expected Area_*/room/room.txt"
            )

    def _scan_rooms(self):
        result = []
        for area_dir in sorted(self.root.iterdir()):
            if not area_dir.is_dir() or not area_dir.name.lower().startswith("area_"):
                continue
            for room_dir in sorted(area_dir.iterdir()):
                point_path = room_dir / f"{room_dir.name}.txt"
                if not room_dir.is_dir() or not point_path.is_file():
                    continue
                annotations = room_dir / "Annotations"
                result.append(
                    S3DISRoom(
                        dataset=self,
                        area=area_dir.name,
                        name=room_dir.name,
                        path=room_dir,
                        point_path=point_path,
                        annotation_dir=annotations if annotations.is_dir() else None,
                    )
                )
        return result

    def __len__(self):
        return len(self.rooms)

    def room(self, room):
        if isinstance(room, int):
            return self.rooms[room]
        query = str(room).replace("\\", "/").strip("/").casefold()
        exact = [item for item in self.rooms if item.key.casefold() == query]
        matches = exact or [item for item in self.rooms if item.name.casefold() == query]
        if len(matches) != 1:
            raise ValueError(f"room not found or ambiguous: {room}")
        return matches[0]

    def _class_id(self, value):
        return value if isinstance(value, int) else self.class_to_id[value.strip().casefold()]

    def clear_cache(self):
        self._txt_cache.clear()

    def _read_points(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        signature = stat.st_mtime_ns, stat.st_size
        cached = self._txt_cache.get(path)
        if cached is None or cached[0] != signature:
            xyz, rgb = read_xyzrgb_txt(path)
            self._txt_cache[path] = signature, xyz, rgb
        else:
            _, xyz, rgb = cached
        return xyz.copy(), rgb.copy()
