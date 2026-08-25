from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
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
        [233, 229, 107], [95, 156, 196], [179, 116, 81], [241, 149, 131],
        [81, 163, 148], [77, 174, 84], [108, 135, 75], [79, 79, 76],
        [41, 49, 101], [223, 52, 52], [89, 47, 95], [81, 109, 114],
        [233, 233, 229],
    ],
    dtype=np.float32,
) / 255.0


@dataclass(frozen=True)
class RoomInfo:
    area: str
    name: str
    path: Path
    point_path: Path
    annotation_dir: Path | None

    @property
    def key(self) -> str:
        return f"{self.area}/{self.name}"


class S3DISDataset:
    """Reader for the aligned S3DIS ``Area_*/room/Annotations`` release."""

    semantic_classes = SEMANTIC_CLASSES

    def __init__(self, root: str | Path | None = None):
        root = S3DIS_ROOT if root is None else root
        self.root = Path(root).expanduser().resolve()
        self.class_to_id = {name: index for index, name in enumerate(self.semantic_classes)}
        self._txt_cache: dict[
            Path, tuple[tuple[int, int], np.ndarray, np.ndarray]
        ] = {}
        self.rooms = self._scan_rooms()
        if not self.rooms:
            raise ValueError(
                f"no S3DIS rooms found under {self.root}; expected Area_*/room/room.txt"
            )

    def _scan_rooms(self) -> list[RoomInfo]:
        result: list[RoomInfo] = []
        for area_dir in sorted(self.root.iterdir()):
            if not area_dir.is_dir() or not area_dir.name.lower().startswith("area_"):
                continue
            for room_dir in sorted(area_dir.iterdir()):
                if not room_dir.is_dir():
                    continue
                point_path = room_dir / f"{room_dir.name}.txt"
                if not point_path.is_file():
                    continue
                annotations = room_dir / "Annotations"
                result.append(
                    RoomInfo(
                        area=area_dir.name,
                        name=room_dir.name,
                        path=room_dir,
                        point_path=point_path,
                        annotation_dir=annotations if annotations.is_dir() else None,
                    )
                )
        return result

    def __len__(self) -> int:
        return len(self.rooms)

    def list_rooms(self) -> list[str]:
        return [room.key for room in self.rooms]

    def resolve_room(self, room: str | int) -> RoomInfo:
        if isinstance(room, int):
            try:
                return self.rooms[room]
            except IndexError as exc:
                raise IndexError(f"room index out of range: {room}") from exc
        query = room.replace("\\", "/").strip("/").casefold()
        exact = [item for item in self.rooms if item.key.casefold() == query]
        matches = exact or [item for item in self.rooms if item.name.casefold() == query]
        if not matches:
            raise ValueError(f"room not found: {room}")
        if len(matches) > 1:
            names = ", ".join(item.key for item in matches[:10])
            raise ValueError(f"ambiguous room '{room}'; use Area_x/room_name. Matches: {names}")
        return matches[0]

    def class_id(self, value: str | int) -> int:
        if isinstance(value, int):
            return value
        return self.class_to_id[value.strip().casefold()]

    def clear_cache(self) -> None:
        """Discard point arrays cached by this dataset instance."""
        self._txt_cache.clear()

    def _read_points(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Read and cache one text file until its size or mtime changes."""
        path = path.resolve()
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._txt_cache.get(path)
        if cached is None or cached[0] != signature:
            xyz, rgb = read_xyzrgb_txt(path)
            self._txt_cache[path] = (signature, xyz, rgb)
        else:
            _, xyz, rgb = cached

        # Callers may freely mutate a returned PointCloud without corrupting
        # future cache hits.
        return xyz.copy(), rgb.copy()

    @staticmethod
    def _class_from_instance(stem: str) -> str:
        head, separator, tail = stem.rpartition("_")
        return head.casefold() if separator and tail.isdigit() else stem.casefold()

    def load_room(
        self,
        room: str | int,
        *,
        labels: bool = True,
    ) -> PointCloud:
        info = self.resolve_room(room)
        if not labels or info.annotation_dir is None:
            xyz, rgb = self._read_points(info.point_path)
            rgb = np.clip(rgb / 255.0, 0.0, 1.0)
            missing = np.full(len(xyz), -1, dtype=np.int32) if labels else None
            return PointCloud(
                xyz,
                rgb,
                semantic_labels=missing,
                instance_labels=None if missing is None else missing.copy(),
                metadata={"area": info.area, "room": info.name, "instances": {}},
            )

        files = sorted(info.annotation_dir.glob("*.txt"))
        if not files:
            return self.load_room(room, labels=False)

        xyz_parts: list[np.ndarray] = []
        rgb_parts: list[np.ndarray] = []
        semantic_parts: list[np.ndarray] = []
        instance_parts: list[np.ndarray] = []
        instances: dict[int, dict[str, object]] = {}
        for instance_id, annotation_path in enumerate(files):
            xyz, rgb = self._read_points(annotation_path)
            class_name = self._class_from_instance(annotation_path.stem)
            class_id = self.class_to_id.get(class_name, self.class_to_id["clutter"])
            rgb = np.clip(rgb / 255.0, 0.0, 1.0)
            xyz_parts.append(xyz)
            rgb_parts.append(rgb)
            semantic_parts.append(np.full(len(xyz), class_id, dtype=np.int32))
            instance_parts.append(np.full(len(xyz), instance_id, dtype=np.int32))
            instances[instance_id] = {
                "name": annotation_path.stem,
                "class_name": class_name,
                "class_id": class_id,
                "path": str(annotation_path),
            }
        return PointCloud(
            np.concatenate(xyz_parts),
            np.concatenate(rgb_parts),
            np.concatenate(semantic_parts),
            np.concatenate(instance_parts),
            metadata={"area": info.area, "room": info.name, "instances": instances},
        )

    def semantic_colors(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.int32)
        colors = np.full((len(labels), 3), 0.5, dtype=np.float32)
        valid = labels >= 0
        colors[valid] = SEMANTIC_PALETTE[labels[valid] % len(SEMANTIC_PALETTE)]
        return colors

    @staticmethod
    def instance_colors(labels: np.ndarray, seed: int = 42) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.int32)
        colors = np.full((len(labels), 3), 0.5, dtype=np.float32)
        valid = labels >= 0
        if valid.any():
            lut = np.random.default_rng(seed).random((int(labels[valid].max()) + 1, 3))
            colors[valid] = lut[labels[valid]]
        return colors

    def filter_room(
        self,
        room: str | int,
        *,
        include_classes: Iterable[str | int] | None = None,
        exclude_classes: Iterable[str | int] | None = None,
        exclude_instances: Iterable[str | int] | None = None,
        color_mode: str = "rgb",
        ignore_missing_instances: bool = False,
    ) -> PointCloud:
        cloud = self.load_room(room, labels=True)
        return self._filter_cloud(
            cloud,
            include_classes=include_classes,
            exclude_classes=exclude_classes,
            exclude_instances=exclude_instances,
            color_mode=color_mode,
            ignore_missing_instances=ignore_missing_instances,
        )

    def _filter_cloud(
        self,
        cloud: PointCloud,
        *,
        include_classes=None,
        exclude_classes=None,
        exclude_instances=None,
        color_mode="rgb",
        ignore_missing_instances=False,
    ) -> PointCloud:
        visible = np.ones(len(cloud.xyz), dtype=bool)
        if include_classes is not None:
            ids = [self.class_id(value) for value in include_classes]
            visible &= np.isin(cloud.semantic_labels, ids)
        if exclude_classes is not None:
            ids = [self.class_id(value) for value in exclude_classes]
            visible &= ~np.isin(cloud.semantic_labels, ids)
        if exclude_instances is not None:
            ids = self.resolve_instances(
                cloud, exclude_instances, ignore_missing=ignore_missing_instances
            )
            visible &= ~np.isin(cloud.instance_labels, ids)
        result = cloud.select(visible)
        if color_mode == "semantic":
            result.rgb = self.semantic_colors(result.semantic_labels)
        elif color_mode == "instance":
            result.rgb = self.instance_colors(result.instance_labels)
        elif color_mode != "rgb":
            raise ValueError("color_mode must be rgb, semantic, or instance")
        return result

    def resolve_instances(
        self,
        cloud: PointCloud,
        values: Iterable[str | int],
        *,
        ignore_missing: bool = False,
    ) -> list[int]:
        instances = cloud.metadata.get("instances", {})
        result: set[int] = set()
        for value in values:
            if isinstance(value, int) or str(value).isdigit():
                result.add(int(value))
                continue
            query = str(value).casefold()
            matches = [
                int(instance_id)
                for instance_id, item in instances.items()
                if str(item["name"]).casefold() == query
                or str(item["class_name"]).casefold() == query
            ]
            if not matches:
                if ignore_missing:
                    continue
                raise ValueError(f"instance or class not found: {value}")
            result.update(matches)
        return sorted(result)

    def get_visualization_cloud(
        self,
        room: str | int,
        *,
        color_mode: str = "rgb",
        hidden_classes: Iterable[str | int] | None = None,
        hidden_instances: Iterable[str | int] | None = None,
        ignore_missing_instances: bool = True,
        max_points: int | None = None,
        seed: int = 42,
    ) -> PointCloud:
        """Prepare a filtered and colored cloud for visualization."""
        cloud = self.filter_room(
            room,
            exclude_classes=hidden_classes,
            exclude_instances=hidden_instances,
            color_mode=color_mode,
            ignore_missing_instances=ignore_missing_instances,
        )
        if max_points is not None:
            cloud = random_downsample(cloud, max_points, seed=seed)
        return cloud

    def visualize_room(
        self,
        room: str | int,
        *,
        color_mode: str = "rgb",
        hidden_classes: Iterable[str | int] | None = None,
        hidden_instances: Iterable[str | int] | None = None,
        bbox_instances: Iterable[str | int] | None = None,
        meshes: Iterable[object] = (),
        ignore_missing_instances: bool = True,
        max_points: int | None = None,
        seed: int = 42,
        point_size: float = 2.0,
        window_name: str | None = None,
        width: int = 1280,
        height: int = 800,
        show_coordinate_frame: bool = False,
        get_parameters: bool = False,
        set_parameters: tuple[list[list[float]], list[list[float]]] | None = None,
    ) -> None:
        """Open an interactive Open3D window for a filtered S3DIS room."""
        info = self.resolve_room(room)
        full_cloud = self.load_room(room, labels=True)
        cloud = self._filter_cloud(
            full_cloud,
            exclude_classes=hidden_classes,
            exclude_instances=hidden_instances,
            color_mode=color_mode,
            ignore_missing_instances=ignore_missing_instances,
        )
        if max_points is not None:
            cloud = random_downsample(cloud, max_points, seed=seed)

        boxes: list[BoundingBox3D] = []
        for target in bbox_instances or ():
            ids = self.resolve_instances(
                full_cloud, [target], ignore_missing=ignore_missing_instances
            )
            if ids:
                points = full_cloud.xyz[np.isin(full_cloud.instance_labels, ids)]
                boxes.append(BoundingBox3D.from_points(points))
        visualize_point_clouds(
            [cloud, *meshes],
            bounding_boxes=boxes,
            window_name=window_name or f"S3DIS | {info.key} | {color_mode}",
            width=width,
            height=height,
            point_size=point_size,
            show_coordinate_frame=show_coordinate_frame,
            get_parameters=get_parameters,
            set_parameters=set_parameters,
        )

    def object_cloud(self, room: str | int, target: str | int) -> PointCloud:
        cloud = self.load_room(room, labels=True)
        ids = self.resolve_instances(cloud, [target])
        selected = cloud.select(np.isin(cloud.instance_labels, ids))
        if not len(selected.xyz):
            raise ValueError(f"object has no points: {target}")
        selected.metadata["target_instance_ids"] = ids
        selected.metadata["bbox"] = BoundingBox3D.from_points(selected.xyz).as_dict()
        return selected

    def get_region_point_cloud(
        self,
        region: str,
        *,
        area: str = "Area_1",
        include_classes: Iterable[str | int] | None = None,
    ) -> PointCloud:
        """Return a BIMSync region's matching S3DIS room point cloud."""
        room = region if "/" in region.replace("\\", "/") else f"{area}/{region}"
        if include_classes is None:
            return self.load_room(room)
        return self.filter_room(room, include_classes=include_classes)

    def object_clouds(
        self,
        room: str | int,
        include_classes: Iterable[str | int] | None = None,
    ) -> list[PointCloud]:
        """Split a room into its instance point clouds with one room load."""
        cloud = self.load_room(room, labels=True)
        class_ids = (
            None
            if include_classes is None
            else {self.class_id(value) for value in include_classes}
        )
        result = []
        for instance_id, info in cloud.metadata["instances"].items():
            if class_ids is not None and int(info["class_id"]) not in class_ids:
                continue
            item = cloud.select(cloud.instance_labels == int(instance_id))
            item.metadata = {
                "area": cloud.metadata["area"],
                "room": cloud.metadata["room"],
                **info,
                "instance_id": int(instance_id),
            }
            result.append(item)
        return sorted(result, key=lambda item: str(item.metadata["name"]))

    def object_bboxes(
        self, room: str | int, include_classes: Iterable[str | int] | None = None
    ) -> list[dict[str, object]]:
        cloud = self.load_room(room, labels=True)
        class_ids = None if include_classes is None else {self.class_id(v) for v in include_classes}
        instances = cloud.metadata["instances"]
        result: list[dict[str, object]] = []
        for instance_id in np.unique(cloud.instance_labels[cloud.instance_labels >= 0]):
            item = instances[int(instance_id)]
            if class_ids is not None and int(item["class_id"]) not in class_ids:
                continue
            points = cloud.xyz[cloud.instance_labels == instance_id]
            result.append({**item, "instance_id": int(instance_id), "bbox": BoundingBox3D.from_points(points).as_dict()})
        return result
