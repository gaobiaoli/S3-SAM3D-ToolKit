"""ScanNet 25k RGB-D frames and labeled scene meshes.

The parser follows the same frame/scene/dataset protocol as :mod:`s23dis` so
that geometry consumers such as :class:`s3dis_sam3d.syncbim.SyncBIMScene` can
operate on either dataset without dataset-specific branches.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image

from .frames import RGBDFrame
from .models import PointCloud
from .pointcloud import transform_points, visualize_point_clouds, voxel_downsample
from .registration import sample_labeled_mesh, stable_seed
from .utils import backproject_regular, project_pinhole_points

SCANNET_DEPTH_SCALE = 1000.0
SCANNET_INVALID_DEPTH = 0

# ScanNet's released vertex and 2D labels use the NYU40 IDs directly.  Keeping
# the unannotated entry at index zero means ``semantic_classes[label]`` works
# without an additional remapping table.
SCANNET_SEMANTIC_CLASSES = (
    "unannotated",
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor mat",
    "clothes",
    "ceiling",
    "books",
    "refrigerator",
    "television",
    "paper",
    "towel",
    "shower curtain",
    "box",
    "whiteboard",
    "person",
    "night stand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    "otherstructure",
    "otherfurniture",
    "otherprop",
)
SCANNET_CLASS_TO_ID = {name: class_id for class_id, name in enumerate(SCANNET_SEMANTIC_CLASSES)}


def _read_matrix(path, shape=(4, 4)):
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"expected a finite {shape[0]}x{shape[1]} matrix: {path}")
    return matrix


def read_scene_metadata(path):
    """Read the key/value metadata from one ScanNet ``scene*.txt`` file."""
    result = {}
    path = Path(path)
    if not path.is_file():
        return result
    for raw_line in path.read_text("utf-8").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _axis_alignment(metadata, path):
    value = metadata.get("axisAlignment")
    if value is None:
        return np.eye(4, dtype=np.float64)
    values = np.fromstring(value, sep=" ", dtype=np.float64)
    if values.size != 16 or not np.isfinite(values).all():
        raise ValueError(f"invalid axisAlignment in {path}")
    return values.reshape(4, 4)


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
            label if label is not None else np.full(len(cloud.xyz), -1, dtype=np.int32)
            for cloud, label in zip(clouds, labels)
        ]
    )


@dataclass(frozen=True)
class ScanNetFrame(RGBDFrame):
    """One calibrated frame from ``scannet_frames_25k``."""

    depth_scale = SCANNET_DEPTH_SCALE
    invalid_depth_value = SCANNET_INVALID_DEPTH

    stem: str
    scene: str
    frame_id: int
    pose_path: Path
    depth_intrinsics_path: Path
    color_intrinsics_path: Path | None = None
    semantic_path: Path | None = None
    instance_path: Path | None = None
    scene_transform_values: tuple[float, ...] = field(
        default=(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        repr=False,
        compare=False,
    )

    @cached_property
    def pose(self):
        """The raw ScanNet camera-to-world pose before axis alignment."""
        return _read_matrix(self.pose_path).astype(np.float32)

    @cached_property
    def scene_transform(self):
        return np.asarray(self.scene_transform_values, dtype=np.float64).reshape(4, 4)

    @cached_property
    def camera_to_world(self):
        return (self.scene_transform @ self.pose).astype(np.float32)

    @cached_property
    def intrinsics(self):
        """Depth-camera intrinsics used by depth back-projection and rendering."""
        return _read_matrix(self.depth_intrinsics_path)[:3, :3].astype(np.float32)

    @cached_property
    def color_intrinsics(self):
        path = self.color_intrinsics_path or self.depth_intrinsics_path
        return _read_matrix(path)[:3, :3].astype(np.float32)

    @property
    def has_semantic(self):
        return self.semantic_path is not None

    @property
    def has_instance(self):
        return self.instance_path is not None

    def _read_rgb(self):
        """Return color resized to the depth grid for point-cloud association."""
        with Image.open(self.rgb_path) as image:
            image = image.convert("RGB")
            height, width = self.image_shape
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.float32)

    @property
    def semantic_labels(self):
        if self.semantic_path is None:
            raise FileNotFoundError(f"semantic labels are unavailable for {self.scene}/{self.stem}")
        return self._read_label_image(self.semantic_path)

    @property
    def semantic(self):
        return self.semantic_labels

    @property
    def instance_labels(self):
        """Return the released ``nyu40id * 1000 + instance_index`` IDs.

        Walls/floors/ceilings have an instance index of zero; these image IDs
        are frame-local and are not the aggregation IDs of the 3D mesh.
        """
        if self.instance_path is None:
            raise FileNotFoundError(f"instance labels are unavailable for {self.scene}/{self.stem}")
        return self._read_label_image(self.instance_path)

    @property
    def instance_indices(self):
        return self.instance_labels % 1000

    def _read_label_image(self, path):
        # Released 25k label/instance images use the color grid (1296x968),
        # whereas depth uses 640x480.  Never bilinearly interpolate label IDs.
        with Image.open(path) as image:
            height, width = self.image_shape
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.NEAREST)
            return np.asarray(image, dtype=np.int32)

    @property
    def semantic_categories(self):
        class_ids = set(np.unique(self.semantic_labels).tolist())
        return tuple(
            name for class_id, name in enumerate(SCANNET_SEMANTIC_CLASSES) if class_id in class_ids
        )

    def project_camera_points(self, points):
        return project_pinhole_points(points, self.intrinsics)

    def _backproject(
        self,
        stride=1,
        depth_min=0.1,
        depth_max=10.0,
        mask=None,
    ):
        depth = self.depth
        points, ys, xs = backproject_regular(
            depth,
            self.intrinsics,
            stride,
            depth_min,
            depth_max,
            _mask_for_frame(mask, self),
        )
        return points, ys, xs, depth

    def point_cloud(
        self,
        *,
        stride=4,
        depth_min=0.1,
        depth_max=8.0,
        world_coordinates=True,
        mask=None,
    ):
        points, ys, xs, _ = self._backproject(
            stride=stride,
            depth_min=depth_min,
            depth_max=depth_max,
            mask=mask,
        )
        if world_coordinates:
            points = transform_points(points, self.camera_to_world)
        return PointCloud(
            xyz=points,
            rgb=self.rgb[ys, xs],
            semantic_labels=(self.semantic_labels[ys, xs] if self.has_semantic else None),
            instance_labels=(self.instance_labels[ys, xs] if self.has_instance else None),
            metadata=self._point_cloud_metadata("world" if world_coordinates else "camera"),
        )

    def _point_cloud_metadata(self, coordinate_frame):
        metadata = {
            "dataset": "ScanNet",
            "scene": self.scene,
            "frame": self.stem,
            "coordinate_frame": coordinate_frame,
            "axis_aligned": not np.allclose(self.scene_transform, np.eye(4)),
        }
        if self.has_semantic:
            metadata["label_names"] = SCANNET_SEMANTIC_CLASSES
            metadata["semantic_path"] = str(self.semantic_path)
        if self.has_instance:
            metadata["instance_path"] = str(self.instance_path)
        return metadata


@dataclass(frozen=True)
class ScanNetMeshScene:
    """One labeled ScanNet mesh exposed through the Stanford mesh protocol."""

    dataset: ScanNetSemanticMesh = field(repr=False, compare=False)
    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    face_labels: np.ndarray
    face_instances: np.ndarray
    instance_records: dict[int, dict] = field(repr=False, compare=False)

    @property
    def key(self):
        return self.name

    def point_cloud(self, include_classes=None, sample_points=None):
        triangles = self.triangles
        labels = self.face_labels
        instances = self.face_instances
        if include_classes is not None:
            keep_ids = [
                SCANNET_CLASS_TO_ID[str(name).casefold()]
                for name in include_classes
                if str(name).casefold() in SCANNET_CLASS_TO_ID
            ]
            keep = np.isin(labels, keep_ids)
            triangles, labels, instances = triangles[keep], labels[keep], instances[keep]
        if not len(triangles):
            raise ValueError(f"no selected labeled faces in {self.key}")

        point_count = self.dataset.sample_points if sample_points is None else int(sample_points)
        if point_count < 1:
            raise ValueError("sample_points must be positive")
        points, sampled_labels = sample_labeled_mesh(
            self.vertices,
            triangles,
            np.column_stack((labels, instances)),
            point_count,
            stable_seed(self.dataset.seed, self.name, "scannet"),
        )
        visible_instances = set(np.unique(sampled_labels[:, 1]).tolist())
        return PointCloud(
            points,
            semantic_labels=sampled_labels[:, 0],
            instance_labels=sampled_labels[:, 1],
            metadata={
                "dataset": "ScanNet",
                "scene": self.name,
                "label_names": SCANNET_SEMANTIC_CLASSES,
                "instances": {
                    instance_id: dict(record)
                    for instance_id, record in self.instance_records.items()
                    if instance_id in visible_instances
                },
            },
        )


_PLY_SCALAR_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def _read_labeled_ply(path):
    """Read the binary little-endian triangle PLY used by ScanNet releases."""
    path = Path(path)
    vertex_count = face_count = None
    vertex_properties = []
    face_list_property = None
    element = None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"unterminated PLY header: {path}")
            line = raw_line.decode("ascii").strip()
            fields = line.split()
            if not fields or fields[0] in {"comment", "obj_info"}:
                continue
            if fields[:2] == ["format", "binary_little_endian"]:
                continue
            if fields[0] == "format":
                raise ValueError(f"only binary_little_endian PLY is supported: {path}")
            if fields[0] == "element":
                element = fields[1]
                if element == "vertex":
                    vertex_count = int(fields[2])
                elif element == "face":
                    face_count = int(fields[2])
            elif fields[0] == "property" and element == "vertex":
                if fields[1] == "list":
                    raise ValueError(f"list-valued vertex property is unsupported: {path}")
                vertex_properties.append((fields[2], _PLY_SCALAR_TYPES[fields[1]]))
            elif fields[:2] == ["property", "list"] and element == "face":
                face_list_property = (fields[2], fields[3], fields[4])
            elif fields[0] == "end_header":
                break

        if vertex_count is None or face_count is None or face_list_property is None:
            raise ValueError(f"missing vertex or face declaration in {path}")
        names = {name for name, _ in vertex_properties}
        if not {"x", "y", "z", "label"}.issubset(names):
            raise ValueError(f"ScanNet labeled PLY properties are incomplete: {path}")

        vertex_data = np.fromfile(
            handle,
            dtype=np.dtype(vertex_properties),
            count=vertex_count,
        )
        if len(vertex_data) != vertex_count:
            raise ValueError(f"truncated PLY vertex data: {path}")

        count_type, index_type, _ = face_list_property
        face_dtype = np.dtype(
            [
                ("count", _PLY_SCALAR_TYPES[count_type]),
                ("indices", _PLY_SCALAR_TYPES[index_type], 3),
            ]
        )
        face_data = np.fromfile(handle, dtype=face_dtype, count=face_count)
        if len(face_data) != face_count or np.any(face_data["count"] != 3):
            raise ValueError(f"ScanNet PLY must contain only triangular faces: {path}")

    vertices = np.column_stack((vertex_data["x"], vertex_data["y"], vertex_data["z"])).astype(
        np.float32
    )
    triangles = np.asarray(face_data["indices"], dtype=np.int32)
    vertex_labels = np.asarray(vertex_data["label"], dtype=np.int32)
    return vertices, triangles, vertex_labels


def _majority_of_three(values, invalid=-1):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("expected an (N, 3) label array")
    a, b, c = values.T
    result = np.full(len(values), invalid, dtype=np.int32)
    result[(a == b) & (a != invalid)] = a[(a == b) & (a != invalid)]
    mask = (result == invalid) & (a == c) & (a != invalid)
    result[mask] = a[mask]
    mask = (result == invalid) & (b == c) & (b != invalid)
    result[mask] = b[mask]
    for candidate in (a, b, c):
        mask = (result == invalid) & (candidate != invalid)
        result[mask] = candidate[mask]
    return result


class ScanNetSemanticMesh:
    """Lazy reader for the labeled meshes under ScanNet's ``scans`` folder."""

    semantic_classes = SCANNET_SEMANTIC_CLASSES

    def __init__(self, scans_path, transforms=None, sample_points=8_000, seed=20260913):
        self.scans_path = Path(scans_path).expanduser().resolve()
        self.transforms = {} if transforms is None else dict(transforms)
        self.sample_points = int(sample_points)
        self.seed = int(seed)
        # Full ScanNet has 1513 meshes: loading a scene must not keep every
        # previously visited mesh resident during dataset preparation.
        self._rooms = OrderedDict()

    @property
    def scene_ids(self):
        return sorted(path.name for path in self.scans_path.glob("scene*") if path.is_dir())

    def _read_instances(self, scene_dir, scene_name, vertex_labels, triangles):
        seg_path = scene_dir / f"{scene_name}_vh_clean_2.0.010000.segs.json"
        aggregation_path = scene_dir / f"{scene_name}.aggregation.json"
        vertex_instances = np.full(len(vertex_labels), -1, dtype=np.int32)
        raw_names = {}
        if seg_path.is_file() and aggregation_path.is_file():
            segments = np.asarray(
                json.loads(seg_path.read_text("utf-8"))["segIndices"],
                dtype=np.int32,
            )
            if len(segments) != len(vertex_labels):
                raise ValueError(f"segIndices length does not match mesh: {seg_path}")
            aggregation = json.loads(aggregation_path.read_text("utf-8"))
            segment_to_instance = {}
            for group in aggregation.get("segGroups", ()):
                instance_id = int(group["objectId"] if "objectId" in group else group["id"])
                raw_names[instance_id] = str(group.get("label", "object"))
                for segment_id in group.get("segments", ()):
                    segment_to_instance[int(segment_id)] = instance_id
            if segment_to_instance:
                max_segment = max(int(segments.max(initial=0)), max(segment_to_instance))
                lookup = np.full(max_segment + 1, -1, dtype=np.int32)
                for segment_id, instance_id in segment_to_instance.items():
                    lookup[segment_id] = instance_id
                valid = (segments >= 0) & (segments < len(lookup))
                vertex_instances[valid] = lookup[segments[valid]]

        face_labels = _majority_of_three(vertex_labels[triangles], invalid=-1)
        face_instances = _majority_of_three(vertex_instances[triangles], invalid=-1)

        # A few mesh faces can be outside an aggregation group.  Give those
        # faces one deterministic fallback instance per semantic class so they
        # remain available to structural plane fitting.
        next_instance = max(raw_names, default=-1) + 1
        for class_id in np.unique(face_labels):
            missing = (face_labels == class_id) & (face_instances < 0)
            if np.any(missing):
                face_instances[missing] = next_instance
                class_name = (
                    SCANNET_SEMANTIC_CLASSES[class_id]
                    if 0 <= class_id < len(SCANNET_SEMANTIC_CLASSES)
                    else "unannotated"
                )
                raw_names[next_instance] = f"{class_name}_unassigned"
                next_instance += 1

        records = {}
        for instance_id in np.unique(face_instances):
            labels = face_labels[face_instances == instance_id]
            labels = labels[(labels >= 0) & (labels < len(SCANNET_SEMANTIC_CLASSES))]
            if not len(labels):
                continue
            counts = np.bincount(labels, minlength=len(SCANNET_SEMANTIC_CLASSES))
            class_name = SCANNET_SEMANTIC_CLASSES[int(np.argmax(counts))]
            records[int(instance_id)] = {
                "name": raw_names.get(int(instance_id), f"{class_name}_{instance_id}"),
                "class_name": class_name,
            }
        return face_labels, face_instances, records

    def _load_room(self, scene_name):
        scene_dir = self.scans_path / scene_name
        ply_path = scene_dir / f"{scene_name}_vh_clean_2.labels.ply"
        if not ply_path.is_file():
            raise FileNotFoundError(f"ScanNet labeled mesh is unavailable: {ply_path}")
        vertices, triangles, vertex_labels = _read_labeled_ply(ply_path)
        face_labels, face_instances, records = self._read_instances(
            scene_dir, scene_name, vertex_labels, triangles
        )
        transform = np.asarray(self.transforms.get(scene_name, np.eye(4)), dtype=np.float64)
        vertices = transform_points(vertices, transform).astype(np.float32)
        return ScanNetMeshScene(
            self,
            scene_name,
            vertices,
            triangles,
            face_labels,
            face_instances,
            records,
        )

    def room(self, value):
        scene_name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
        if scene_name not in self._rooms:
            self._rooms[scene_name] = self._load_room(scene_name)
            while len(self._rooms) > 2:
                self._rooms.popitem(last=False)
        self._rooms.move_to_end(scene_name)
        return self._rooms[scene_name]

    def __len__(self):
        return len(self.scene_ids)

    def __iter__(self):
        return (self.room(scene_id) for scene_id in self.scene_ids)


@dataclass(frozen=True)
class ScanNetScene:
    """One ScanNet scan and its ordered 25k subset frames."""

    dataset: ScanNetDataset = field(repr=False, compare=False)
    name: str
    frames: tuple[ScanNetFrame, ...]

    @property
    def key(self):
        return self.name

    @property
    def scene_id(self):
        return self.name

    @property
    def metadata(self):
        return dict(self.dataset._scene_metadata[self.name])

    @property
    def axis_alignment(self):
        return self.dataset._scene_transforms[self.name].copy()

    @property
    def scan_path(self):
        return self.dataset.scans_path / self.name

    @property
    def mesh_path(self):
        return self.scan_path / f"{self.name}_vh_clean_2.ply"

    @property
    def labeled_mesh_path(self):
        return self.scan_path / f"{self.name}_vh_clean_2.labels.ply"

    @property
    def segmentation_path(self):
        return self.scan_path / f"{self.name}_vh_clean_2.0.010000.segs.json"

    @property
    def aggregation_path(self):
        return self.scan_path / f"{self.name}.aggregation.json"

    def get_frame(self, frame_id):
        if isinstance(frame_id, str):
            query = Path(frame_id).stem
            for frame in self.frames:
                if frame.stem == query:
                    return frame
        else:
            for frame in self.frames:
                if frame.frame_id == int(frame_id):
                    return frame
        raise KeyError(f"frame {frame_id!r} is unavailable in {self.name}")

    def structural_point_cloud(
        self,
        include_classes=("floor", "wall", "ceiling"),
        sample_points=None,
    ):
        return self.dataset.semantic_mesh.room(self.name).point_cloud(
            include_classes=include_classes,
            sample_points=sample_points,
        )

    def reconstruct(
        self,
        frame_id=None,
        stride=4,
        depth_min=0.1,
        depth_max=8.0,
        voxel_size=0.03,
        max_frames=None,
        mask=None,
        world_coordinates=True,
        progress=False,
    ):
        frames = list(self.frames)
        if frame_id is not None:
            frames = [self.get_frame(frame_id)]
        if isinstance(mask, Mapping):
            frames = [frame for frame in frames if frame.stem in mask]
        if frame_id is None and max_frames is not None:
            if max_frames < 1:
                raise ValueError("max_frames must be positive")
            frames = frames[:max_frames]
        if progress:
            from tqdm import tqdm

            frames = tqdm(frames, desc=f"Reconstructing {self.name}")

        clouds = []
        for frame in frames:
            cloud = frame.point_cloud(
                stride=stride,
                depth_min=depth_min,
                depth_max=depth_max,
                mask=mask,
                world_coordinates=world_coordinates,
            )
            if len(cloud.xyz):
                clouds.append(cloud)
        if not clouds:
            raise ValueError(f"no usable frames for scene: {self.name}")
        result = PointCloud(
            np.concatenate([cloud.xyz for cloud in clouds]),
            np.concatenate([cloud.rgb for cloud in clouds]),
            _concatenate_optional_labels(clouds, "semantic_labels"),
            _concatenate_optional_labels(clouds, "instance_labels"),
            metadata={
                "dataset": "ScanNet",
                "scene": self.name,
                "frame_count": len(clouds),
                "coordinate_frame": clouds[0].metadata["coordinate_frame"],
                "label_names": SCANNET_SEMANTIC_CLASSES,
            },
        )
        return voxel_downsample(result, voxel_size)

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
            window_name=window_name or f"ScanNet | {self.name}",
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
        return self.get_frame(index) if isinstance(index, str) else self.frames[index]

    def __repr__(self):
        return f"ScanNetScene(name={self.name!r}, frames={len(self)})"


class ScanNetDataset:
    """Discover ScanNet 25k frames and corresponding scan geometry."""

    def __init__(
        self,
        root,
        scans_path=None,
        *,
        axis_align=True,
        scene_ids=None,
        skip_invalid_poses=True,
    ):
        root = Path(root).expanduser().resolve()
        if root.name == "scannet_frames_25k":
            self.root = root.parent
            self.frames_path = root
        else:
            self.root = root
            self.frames_path = root / "scannet_frames_25k"
        self.scans_path = (
            Path(scans_path).expanduser().resolve()
            if scans_path is not None
            else self.root / "scans"
        )
        if not self.frames_path.is_dir():
            raise FileNotFoundError(f"ScanNet 25k frames directory not found: {self.frames_path}")
        self.axis_align = bool(axis_align)
        self.skip_invalid_poses = bool(skip_invalid_poses)
        selected = None if scene_ids is None else {str(value) for value in scene_ids}
        scenes = []
        transforms = {}
        metadata_by_scene = {}
        invalid_pose_count = 0
        for scene_dir in sorted(self.frames_path.glob("scene*")):
            if not scene_dir.is_dir() or (selected is not None and scene_dir.name not in selected):
                continue
            scene_name = scene_dir.name
            metadata_path = self.scans_path / scene_name / f"{scene_name}.txt"
            metadata = read_scene_metadata(metadata_path)
            metadata_by_scene[scene_name] = metadata
            transform = (
                _axis_alignment(metadata, metadata_path)
                if self.axis_align
                else np.eye(4, dtype=np.float64)
            )
            transforms[scene_name] = transform
            frames, skipped = self._index_scene(scene_dir, transform)
            invalid_pose_count += skipped
            if frames:
                scenes.append(ScanNetScene(self, scene_name, tuple(frames)))
        self.scenes = tuple(scenes)
        self.frames = tuple(frame for scene in self.scenes for frame in scene.frames)
        self.invalid_pose_count = invalid_pose_count
        self._scene_transforms = transforms
        self._scene_metadata = metadata_by_scene

    def _index_scene(self, scene_dir, transform):
        depth_intrinsics_path = scene_dir / "intrinsics_depth.txt"
        color_intrinsics_path = scene_dir / "intrinsics_color.txt"
        if not depth_intrinsics_path.is_file():
            return [], 0
        frames = []
        skipped = 0
        for pose_path in sorted((scene_dir / "pose").glob("*.txt")):
            stem = pose_path.stem
            try:
                frame_id = int(stem)
            except ValueError:
                continue
            rgb_path = scene_dir / "color" / f"{stem}.jpg"
            depth_path = scene_dir / "depth" / f"{stem}.png"
            if not rgb_path.is_file() or not depth_path.is_file():
                continue
            if self.skip_invalid_poses:
                try:
                    pose = _read_matrix(pose_path)
                    if abs(np.linalg.det(pose[:3, :3])) < 1e-8:
                        raise ValueError
                except (OSError, ValueError):
                    skipped += 1
                    continue
            semantic_path = scene_dir / "label" / f"{stem}.png"
            instance_path = scene_dir / "instance" / f"{stem}.png"
            frames.append(
                ScanNetFrame(
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    stem=stem,
                    scene=scene_dir.name,
                    frame_id=frame_id,
                    pose_path=pose_path,
                    depth_intrinsics_path=depth_intrinsics_path,
                    color_intrinsics_path=(
                        color_intrinsics_path if color_intrinsics_path.is_file() else None
                    ),
                    semantic_path=semantic_path if semantic_path.is_file() else None,
                    instance_path=instance_path if instance_path.is_file() else None,
                    scene_transform_values=tuple(transform.reshape(-1).tolist()),
                )
            )
        frames.sort(key=lambda frame: frame.frame_id)
        return frames, skipped

    @cached_property
    def semantic_mesh(self):
        if not self.scans_path.is_dir():
            raise FileNotFoundError(f"ScanNet scans directory not found: {self.scans_path}")
        return ScanNetSemanticMesh(
            self.scans_path,
            transforms=self._scene_transforms,
        )

    @property
    def scene_ids(self):
        return [scene.scene_id for scene in self.scenes]

    def list_scenes(self):
        return self.scene_ids

    def get_scene(self, scene_id):
        if isinstance(scene_id, ScanNetScene):
            if scene_id.dataset is self:
                return scene_id
            raise ValueError("scene belongs to another ScanNetDataset")
        if isinstance(scene_id, int):
            return self.scenes[scene_id]
        query = str(scene_id).replace("\\", "/").strip("/").rsplit("/", 1)[-1].casefold()
        matches = [scene for scene in self.scenes if scene.name.casefold() == query]
        if not matches:
            raise KeyError(f"unknown ScanNet scene: {scene_id}")
        return matches[0]

    def iter_scenes(self):
        return iter(self)

    def scene_frames(self, scene_id):
        return list(self.get_scene(scene_id).frames)

    def get_frame(self, scene_id=None, frame_id=None):
        if scene_id is None or frame_id is None:
            raise TypeError("scene_id and frame_id are required")
        return self.get_scene(scene_id).get_frame(frame_id)

    def __len__(self):
        return len(self.scenes)

    def __iter__(self):
        return iter(self.scenes)

    def __getitem__(self, index):
        return self.get_scene(index)

    def __repr__(self):
        return (
            f"ScanNetDataset(root={str(self.root)!r}, axis_align={self.axis_align!r}, "
            f"scenes={len(self)}, frames={len(self.frames)})"
        )
