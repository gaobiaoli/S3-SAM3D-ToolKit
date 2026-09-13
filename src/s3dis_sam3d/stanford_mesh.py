"""Structural room meshes from Stanford 2D-3D-S semantic.obj."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import PointCloud
from .registration import sample_labeled_mesh, stable_seed

STRUCTURAL_CLASSES = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
)
OBJ_TO_AREA = np.asarray(((1, 0, 0), (0, 0, -1), (0, 1, 0)), dtype=np.float64)


def _material(value, expected_area):
    semantic_class, remainder = value.split("_", 1)
    instance_number, room_and_indices = remainder.split("_", 1)
    room_type, room_number, area = room_and_indices.rsplit("_", 2)
    if int(area) != expected_area:
        raise ValueError(f"material {value!r} is not in Area_{expected_area}")
    return (
        semantic_class.casefold(),
        f"{room_type}_{int(room_number)}",
        f"{semantic_class.casefold()}_{int(instance_number)}",
    )


def _vertex_index(token, vertex_count):
    value = int(token.split("/", 1)[0])
    index = value - 1 if value > 0 else vertex_count + value
    if value == 0 or not 0 <= index < vertex_count:
        raise ValueError(f"invalid OBJ vertex index: {token}")
    return index


@dataclass(frozen=True)
class StanfordMeshRoom:
    dataset: StanfordSemanticMesh
    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    face_labels: np.ndarray
    face_instances: np.ndarray
    instance_names: tuple[str, ...]

    @property
    def area(self):
        return self.dataset.area

    @property
    def key(self):
        return f"{self.area}/{self.name}"

    def point_cloud(self, include_classes=None, sample_points=None):
        triangles = self.triangles
        labels = self.face_labels
        instances = self.face_instances
        if include_classes is not None:
            keep_ids = [
                STRUCTURAL_CLASSES.index(str(name).casefold())
                for name in include_classes
                if str(name).casefold() in STRUCTURAL_CLASSES
            ]
            keep = np.isin(labels, keep_ids)
            triangles = triangles[keep]
            labels = labels[keep]
            instances = instances[keep]
        if not len(triangles):
            raise ValueError(f"no selected structural faces in {self.key}")

        point_count = self.dataset.sample_points if sample_points is None else int(sample_points)
        points, sampled_labels = sample_labeled_mesh(
            self.vertices,
            triangles,
            np.column_stack((labels, instances)),
            point_count,
            stable_seed(self.dataset.seed, self.name, "stanford"),
        )
        return PointCloud(
            points,
            semantic_labels=sampled_labels[:, 0],
            instance_labels=sampled_labels[:, 1],
            metadata={
                "area": self.area,
                "room": self.name,
                "label_names": STRUCTURAL_CLASSES,
                "instances": {
                    instance_id: {
                        "name": name,
                        "class_name": name.rsplit("_", 1)[0],
                    }
                    for instance_id, name in enumerate(self.instance_names)
                },
            },
        )


class StanfordSemanticMesh:
    """Area-level Stanford structural meshes exposed like an S3DIS dataset."""

    semantic_classes = STRUCTURAL_CLASSES

    def __init__(self, path, area="Area_1", sample_points=8_000, seed=20260810):
        self.path = Path(path).expanduser().resolve()
        self.area = str(area)
        self.sample_points = int(sample_points)
        self.seed = int(seed)
        expected_area = int(self.area.rsplit("_", 1)[-1].casefold().rstrip("ab"))
        self.rooms = self._read(expected_area)

    def _read(self, expected_area):
        vertices = array("d")
        room_faces = {}
        room_labels = {}
        room_face_instances = {}
        room_instances = {}
        active = None
        vertex_count = 0

        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                fields = raw_line.split("#", 1)[0].split()
                if not fields:
                    continue
                if fields[0] == "v":
                    vertices.extend(map(float, fields[1:4]))
                    vertex_count += 1
                elif fields[0] == "usemtl":
                    try:
                        active = _material(fields[1], expected_area)
                    except ValueError:
                        if fields[1] == "<UNK>_0_<UNK>_0_0":
                            active = None
                        else:
                            raise
                elif fields[0] == "f":
                    if active is None or active[0] not in STRUCTURAL_CLASSES:
                        continue
                    try:
                        indices = [_vertex_index(value, vertex_count) for value in fields[1:]]
                    except (ValueError, IndexError) as error:
                        raise ValueError(f"invalid OBJ face at line {line_number}") from error
                    faces = room_faces.setdefault(active[1], array("q"))
                    labels = room_labels.setdefault(active[1], array("B"))
                    face_instances = room_face_instances.setdefault(active[1], array("H"))
                    instance_names = room_instances.setdefault(active[1], {})
                    instance_id = instance_names.setdefault(active[2], len(instance_names))
                    class_id = STRUCTURAL_CLASSES.index(active[0])
                    for offset in range(1, len(indices) - 1):
                        faces.extend((indices[0], indices[offset], indices[offset + 1]))
                        labels.append(class_id)
                        face_instances.append(instance_id)

        all_vertices = np.frombuffer(vertices, dtype=np.float64).reshape(-1, 3)
        rooms = []
        for name in sorted(room_faces):
            global_triangles = np.frombuffer(room_faces[name], dtype=np.int64).reshape(-1, 3)
            used, inverse = np.unique(global_triangles, return_inverse=True)
            room_vertices = (all_vertices[used] @ OBJ_TO_AREA.T).astype(np.float32)
            rooms.append(
                StanfordMeshRoom(
                    self,
                    name,
                    room_vertices,
                    inverse.reshape(-1, 3).astype(np.int32),
                    np.frombuffer(room_labels[name], dtype=np.uint8).astype(np.int32),
                    np.frombuffer(room_face_instances[name], dtype=np.uint16).astype(np.int32),
                    tuple(
                        value
                        for value, _ in sorted(
                            room_instances[name].items(), key=lambda item: item[1]
                        )
                    ),
                )
            )
        if not rooms:
            raise RuntimeError(f"no structural rooms found in {self.path}")
        return tuple(rooms)

    def room(self, value):
        query = str(value).replace(chr(92), "/").rsplit("/", 1)[-1].casefold()
        matches = [room for room in self.rooms if room.name.casefold() == query]
        if len(matches) != 1:
            raise KeyError(f"room not found or ambiguous: {value}")
        return matches[0]

    def __len__(self):
        return len(self.rooms)

    def __iter__(self):
        return iter(self.rooms)
