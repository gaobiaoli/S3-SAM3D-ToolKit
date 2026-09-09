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
    _, room_and_indices = remainder.split("_", 1)
    room_type, room_number, area = room_and_indices.rsplit("_", 2)
    if int(area) != expected_area:
        raise ValueError(f"material {value!r} is not in Area_{expected_area}")
    return semantic_class.casefold(), f"{room_type}_{int(room_number)}"


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

    @property
    def area(self):
        return self.dataset.area

    @property
    def key(self):
        return f"{self.area}/{self.name}"

    def point_cloud(self, include_classes=None):
        points, labels = sample_labeled_mesh(
            self.vertices,
            self.triangles,
            self.face_labels,
            self.dataset.sample_points,
            stable_seed(self.dataset.seed, self.name, "stanford"),
        )
        if include_classes is not None:
            keep_ids = [
                STRUCTURAL_CLASSES.index(str(name).casefold())
                for name in include_classes
                if str(name).casefold() in STRUCTURAL_CLASSES
            ]
            keep = np.isin(labels, keep_ids)
            points, labels = points[keep], labels[keep]
        return PointCloud(
            points,
            semantic_labels=labels,
            metadata={"label_names": STRUCTURAL_CLASSES, "room": self.name},
        )


class StanfordSemanticMesh:
    """Area-level Stanford structural meshes exposed like an S3DIS dataset."""

    semantic_classes = STRUCTURAL_CLASSES

    def __init__(self, path, area="Area_1", sample_points=8_000, seed=20260810):
        self.path = Path(path).expanduser().resolve()
        self.area = str(area)
        self.sample_points = int(sample_points)
        self.seed = int(seed)
        expected_area = int(self.area.rsplit("_", 1)[-1])
        self.rooms = self._read(expected_area)

    def _read(self, expected_area):
        vertices = array("d")
        room_faces = {}
        room_labels = {}
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
                    class_id = STRUCTURAL_CLASSES.index(active[0])
                    for offset in range(1, len(indices) - 1):
                        faces.extend((indices[0], indices[offset], indices[offset + 1]))
                        labels.append(class_id)

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
