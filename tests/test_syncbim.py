from inspect import signature
from types import SimpleNamespace

import numpy as np

from s3dis_sam3d.syncbim import SyncBIMScene


def test_syncbim_keeps_floor_and_ceiling_triangles():
    floor = np.asarray(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)), dtype=np.float64)
    ceiling = floor + (0, 0, 3)
    room = SimpleNamespace(
        vertices=np.concatenate((floor, ceiling)),
        triangles=np.asarray(((0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6))),
        face_labels=np.asarray((1, 1, 0, 0)),
        dataset=SimpleNamespace(semantic_classes=("ceiling", "floor", "wall")),
    )
    scene = SimpleNamespace(
        name="room",
        dataset=SimpleNamespace(
            semantic_mesh=SimpleNamespace(room=lambda _: room),
        ),
    )
    wall_points = (
        np.asarray(((0, 0, 0), (0, 1, 3))),
        np.asarray(((2, 0, 0), (2, 1, 3))),
        np.asarray(((0, 0, 0), (2, 0, 3))),
        np.asarray(((0, 1, 0), (2, 1, 3))),
    )
    planes = [
        {
            "class_name": "floor",
            "normal": np.asarray((0.0, 0.0, 1.0)),
            "offset": 0.0,
            "_points": floor,
        },
        {
            "class_name": "ceiling_clip",
            "normal": np.asarray((0.0, 0.0, -1.0)),
            "offset": 3.0,
            "_points": ceiling,
        },
    ]
    for normal, offset, points in zip(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)),
        (0, 2, 0, 1),
        wall_points,
    ):
        planes.append(
            {
                "class_name": "wall",
                "normal": np.asarray(normal, dtype=np.float64),
                "offset": float(offset),
                "_points": points,
            }
        )

    syncbim = object.__new__(SyncBIMScene)
    syncbim.scene = scene
    mesh, statistics = syncbim._build_mesh(planes)

    vertices = np.asarray(mesh.vertices)
    wall_vertex_count = 4 * statistics["walls"]
    floor_vertex_count = 3 * statistics["floor_triangles"]
    np.testing.assert_allclose(
        vertices[wall_vertex_count : wall_vertex_count + floor_vertex_count, 2], 0
    )
    np.testing.assert_allclose(vertices[-6:, 2], 3)
    assert statistics["floor_triangles"] == 2
    assert statistics["ceiling_triangles"] == 2


def test_syncbim_rebuilds_one_complete_slab_per_instance():
    first = np.asarray(
        (
            (0, 0, 0),
            (2, 0, 0),
            (2, 2, 0),
            (0, 2, 0),
            (0, 1, 0),
            (2, 1, 0),
        ),
        dtype=np.float64,
    )
    second = np.asarray(
        ((4, 0, 0), (5, 0, 0), (5, 1, 0), (4, 1, 0)),
        dtype=np.float64,
    )
    plane = {
        "normal": np.asarray((0.0, 0.0, 1.0)),
        "offset": 0.0,
        "_points": np.concatenate((first, second)),
        "_instance_points": {1: first, 2: second},
    }

    syncbim = object.__new__(SyncBIMScene)
    triangles = syncbim._filled_slab_triangles(plane)
    double_areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )

    assert len(triangles) == 4
    np.testing.assert_allclose(double_areas.sum() / 2, 5.0)
    assert np.all(np.ptp(triangles[..., 0], axis=1) <= 2.0)


def test_syncbim_fill_plane_is_opt_in():
    parameter = signature(SyncBIMScene).parameters["fill_plane"]
    assert parameter.default is False
