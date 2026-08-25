import json
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from s3dis_sam3d.pointcloud import to_open3d_geometry
from s3dis_sam3d.sam3d import (
    AssetPose,
    GLBMesh,
    build_transform,
    load_asset_transform,
    load_pose_entry,
)


def make_glb(path):
    trimesh.Scene(trimesh.creation.box()).export(path)


def write_pose(path, translation=(1, 2, 3), scale=(1, 1, 1)):
    data = {
        "object_0": {
            "rotation": [1, 0, 0, 0],
            "translation": translation,
            "scale": scale,
        }
    }
    path.write_text(json.dumps(data), "utf-8")


def test_pose_loading():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "sample_optimized.json"
        write_pose(path, scale=(2, 2, 2))
        rotation, translation, scale = load_pose_entry(path)
        transform = load_asset_transform(path)
        direct_transform = AssetPose.load(path).transform()

    np.testing.assert_allclose(rotation, [1, 0, 0, 0])
    np.testing.assert_allclose(translation, [1, 2, 3])
    np.testing.assert_allclose(scale, [2, 2, 2])
    np.testing.assert_allclose(transform, direct_transform)


def test_raw_sam3d_pose_conversion():
    transform = build_transform(
        [1, 0, 0, 0],
        [1, 2, 3],
        [1, 1, 1],
        include_glb_axis_conversion=True,
    )
    expected = np.array(
        [
            [-1, 0, 0, -1],
            [0, 0, 1, -2],
            [0, 1, 0, 3],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(transform, expected)


def test_glb_mesh_pose_sampling_and_export():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        glb_path = root / "asset.glb"
        pose_path = root / "asset_optimized.json"
        output_path = root / "asset_world.glb"
        make_glb(glb_path)
        write_pose(pose_path)

        raw = GLBMesh(glb_path)
        posed = raw.copy().apply_pose(pose_path)

        np.testing.assert_allclose(raw.transform_matrix, np.eye(4))
        np.testing.assert_allclose(posed.transform_matrix[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(posed.bounding_box().center, [1, 2, 3])
        cloud = posed.sample_points(200, seed=7)
        assert cloud.xyz.shape == (200, 3)
        np.testing.assert_allclose(cloud.metadata["transform_matrix"], posed.transform_matrix)

        assert posed.export(output_path) == output_path
        assert output_path.is_file()


def test_glb_mesh_constructor_pose_and_get():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        glb_path = root / "asset.glb"
        pose_path = root / "asset_optimized.json"
        make_glb(glb_path)
        write_pose(pose_path, translation=(1, 2, 3))

        posed = GLBMesh(glb_path, pose_path=pose_path)
        assert isinstance(posed.get(), o3d.geometry.TriangleMesh)
        assert posed.get() is posed.mesh
        assert to_open3d_geometry(posed) is posed.mesh
        np.testing.assert_allclose(posed.transform_matrix[:3, 3], [1, 2, 3])

        raw = GLBMesh(glb_path)
        returned = raw.get(pose_path)
        assert returned is raw.mesh
        np.testing.assert_allclose(raw.transform_matrix[:3, 3], [1, 2, 3])
