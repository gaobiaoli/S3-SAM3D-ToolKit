from pathlib import Path

import numpy as np
from PIL import Image

from s3dis_sam3d.matterport import (
    Matterport3DDataset,
    MatterportFrame,
    MatterportScene,
)
from s3dis_sam3d.models import PointCloud


def _write_scene(path: Path, scene_id: str) -> Path:
    rgb_dir = path / "undistorted_color_images"
    depth_dir = path / "undistorted_depth_images"
    camera_dir = path / "undistorted_camera_parameters"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    camera_dir.mkdir(parents=True)

    panorama_id = "0123456789abcdef0123456789abcdef"
    rgb_name = f"{panorama_id}_i0_0.jpg"
    depth_name = f"{panorama_id}_d0_0.png"
    Image.fromarray(np.full((2, 2, 3), (255, 128, 0), dtype=np.uint8)).save(
        rgb_dir / rgb_name
    )
    Image.fromarray(np.full((2, 2), 4000, dtype=np.uint16)).save(depth_dir / depth_name)

    pose = "1 0 0 1 0 1 0 2 0 0 1 3 0 0 0 1"
    (camera_dir / f"{scene_id}.conf").write_text(
        "\n".join(
            (
                "dataset matterport",
                "intrinsics_matrix 1 0 0 0 1 0 0 0 1",
                f"scan {depth_name} {rgb_name} {pose}",
            )
        ),
        encoding="utf-8",
    )
    return path


def test_frame_point_map_and_point_cloud(tmp_path):
    scene = MatterportScene("scene", _write_scene(tmp_path / "scene", "scene"))
    frame = scene.frames[0]

    assert isinstance(frame, MatterportFrame)
    assert frame.frame_id.endswith("_i0_0")
    np.testing.assert_allclose(frame.point_map()[1, 1], [1, 1, 1])
    np.testing.assert_allclose(
        frame.point_map(world_coordinates=True)[1, 1],
        [2, 1, 2],
    )
    np.testing.assert_allclose(
        frame.camera_to_world,
        [[1, 0, 0, 1], [0, -1, 0, 2], [0, 0, -1, 3], [0, 0, 0, 1]],
    )
    uv, projected_depth = frame.project_world_points([[2, 1, 2]])
    np.testing.assert_allclose(uv, [[1, 1]])
    np.testing.assert_allclose(projected_depth, [1])

    cloud = frame.point_cloud(stride=1)
    assert isinstance(cloud, PointCloud)
    assert cloud.xyz.shape == (4, 3)
    # JPEG compression may shift a channel by one or two integer values.
    np.testing.assert_allclose(cloud.rgb[0], [1, 128 / 255, 0], atol=2 / 255)
    assert cloud.metadata["coordinate_frame"] == "world"


def test_scene_reconstruct_reuses_shared_point_cloud(tmp_path):
    scene = MatterportScene("scene", _write_scene(tmp_path / "scene", "scene"))
    cloud = scene.reconstruct(stride=1, voxel_size=None)

    assert isinstance(cloud, PointCloud)
    assert cloud.xyz.shape == (4, 3)
    assert cloud.metadata["scene_id"] == "scene"
    assert cloud.metadata["frame_count"] == 1
    assert cloud.metadata["coordinate_frame"] == "world"
    assert len(scene.panoramas) == 1
    np.testing.assert_allclose(next(iter(scene.panoramas.values())).camera_position, [1, 2, 3])


def test_dataset_discovers_nested_official_layout(tmp_path):
    nested = tmp_path / "v1" / "scans" / "scene" / "scene"
    _write_scene(nested, "scene")

    dataset = Matterport3DDataset(tmp_path)

    assert dataset.scene_ids == ["scene"]
    assert dataset["scene"].scene_root == nested
    assert dataset[0] is dataset["scene"]


def test_scene_discovers_mesh_below_hash_directory(tmp_path):
    root = _write_scene(tmp_path / "scene", "scene")
    mesh_path = root / "matterport_mesh" / "mesh-hash" / "mesh-hash.obj"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.touch()

    scene = MatterportScene("scene", root)

    assert scene.has_raw_mesh
    assert scene.raw_mesh_path == mesh_path
