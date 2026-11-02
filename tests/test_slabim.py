import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s3dis_sam3d.config import Config, slabim_registration_dir
from s3dis_sam3d.slabim import load_camera_poses, load_pose_table, recover_region
from s3dis_sam3d.slabim.poses import (
    interpolate_camera_poses,
    map_to_bim,
    save_pose_table,
)
from s3dis_sam3d.slabim.recovery import (
    _bag_header_compat,
    _indexed_files,
    _register_frame,
)


def _inputs(root, region="1F_Region1", n=12):
    source = root / "sensor_data" / region
    for folder in ("points/data", "images/data", "rosbag"):
        (source / folder).mkdir(parents=True)
    times = np.arange(n, dtype=float) + 1000
    np.savetxt(source / "points/timestamps.txt", times)
    transforms = np.repeat(np.eye(4)[None], n, axis=0)
    transforms[:, 0, 3] = 10
    save_pose_table(source / "points/pose_frame_to_bim.txt", times, transforms)
    for i in range(n):
        (source / f"points/data/{i}.pcd").touch()
    np.savetxt(source / "images/timestamps.txt", [999, 1000.5, 1008.5, 1012])
    for i in range(4):
        (source / f"images/data/{i}.png").touch()
    (source / "rosbag/data_0.bag").touch()
    calibration = root / "calibration_files"
    calibration.mkdir()
    extrinsic = np.eye(4)
    extrinsic[1, 3] = 2
    np.savetxt(calibration / "cam_to_lidar.txt", extrinsic)
    return source, times


def _messages(n=12):
    for i in range(n):
        yield 1000.0 + i, np.ones((8, 3)) * i


def _registration(local, target, estimate, first):
    transform = np.eye(4)
    transform[0, 3] = local[0, 0]
    return transform, 1.0, 0.0


def test_pose_round_trip_and_constant_map_validation(tmp_path):
    _, times = _inputs(tmp_path)
    path = tmp_path / "sensor_data/1F_Region1/points/pose_frame_to_bim.txt"
    actual_times, transforms = load_pose_table(path)
    np.testing.assert_array_equal(actual_times, times)
    np.testing.assert_allclose(map_to_bim(path, times), transforms[0])
    transforms[-1, 0, 3] += 1
    save_pose_table(path, times, transforms)
    with pytest.raises(ValueError, match="constant"):
        map_to_bim(path, times)
    table = np.loadtxt(path)
    table[:, 4:8] *= 2
    np.savetxt(path, table)
    with pytest.raises(ValueError, match="quaternion"):
        load_pose_table(path)


def test_camera_interpolation_composes_extrinsic_on_right():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[:, :3, :3] = Rotation.from_euler("z", [0, 180], degrees=True).as_matrix()
    poses[1, 0, 3] = 2
    extrinsic = np.eye(4)
    extrinsic[0, 3] = 1
    result = interpolate_camera_poses(np.array([100.0, 102.0]), poses, [101.0], extrinsic)
    np.testing.assert_allclose(result[0, :3, 3], [1, 1, 0], atol=1e-12)
    np.testing.assert_allclose(
        result[0, :3, :3], Rotation.from_euler("z", 90, degrees=True).as_matrix(), atol=1e-12
    )
    with pytest.raises(ValueError, match="extrapolat"):
        interpolate_camera_poses(np.array([100.0, 102.0]), poses, [99.0], extrinsic)
    extrinsic[2, 2] = -1
    with pytest.raises(ValueError, match="extrinsic"):
        interpolate_camera_poses(np.array([100.0, 102.0]), poses, [101.0], extrinsic)


def test_indexed_files_use_numeric_order_and_reject_gaps(tmp_path):
    source, times = _inputs(tmp_path)
    paths = _indexed_files(source / "points/data", ".pcd", len(times))
    assert [int(p.stem) for p in paths] == list(range(12))
    paths[2].unlink()
    with pytest.raises(ValueError, match="indices"):
        _indexed_files(source / "points/data", ".pcd", len(times))


def test_nonstandard_ros_header_patch_is_scoped():
    reader_module = pytest.importorskip("rosbags.rosbag1.reader")
    import re

    original = reader_module.re
    with pytest.raises(RuntimeError), _bag_header_compat():
        assert reader_module.re.match(r"#ROSBAG V(\d+).(\d+)\n", "#ROSBAG  V2.0\n")
        assert re.match(r"#ROSBAG V(\d+).(\d+)\n", "#ROSBAG  V2.0\n") is None
        raise RuntimeError("open failed")
    assert reader_module.re is original


def test_recovery_outputs_only_usable_poses_and_reuses_verified_results(tmp_path):
    raw = tmp_path / "raw"
    _, times = _inputs(raw)
    output = tmp_path / "results"
    with (
        patch("s3dis_sam3d.slabim.recovery._livox_messages", return_value=_messages()),
        patch("s3dis_sam3d.slabim.recovery._register_frame", side_effect=_registration),
        patch("s3dis_sam3d.slabim.recovery._read_points", return_value=np.ones((8, 3))),
    ):
        record = recover_region(raw, "1F_Region1", output)
    assert record["lidar_frames"] == 12
    assert record["camera_frames"] == 2
    assert len(record["excluded_images"]) == 2
    assert {p.name for p in output.iterdir()} == {"summary.json", "1F_Region1"}
    assert {p.name for p in (output / "1F_Region1").iterdir()} == {
        "lidar_pose_to_bim.txt",
        "camera_pose_to_bim.txt",
        "camera_frames.txt",
    }
    names, camera_times, poses = load_camera_poses("1F_Region1", output)
    assert names == ["1.png", "2.png"]
    np.testing.assert_allclose(camera_times, [1000.5, 1008.5])
    np.testing.assert_allclose(poses[:, :3, 3], [[10.5, 2, 0], [18.5, 2, 0]])
    actual_times, lidar = load_pose_table(output / "1F_Region1/lidar_pose_to_bim.txt")
    np.testing.assert_array_equal(actual_times, times)
    np.testing.assert_allclose(lidar[:, 0, 3], 10 + np.arange(12))
    with patch("s3dis_sam3d.slabim.recovery._livox_messages") as messages:
        assert recover_region(raw, "1F_Region1", output) == record
        messages.assert_not_called()
    path = output / "1F_Region1/camera_pose_to_bim.txt"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        recover_region(raw, "1F_Region1", output)


def test_incomplete_recovery_keeps_checkpoint_and_resumes(tmp_path):
    raw = tmp_path / "raw"
    _inputs(raw)
    output = tmp_path / "results"
    with (
        patch("s3dis_sam3d.slabim.recovery._livox_messages", return_value=_messages(7)),
        patch("s3dis_sam3d.slabim.recovery._register_frame", side_effect=_registration),
        patch("s3dis_sam3d.slabim.recovery._read_points", return_value=np.ones((8, 3))),
        pytest.raises(RuntimeError, match="5.*missing"),
    ):
        recover_region(raw, "1F_Region1", output)
    assert not (output / "summary.json").exists()
    assert not (output / "1F_Region1/camera_pose_to_bim.txt").exists()
    with (
        patch("s3dis_sam3d.slabim.recovery._livox_messages", return_value=_messages()),
        patch("s3dis_sam3d.slabim.recovery._register_frame", side_effect=_registration) as register,
        patch("s3dis_sam3d.slabim.recovery._read_points", return_value=np.ones((8, 3))),
    ):
        result = recover_region(raw, "1F_Region1", output)
    assert register.call_count == 5
    assert result["lidar_frames"] == 12
    assert not (output / ".cache").exists()


def test_checkpoint_rejects_changed_raw_timestamps(tmp_path):
    raw = tmp_path / "raw"
    source, times = _inputs(raw)
    output = tmp_path / "results"
    with (
        patch("s3dis_sam3d.slabim.recovery._livox_messages", return_value=_messages(0)),
        pytest.raises(RuntimeError, match="missing"),
    ):
        recover_region(raw, "1F_Region1", output)
    times += 1
    np.savetxt(source / "points/timestamps.txt", times)
    transforms = np.repeat(np.eye(4)[None], len(times), axis=0)
    save_pose_table(source / "points/pose_frame_to_bim.txt", times, transforms)
    with pytest.raises(ValueError, match="fingerprint"):
        recover_region(raw, "1F_Region1", output)


def test_output_cannot_overlap_raw_data(tmp_path):
    raw = tmp_path / "raw"
    _inputs(raw)
    with pytest.raises(ValueError, match="separate"):
        recover_region(raw, "1F_Region1", raw / "sensor_data/derived")
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "sensor_data").symlink_to(raw / "sensor_data")
    with pytest.raises(ValueError, match="separate"):
        recover_region(linked, "1F_Region1", raw / "sensor_data/derived")


def test_icp_recovers_local_to_slam_direction():
    rng = np.random.default_rng(12)
    local = np.indices((8, 8, 8)).reshape(3, -1).T * 0.5
    local += rng.uniform(-0.05, 0.05, local.shape)
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()
    expected[:3, 3] = [0.03, -0.04, 0.02]
    target = local @ expected[:3, :3].T + expected[:3, 3]
    actual, fitness, rmse = _register_frame(local, target, np.eye(4), first=True)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert fitness == 1.0
    assert rmse < 1e-12


def test_config_preserves_existing_paths_and_slabim(tmp_path):
    path = tmp_path / "config.json"
    old = tmp_path / "old"
    config = Config(s3dis_root=old, slabim_root=tmp_path / "SLABIM")
    config.save(path)
    loaded = Config.load(path)
    assert loaded.slabim_root == config.slabim_root
    assert loaded.s3dis_root == old


def test_bundled_v1_is_complete_and_contains_only_final_results():
    root = slabim_registration_dir()
    summary = json.loads((root / "summary.json").read_text())
    assert summary["version"] == "v1"
    assert set(summary["regions"]) == {
        f"{floor}F_Region{region}" for floor in [1, 3, 4, 5] for region in [1, 2, 3]
    }
    assert {p.name for p in root.iterdir()} == {"summary.json", *summary["regions"]}
    for region, record in summary["regions"].items():
        directory = root / region
        assert {p.name for p in directory.iterdir()} == set(record["sha256"])
        for name, expected in record["sha256"].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
        times, _ = load_pose_table(directory / "lidar_pose_to_bim.txt")
        assert len(times) == record["lidar_frames"]
        names, camera_times, _ = load_camera_poses(region)
        assert len(names) == record["camera_frames"]
        assert camera_times[0] >= times[0] and camera_times[-1] <= times[-1]
