import hashlib
import importlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s3dis_sam3d.config import slabim_registration_dir
from s3dis_sam3d.slabim.poses import load_pose_table, save_pose_table
from s3dis_sam3d.slabim.recovery import METHOD_PARAMETERS, RESULT_FILES, _source_hashes


def _module():
    name = "s3dis_sam3d.slabim.correction"
    assert importlib.util.find_spec(name) is not None, "Missing whole-region correction module"
    return importlib.import_module(name)


def _ply(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    rows = [" ".join(map(str, p)) for p in vertices]
    rows += [str(len(f)) + " " + " ".join(map(str, f)) for f in faces]
    path.write_text(header + "\n".join(rows) + "\n")


def _fixture(tmp_path):
    raw = tmp_path / "raw"
    mesh = raw / "BIM/1F/mesh"
    vertices = [
        [0, 0, 0],
        [4, 0, 0],
        [4, 4, 0],
        [0, 4, 0],
        [0, 0, 3],
        [4, 0, 3],
        [4, 4, 3],
        [0, 4, 3],
    ]
    _ply(mesh / "walls.ply", vertices, [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    _ply(mesh / "floors.ply", vertices, [[0, 3, 2, 1]])
    _ply(mesh / "columns.ply", [[2, 2, 0], [2, 2, 3], [2, 2.2, 3], [2, 2.2, 0]], [[0, 1, 2, 3]])
    # Nonplanar and mixed polygons exercise the full PLY reader, not Open3D's partial reader.
    x, y = np.meshgrid(np.linspace(0.2, 3.8, 19), np.linspace(0.2, 3.8, 19))
    z, w = np.meshgrid(np.linspace(0.2, 2.8, 15), np.linspace(0.2, 3.8, 19))
    points = np.concatenate(
        [
            np.c_[x.ravel(), y.ravel(), np.zeros(x.size)],
            np.c_[np.zeros(z.size), w.ravel(), z.ravel()],
            np.c_[np.full(z.size, 4), w.ravel(), z.ravel()],
            np.c_[w.ravel(), np.zeros(z.size), z.ravel()],
            np.c_[w.ravel(), np.full(z.size, 4), z.ravel()],
        ]
    )
    correction = np.eye(4)
    correction[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.1], degrees=True).as_matrix()
    correction[:3, 3] = [0.035, -0.025, 0.02]
    official = np.eye(4)
    official[:3, 3] = [10, 20, 30]
    inverse = np.linalg.inv(correction @ official)
    local_map = points @ inverse[:3, :3].T + inverse[:3, 3]
    source = raw / "sensor_data/1F_Region1"
    _ply(source / "map/data/uncolorized.ply", local_map, [])
    np.savetxt(source / "map/pose_map_to_bim.txt", official)
    (source / "points").mkdir()
    (source / "images").mkdir()
    (raw / "calibration_files").mkdir()
    times = np.arange(8.0) + 1000
    np.savetxt(source / "points/timestamps.txt", times)
    np.savetxt(source / "images/timestamps.txt", times[::2])
    np.savetxt(raw / "calibration_files/cam_to_lidar.txt", np.eye(4))
    save_pose_table(
        source / "points/pose_frame_to_bim.txt",
        times,
        np.repeat(official[None], len(times), axis=0),
    )
    v1 = tmp_path / "v1"
    directory = v1 / "1F_Region1"
    directory.mkdir(parents=True)
    poses = np.repeat(np.eye(4)[None], len(times), axis=0)
    poses[:, :3, 3] = np.c_[
        np.linspace(0.5, 3.5, len(times)), np.ones(len(times)), np.ones(len(times))
    ]
    save_pose_table(directory / RESULT_FILES[0], times, poses)
    save_pose_table(directory / RESULT_FILES[1], times[::2], poses[::2])
    (directory / RESULT_FILES[2]).write_text("0.png\n2.png\n4.png\n6.png\n")
    checksums = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in RESULT_FILES
    }
    summary = {
        "version": "v1",
        "method_parameters": METHOD_PARAMETERS,
        "regions": {
            "1F_Region1": {
                "lidar_frames": 8,
                "camera_frames": 4,
                "excluded_images": [],
                "sha256": checksums,
                "source_sha256": _source_hashes(raw, source),
            }
        },
    }
    (v1 / "summary.json").write_text(json.dumps(summary))
    return raw, v1, correction, checksums


def test_region_correction_recovers_known_transform_and_preserves_v1(tmp_path):
    module = _module()
    raw, v1, expected, checksums = _fixture(tmp_path)
    v2 = tmp_path / "v2"
    record = module.correct_region(raw, "1F_Region1", reference_dir=v1, output_dir=v2)
    actual = np.array(record["correction_to_bim"])
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
    for name in RESULT_FILES[:2]:
        t1, poses = load_pose_table(v1 / "1F_Region1" / name)
        t2, corrected = load_pose_table(v2 / "1F_Region1" / name)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_allclose(corrected, actual[None] @ poses, atol=1e-12)
    assert (v2 / "1F_Region1/camera_frames.txt").read_bytes() == (
        v1 / "1F_Region1/camera_frames.txt"
    ).read_bytes()
    assert {p.name for p in v2.iterdir()} == {"summary.json", "1F_Region1"}
    assert {p.name for p in (v2 / "1F_Region1").iterdir()} == set(RESULT_FILES)
    for name, digest in checksums.items():
        assert hashlib.sha256((v1 / "1F_Region1" / name).read_bytes()).hexdigest() == digest
    assert record["after"]["median_cm"] < record["before"]["median_cm"] / 100
    assert record["bootstrap"]["resamples_requested"] == 200
    # Resampling tiles on both sides of x/y=0 can omit the constraining walls.
    assert 0 < record["bootstrap"]["resamples_valid"] <= 200
    assert module.correct_region(raw, "1F_Region1", reference_dir=v1, output_dir=v2) == record
    table = v2 / "1F_Region1/camera_pose_to_bim.txt"
    table.write_text(table.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        module.correct_region(raw, "1F_Region1", reference_dir=v1, output_dir=v2)


def test_surface_reads_every_quad_and_triangle(tmp_path):
    module = _module()
    raw, _, _, _ = _fixture(tmp_path)
    surface = module.BIMSurface(raw / "BIM/1F/mesh")
    assert surface.triangles == 12
    q, normals, labels = surface.closest(np.array([[1, 1, 0.1], [0.1, 1, 1], [2.05, 2.1, 1]]))
    np.testing.assert_allclose(q[:2], [[1, 1, 0], [0, 1, 1]], atol=1e-6)
    assert list(labels) == ["floors", "walls", "columns"]
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)


def test_cached_surface_must_match_current_mesh_files(tmp_path):
    module = _module()
    raw, v1, _, _ = _fixture(tmp_path)
    surface = module.BIMSurface(raw / "BIM/1F/mesh")
    path = raw / "BIM/1F/mesh/walls.ply"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="BIM mesh changed"):
        module.correct_region(
            raw, "1F_Region1", reference_dir=v1, output_dir=tmp_path / "v2", surface=surface
        )


def test_zero_area_bim_faces_are_excluded_from_surface_normals(tmp_path):
    module = _module()
    raw, _, _, _ = _fixture(tmp_path)
    path = raw / "BIM/1F/mesh/walls.ply"
    text = path.read_text().replace("element face 4\n", "element face 5\n")
    path.write_text(text + "3 0 0 1\n")
    surface = module.BIMSurface(raw / "BIM/1F/mesh")
    assert surface.triangles == 12
    assert surface.degenerate_triangles_removed == {"columns": 0, "floors": 0, "walls": 1}
    _, normals, _ = surface.closest(np.array([[1.0, -0.1, 0.0], [0.1, 1.0, 1.0]]))
    assert np.isfinite(normals).all()
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)


def test_correction_refuses_reapplying_v2_and_overlapping_outputs(tmp_path):
    module = _module()
    raw, v1, _, _ = _fixture(tmp_path)
    for output in [v1, raw / "sensor_data/generated", v1 / "nested"]:
        with pytest.raises(ValueError, match="separate"):
            module.correct_region(raw, "1F_Region1", reference_dir=v1, output_dir=output)
    summary = json.loads((v1 / "summary.json").read_text())
    summary["version"] = "v2"
    (v1 / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="v1"):
        module.correct_region(raw, "1F_Region1", reference_dir=v1, output_dir=tmp_path / "v2")


def test_correction_refuses_changed_v1_and_inconsistent_official_transform(tmp_path):
    module = _module()
    raw, v1, _, _ = _fixture(tmp_path)
    transform_path = raw / "sensor_data/1F_Region1/map/pose_map_to_bim.txt"
    transform = np.loadtxt(transform_path)
    transform[0, 3] += 1
    np.savetxt(transform_path, transform)
    with pytest.raises(ValueError, match="map.to.BIM"):
        module.correct_region(raw, "1F_Region1", reference_dir=v1, output_dir=tmp_path / "v2")
    transform[0, 3] -= 1
    np.savetxt(transform_path, transform)
    path = v1 / "1F_Region1/lidar_pose_to_bim.txt"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        module.correct_region(raw, "1F_Region1", reference_dir=v1, output_dir=tmp_path / "v2")


def test_version_paths_are_explicit():
    assert slabim_registration_dir("v1").name == "v1"
    assert slabim_registration_dir("v2").name == "v2"
    with pytest.raises(ValueError, match="version"):
        slabim_registration_dir("../bad")


def test_bundled_v2_is_a_single_rigid_correction_of_every_v1_pose():
    v1, v2 = slabim_registration_dir("v1"), slabim_registration_dir("v2")
    old = json.loads((v1 / "summary.json").read_text())
    new = json.loads((v2 / "summary.json").read_text())
    assert new["version"] == "v2" and new["reference_version"] == "v1"
    assert set(new["regions"]) == set(old["regions"])
    assert {p.name for p in v2.iterdir()} == {"summary.json", *old["regions"]}
    for region, record in new["regions"].items():
        correction = np.array(record["correction_to_bim"])
        np.testing.assert_allclose(correction[3], [0, 0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(correction[:3, :3].T @ correction[:3, :3], np.eye(3), atol=1e-12)
        assert record["v1_sha256"] == old["regions"][region]["sha256"]
        assert record["excluded_images"] == old["regions"][region]["excluded_images"]
        assert {p.name for p in (v2 / region).iterdir()} == set(RESULT_FILES)
        for name in RESULT_FILES:
            assert (
                hashlib.sha256((v2 / region / name).read_bytes()).hexdigest()
                == record["sha256"][name]
            )
        for name in RESULT_FILES[:2]:
            times1, poses1 = load_pose_table(v1 / region / name)
            times2, poses2 = load_pose_table(v2 / region / name)
            np.testing.assert_array_equal(times1, times2)
            np.testing.assert_allclose(poses2, correction[None] @ poses1, atol=1e-11, rtol=0)
        assert (v1 / region / RESULT_FILES[2]).read_bytes() == (
            v2 / region / RESULT_FILES[2]
        ).read_bytes()
