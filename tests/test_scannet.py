import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from s3dis_sam3d import RGBDFrame, ScanNetDataset, ScanNetFrame, ScanNetScene, SyncBIMScene


@pytest.fixture
def scannet_root(tmp_path):
    root = tmp_path / "ScanNet"
    name = "scene0000_00"
    frames = root / "scannet_frames_25k" / name
    scans = root / "scans" / name
    scans.mkdir(parents=True)
    for modality in ("color", "depth", "pose", "label", "instance"):
        (frames / modality).mkdir(parents=True)
    intrinsic = np.eye(4)
    intrinsic[0, 0] = intrinsic[1, 1] = 2
    np.savetxt(frames / "intrinsics_depth.txt", intrinsic)
    np.savetxt(frames / "intrinsics_color.txt", intrinsic * [2, 2, 1, 1])
    pose = np.eye(4)
    pose[:3, 3] = (1, 1, 1.5)
    for number in (0, 100, 200, 300):
        stem = f"{number:06d}"
        Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(
            frames / "color" / f"{stem}.jpg"
        )
        Image.fromarray(np.array([[0, 1000], [2000, 3000]], dtype=np.uint16)).save(
            frames / "depth" / f"{stem}.png"
        )
        labels = np.repeat(np.repeat([[2, 1], [5, 0]], 2, axis=0), 2, axis=1)
        instances = np.repeat(np.repeat([[2000, 1000], [5001, 0]], 2, axis=0), 2, axis=1)
        Image.fromarray(labels.astype(np.uint8)).save(frames / "label" / f"{stem}.png")
        Image.fromarray(instances.astype(np.uint16)).save(frames / "instance" / f"{stem}.png")
        matrix = pose if number != 300 else np.full((4, 4), np.inf)
        np.savetxt(frames / "pose" / f"{stem}.txt", matrix)
    alignment = np.eye(4)
    alignment[0, 3] = 10
    (scans / f"{name}.txt").write_text(
        "axisAlignment = " + " ".join(map(str, alignment.ravel())) + "\nsceneType = Office\n"
    )

    floor = np.array(((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)), dtype=np.float32)
    surfaces = [floor, floor + (0, 0, 3)]
    for a, b in zip(floor, np.roll(floor, -1, axis=0)):
        surfaces.append(np.array((a, b, b + (0, 0, 3), a + (0, 0, 3))))
    vertices = np.concatenate(surfaces)
    vertex_labels = np.repeat([2, 22, 1, 1, 1, 1], 4)
    triangles = np.concatenate([np.array(((0, 1, 2), (0, 2, 3))) + 4 * i for i in range(6)])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar alpha\nproperty ushort label\n"
        f"element face {len(triangles)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
            ("label", "<u2"),
        ]
    )
    vertex_data = np.zeros(len(vertices), dtype=vertex_dtype)
    for column, key in enumerate(("x", "y", "z")):
        vertex_data[key] = vertices[:, column]
    vertex_data["label"] = vertex_labels
    face_data = np.zeros(len(triangles), dtype=[("count", "u1"), ("indices", "<i4", 3)])
    face_data["count"] = 3
    face_data["indices"] = triangles
    (scans / f"{name}_vh_clean_2.labels.ply").write_bytes(
        header.encode("ascii") + vertex_data.tobytes() + face_data.tobytes()
    )
    (scans / f"{name}_vh_clean_2.0.010000.segs.json").write_text(
        json.dumps({"segIndices": np.repeat(np.arange(6), 4).tolist()})
    )
    (scans / f"{name}.aggregation.json").write_text(
        json.dumps(
            {
                "segGroups": [
                    {"id": i, "objectId": i, "label": label, "segments": [i]}
                    for i, label in enumerate(("floor", "ceiling", "wall", "wall", "wall", "wall"))
                ]
            }
        )
    )
    return root


def test_frame_scene_dataset_protocol(scannet_root):
    dataset = ScanNetDataset(scannet_root)
    assert dataset.scene_ids == ["scene0000_00"]
    assert dataset.invalid_pose_count == 1
    assert len(dataset.frames) == 3
    scene = dataset[0]
    assert isinstance(scene, ScanNetScene)
    assert scene.metadata["sceneType"] == "Office"
    assert scene.labeled_mesh_path.is_file()
    assert scene.segmentation_path.is_file()
    np.testing.assert_allclose(scene.axis_alignment[:3, 3], [10, 0, 0])
    assert dataset.get_scene(scene) is scene
    assert dataset["ScanNet/scene0000_00"] is scene
    assert list(dataset.iter_scenes()) == [scene]
    assert [frame.frame_id for frame in scene] == [0, 100, 200]
    frame = dataset.get_frame(scene.name, 100)
    assert isinstance(frame, (RGBDFrame, ScanNetFrame))
    assert scene["000100"] is frame
    assert scene.get_frame("000100.txt") is frame
    assert len(ScanNetDataset(scannet_root / "scannet_frames_25k").frames) == 3
    with pytest.raises(KeyError):
        scene.get_frame(300)
    with pytest.raises(KeyError):
        dataset.get_scene("missing")


def test_depth_calibration_labels_and_axis_alignment(scannet_root):
    dataset = ScanNetDataset(scannet_root)
    frame = dataset[0][0]
    np.testing.assert_allclose(frame.depth, [[0, 1], [2, 3]])
    np.testing.assert_allclose(frame.intrinsics, np.diag([2, 2, 1]))
    np.testing.assert_allclose(frame.intrinsics_for_size((4, 6)), np.diag([6, 4, 1]))
    np.testing.assert_allclose(frame.rgb, 128)
    assert frame.rgb.dtype == np.float32
    assert frame.rgb.shape == (2, 2, 3)
    np.testing.assert_array_equal(frame.semantic_labels, [[2, 1], [5, 0]])
    np.testing.assert_array_equal(frame.instance_labels, [[2000, 1000], [5001, 0]])
    np.testing.assert_array_equal(frame.instance_indices, [[0, 0], [1, 0]])
    np.testing.assert_allclose(frame.pose[:3, 3], [1, 1, 1.5])
    np.testing.assert_allclose(frame.camera_to_world[:3, 3], [11, 1, 1.5])
    np.testing.assert_allclose(frame.point_map()[1, 1], [1.5, 1.5, 3])
    cloud = frame.point_cloud(stride=1)
    np.testing.assert_allclose(cloud.xyz[-1], [12.5, 2.5, 4.5])
    np.testing.assert_array_equal(cloud.semantic_labels, [1, 5, 0])
    pixels, depth = frame.project_world_points(cloud.xyz)
    np.testing.assert_allclose(pixels, [[1, 0], [0, 1], [1, 1]])
    np.testing.assert_allclose(depth, [1, 2, 3])
    raw = ScanNetDataset(scannet_root, axis_align=False)
    np.testing.assert_allclose(raw[0][0].camera_to_world, frame.pose)
    np.testing.assert_allclose(raw.semantic_mesh.room(raw[0].name).vertices.min(axis=0), [0, 0, 0])


def test_registered_grids_use_intrinsics_not_plain_resize(scannet_root):
    frames = scannet_root / "scannet_frames_25k/scene0000_00"
    intrinsic = np.eye(4)
    intrinsic[0, 0] = intrinsic[1, 1] = 2
    intrinsic[0, 2] = 1
    np.savetxt(frames / "intrinsics_color.txt", intrinsic)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[:, 0] = 10
    rgb[:, 1] = 20
    Image.fromarray(rgb).save(frames / "color/000000.jpg", quality=100, subsampling=0)
    Image.fromarray(np.array([[2, 1], [5, 0]], dtype=np.uint8)).save(frames / "label/000000.png")
    frame = ScanNetDataset(scannet_root)[0][0]

    # Depth x=0 projects to RGB x=1; x=1 lies outside the RGB field of view.
    np.testing.assert_allclose(frame.rgb[:, 0], 20, atol=1)
    np.testing.assert_array_equal(frame.rgb[:, 1], 0)
    np.testing.assert_array_equal(frame.semantic_labels, [[1, 0], [0, 0]])
    # The reverse mapping keeps out-of-view RGB pixels invalid, without filling.
    np.testing.assert_array_equal(frame.depth_on_color_grid(), [[0, 0], [0, 2]])


def test_rgb_resize_intrinsics_follow_pixel_centers(scannet_root):
    frame = ScanNetDataset(scannet_root)[0][0]
    intrinsic = frame.color_intrinsics_for_size((2, 2))
    np.testing.assert_allclose(intrinsic[:2, :2], np.diag([2, 2]))
    np.testing.assert_allclose(intrinsic[:2, 2], [-0.25, -0.25])


def test_labeled_mesh_instances_and_syncbim(scannet_root):
    dataset = ScanNetDataset(scannet_root)
    assert not dataset.semantic_mesh._rooms  # Geometry is loaded only on demand.
    room = dataset.semantic_mesh.room(dataset[0].name)
    assert dataset.semantic_mesh.room(room.name) is room
    assert room.triangles.shape == (12, 3)
    np.testing.assert_allclose(room.vertices.min(axis=0), [10, 0, 0])
    np.testing.assert_array_equal(room.face_labels, np.repeat([2, 22, 1, 1, 1, 1], 2))
    np.testing.assert_array_equal(room.face_instances, np.repeat(np.arange(6), 2))
    cloud = dataset[0].structural_point_cloud(sample_points=8000)
    assert cloud.xyz.shape == (8000, 3)
    assert cloud.metadata["instances"][2]["class_name"] == "wall"
    floor = room.point_cloud(include_classes=("floor",), sample_points=20)
    np.testing.assert_array_equal(floor.semantic_labels, np.full(20, 2))
    np.testing.assert_allclose(floor.xyz[:, 2], 0)
    bim = SyncBIMScene(dataset[0], sample_points=8000)
    assert bim.statistics["walls"] == 4
    depth = bim.render_depth(dataset[0][0], (2, 2))
    np.testing.assert_allclose(depth, 1.5, atol=1e-5)


def test_scene_reconstruction_and_optional_labels(scannet_root):
    dataset = ScanNetDataset(scannet_root)
    scene = dataset[0]
    cloud = scene.reconstruct(stride=1, max_frames=2, voxel_size=None)
    assert len(cloud.xyz) == 6
    assert cloud.metadata["frame_count"] == 2
    assert len(scene.reconstruct(frame_id=100, stride=1, voxel_size=None).xyz) == 3
    assert len(scene.reconstruct(stride=1, mask={"000000": [[0, 0], [0, 1]]}).xyz) == 1
    frame = scene[0]
    frame.semantic_path.unlink()
    frame.instance_path.unlink()
    frame = ScanNetDataset(scannet_root)[0][0]
    assert not frame.has_semantic and not frame.has_instance
    assert frame.point_cloud(stride=1).semantic_labels is None


def test_mesh_without_aggregation_keeps_structural_points(scannet_root):
    scan = scannet_root / "scans" / "scene0000_00"
    (scan / "scene0000_00.aggregation.json").unlink()
    cloud = ScanNetDataset(scannet_root)[0].structural_point_cloud(sample_points=200)
    assert {record["class_name"] for record in cloud.metadata["instances"].values()} == {
        "floor",
        "wall",
        "ceiling",
    }


@pytest.fixture
def prepare_module(monkeypatch):
    script_dir = Path(__file__).resolve().parents[1] / "script"
    monkeypatch.syspath_prepend(str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "prepare_scannet_syncbim", script_dir / "prepare_scannet_syncbim.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparation_matches_s23_schema_and_reuses_samples(
    scannet_root, tmp_path, prepare_module, monkeypatch
):
    module = prepare_module
    calls = []

    class Predictor:
        def __init__(self, **kwargs):
            calls.append("load")

        def predict_raw(self, path, size):
            with Image.open(path) as image:
                assert image.size == size[::-1]
            assert Path(path).parent.name == "color"
            assert "images/scannet" in str(path)
            calls.append(str(path))
            return np.full(size, 2, dtype=np.float32)

    monkeypatch.setattr(module, "DA3Predictor", Predictor)
    output = tmp_path / "prepared"
    args = [
        "--scannet-root",
        str(scannet_root),
        "--output-root",
        str(output),
        "--max-frames-per-scene",
        "1",
        "--syncbim-sample-points",
        "8000",
    ]
    module.main(args)
    manifest = output / "manifests" / "train.jsonl"
    record = json.loads(manifest.read_text())
    assert record["id"] == "scannet/scene0000_00/000000"
    assert record["training_source"] == "syncbim"
    assert Path(record["rgb"]).is_absolute() and Path(record["rgb"]).is_file()
    sample = output / record["sample"]
    module.validate_sample(sample)
    with np.load(sample) as item:
        assert set(item.files) == {
            "sample_schema_version",
            "intrinsic",
            "da3_depth_raw",
            "da3_focal_scale",
            "bim_depth",
            "bim_valid",
            "gt_depth",
            "gt_valid",
        }
        assert item["gt_depth"].dtype == np.float32
        assert item["bim_depth"].dtype == np.float16
        assert item["gt_depth"].max() == 3
        frame = ScanNetDataset(scannet_root)[0][0]
        np.testing.assert_allclose(item["intrinsic"], module.training_intrinsics(frame))
        expected_depth, _ = module.load_gt_depth(frame)
        np.testing.assert_array_equal(item["gt_depth"], expected_depth)
        np.testing.assert_allclose(item["bim_depth"], 1.5, atol=0.002)
    with Image.open(record["rgb"]) as image:
        assert image.size == (504, 504)
    label_path = Path(record["rgb"]).parent.parent / "label/000000.png"
    with Image.open(label_path) as image:
        assert image.size == (504, 504)
        assert set(np.unique(image)) <= {0, 1, 2, 5}
    assert Path(record["rgb"]) != frame.rgb_path
    assert (output / "manifests" / "val.jsonl").read_text() == ""
    assert (output / "manifests" / "test.jsonl").read_text() == ""
    metadata = json.loads((output / "syncbim.json").read_text())
    assert metadata["split"] == "train-only" and metadata["samples"] == 1
    assert metadata["scene_statistics"]["scene0000_00"]["fake_bim"]["walls"] == 4
    before = sample.stat().st_mtime_ns
    module.main(args)
    assert sample.stat().st_mtime_ns == before
    assert len(calls) == 2  # The resumed run does not instantiate DA3.


def test_preparation_resize_crop_updates_intrinsics_and_focal_scale(prepare_module):
    intrinsic = np.array([[577.591, 0, 318.905], [0, 578.73, 242.684], [0, 0, 1]])
    frame = SimpleNamespace(image_shape=(480, 640), intrinsics=intrinsic)
    assert prepare_module.resize_crop_geometry(frame.image_shape) == ((504, 672), 0, 84)
    processed = prepare_module.training_intrinsics(frame)
    assert processed[0, 0] == pytest.approx(577.591 * 1.05)
    assert processed[1, 1] == pytest.approx(578.73 * 1.05)
    assert processed[0, 2] == pytest.approx((318.905 + 0.5) * 1.05 - 0.5 - 84)
    assert processed[1, 2] == pytest.approx((242.684 + 0.5) * 1.05 - 0.5)
    expected = (577.591 + 578.73) * 1.05 / 2 / 300
    assert prepare_module.da3_focal_scale(processed) == pytest.approx(expected)


def test_preparation_resize_crop_preserves_depth_values_and_holes(prepare_module):
    import cv2

    depth = np.ones((480, 640), dtype=np.float32)
    depth[:, :320] = 2
    depth[200:240, 300:340] = 0
    result = prepare_module.resize_crop(depth, cv2.INTER_NEAREST_EXACT)
    expected = cv2.resize(depth, (672, 504), interpolation=cv2.INTER_NEAREST_EXACT)[:, 84:588]
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (504, 504)
    assert set(np.unique(result)) == {0, 1, 2}


def test_preparation_rejects_old_focal_scale_and_overwrite_repairs_it(
    scannet_root, tmp_path, prepare_module, monkeypatch
):
    module = prepare_module

    class Predictor:
        def __init__(self, **kwargs):
            pass

        def predict_raw(self, path, size):
            return np.full(size, 2, dtype=np.float32)

    monkeypatch.setattr(module, "DA3Predictor", Predictor)
    output = tmp_path / "prepared"
    args = [
        "--scannet-root", str(scannet_root),
        "--output-root", str(output),
        "--max-frames-per-scene", "1",
        "--syncbim-sample-points", "8000",
    ]
    module.main(args)
    sample = output / "samples/scannet/scene0000_00/000000.npz"
    with np.load(sample, allow_pickle=False) as item:
        payload = {key: item[key] for key in item.files}
    payload["da3_focal_scale"] *= 1.2
    module.atomic_savez(sample, **payload)

    with pytest.raises(ValueError, match="reprepare with --overwrite"):
        module.main(args)
    module.main(args + ["--overwrite"])
    frame = ScanNetDataset(scannet_root)[0][0]
    with np.load(sample, allow_pickle=False) as item:
        assert float(item["da3_focal_scale"]) == pytest.approx(
            module.da3_focal_scale(module.training_intrinsics(frame))
        )


def test_preparation_skips_low_coverage_before_da3(
    scannet_root, tmp_path, prepare_module, monkeypatch
):
    class EmptyBIM:
        def __init__(self, *args, **kwargs):
            self.statistics = {}

        def render_depth(self, frame, size, *, intrinsics=None):
            expected = prepare_module.training_intrinsics(frame)
            expected[:2, 2] += 0.5  # Match Open3D's half-pixel ray centers.
            np.testing.assert_allclose(intrinsics, expected)
            return np.zeros(size, dtype=np.float32)

    monkeypatch.setattr(prepare_module, "SyncBIMScene", EmptyBIM)
    monkeypatch.setattr(prepare_module, "DA3Predictor", lambda **kwargs: pytest.fail("DA3 loaded"))
    output = tmp_path / "empty"
    prepare_module.main(
        [
            "--scannet-root",
            str(scannet_root),
            "--output-root",
            str(output),
            "--max-frames-per-scene",
            "1",
        ]
    )
    assert (output / "manifests" / "train.jsonl").read_text() == ""
    metadata = json.loads((output / "syncbim.json").read_text())
    assert metadata["scene_statistics"]["scene0000_00"]["skipped_low_coverage"] == 1


def test_preparation_scene_list_and_stride(scannet_root, tmp_path, prepare_module):
    scene = ScanNetDataset(scannet_root)[0]
    assert [f.frame_id for f in prepare_module.collect_frames(scene, 2)] == [0, 200]
    path = tmp_path / "train.txt"
    path.write_text("# official train subset\nscene0000_00\n")
    args = prepare_module.parse_args(
        [
            "--scannet-root",
            str(scannet_root),
            "--output-root",
            str(tmp_path / "out"),
            "--scene-list",
            str(path),
        ]
    )
    assert prepare_module._selected_scenes(args) == ["scene0000_00"]
    with pytest.raises(ValueError):
        prepare_module.collect_frames(scene, 0)
