import json
from pathlib import Path
from unittest.mock import call, patch

import numpy as np
import open3d as o3d
import pytest
from PIL import Image

from s3dis_sam3d import (
    BIMNetDataset,
    BIMNetElement,
    BIMNetFrameRender,
    BIMNetRoom,
    BIMNetScanScene,
    BIMNetScene,
    FrameRender,
    MatterportFrame,
    PointCloud,
)


def _record(filename, ifc_id, rvt_id, guid, ifc_type):
    return {
        "id": ifc_id,
        "guid": guid,
        "name": f"component-{ifc_id}",
        "ifctype": ifc_type,
        "type": "test-family",
        "angle": 0.0,
        "transform": {
            **{
                f"M{row}{column}": float(row == column)
                for row in range(1, 5)
                for column in range(1, 5)
            },
            "IsIdentity": True,
        },
        "path": filename,
        "rvtid": str(rvt_id),
        "curved": False,
    }


def _write_scene(root: Path, split="train", scene_id="1px", with_point_cloud=True):
    (root / "ifc" / split).mkdir(parents=True, exist_ok=True)
    (root / "mat_pc2obj" / split).mkdir(parents=True, exist_ok=True)
    obj_dir = root / "obj" / split / scene_id
    obj_dir.mkdir(parents=True, exist_ok=True)
    (root / "ifc" / split / f"{scene_id}.ifc").touch()

    transform = np.eye(4)
    transform[:3, 3] = [10, 20, 30]
    np.savetxt(root / "mat_pc2obj" / split / f"{scene_id}.txt", transform)

    wall_name = "IFCWALL-IFC#11-RVT#101-CurvedFalse-el1.11_guid.obj"
    door_name = "IFCDOOR-IFC#12-RVT#102-CurvedFalse-el2.12_guid.obj"
    (obj_dir / wall_name).write_text("v 0 0 0\n", encoding="utf-8")
    (obj_dir / door_name).write_text("v 0 0 0\n", encoding="utf-8")
    (obj_dir / wall_name).with_suffix(".mtl").write_text(
        "newmtl PhongDefault\nKd 0.25 0.5 0.75\n",
        encoding="utf-8",
    )
    records = (
        _record(wall_name, 11, 101, "wall-guid", "IFCWALL"),
        _record(door_name, 12, 102, "door-guid", "IFCDOOR"),
    )
    (obj_dir / "ifcinstances.json").write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8",
    )
    rooms = [
        {
            "id": 21,
            "guid": "room-guid",
            "name": "Room",
            "instances": ["wall-guid"],
            "boundings": ["door-guid"],
        }
    ]
    (obj_dir / "ifcrooms.json").write_text(json.dumps(rooms), encoding="utf-8")

    if with_point_cloud:
        point_dir = root / "point_cloud" / split
        point_dir.mkdir(parents=True, exist_ok=True)
        scan_id = "1pXnuDYAj8r" if scene_id == "1px" else "7y3sRwLe3Va"
        floor_suffix = "_1" if scene_id.endswith("_1") else ""
        np.savetxt(
            point_dir / f"{scan_id}{floor_suffix}.txt",
            np.array(
                [
                    [1, 2, 3, 255, 128, 0, 0],
                    [4, 5, 6, 0, 255, 255, 4],
                ],
            ),
        )
    return obj_dir


def test_dataset_discovers_splits_and_matterport_mapping(tmp_path):
    _write_scene(tmp_path, "train", "1px")
    _write_scene(tmp_path, "test", "7y3")
    _write_scene(tmp_path, "train", "hxp")

    dataset = BIMNetDataset(tmp_path)

    assert len(dataset) == 3
    assert isinstance(dataset["train/1px"], BIMNetScene)
    assert dataset["1px"] is dataset["train/1px"]
    assert [scene.scene_id for scene in dataset.split("test")] == ["7y3"]
    assert dataset.scenes_for_scan("7y3sRwLe3Va") == (dataset["7y3"],)
    assert dataset["1px"].matterport_scan_id == "1pXnuDYAj8r"
    assert dataset.scene("HxpKQynjfin") is dataset["hxp"]
    assert dataset.scene(("train", "HXP")) is dataset["hxp"]
    assert dataset.scene(tmp_path / "ifc" / "train" / "hxp.ifc") is dataset["hxp"]
    assert dataset.scene(dataset["hxp"]) is dataset["hxp"]


def test_dataset_combines_scenes_for_matterport_scan_lookup(tmp_path):
    _write_scene(tmp_path, "train", "7y3")
    _write_scene(tmp_path, "test", "7y3_1")
    dataset = BIMNetDataset(tmp_path)

    assert dataset.scenes_for_scan("7y3sRwLe3Va") == (
        dataset["train/7y3"],
        dataset["test/7y3_1"],
    )
    scene = dataset.scene("7y3sRwLe3Va")
    assert isinstance(scene, BIMNetScanScene)
    assert scene.scene_ids == ("7y3", "7y3_1")
    assert scene.scenes == dataset.scenes_for_scan("7y3sRwLe3Va")
    assert dataset.scene("7y3sRwLe3Va") is scene

    first = o3d.geometry.TriangleMesh.create_box()
    second = o3d.geometry.TriangleMesh.create_box().translate((2, 0, 0))
    with patch.object(BIMNetScene, "mesh", side_effect=(first, second)) as load_mesh:
        merged = scene.mesh()

    np.testing.assert_allclose(merged.get_min_bound(), [0, 0, 0])
    np.testing.assert_allclose(merged.get_max_bound(), [3, 1, 1])
    assert load_mesh.call_count == 2
    for mesh_call in load_mesh.call_args_list:
        assert mesh_call.kwargs["coordinates"] == "point_cloud"


def test_scene_parses_elements_rooms_and_registration_matrix(tmp_path):
    _write_scene(tmp_path)
    scene = BIMNetDataset(tmp_path)["1px"]

    assert len(scene) == 2
    assert isinstance(scene.instances[0], BIMNetElement)
    assert scene.element("wall-guid").ifc_id == 11
    assert scene.element("IFC#12").rvt_id == "102"
    assert [element.ifc_type for element in scene.elements("IfcWall")] == ["IFCWALL"]
    assert scene.instances[0].has_material
    np.testing.assert_allclose(scene.instances[0].material_color, [0.25, 0.5, 0.75])
    np.testing.assert_allclose(scene.point_cloud_to_obj[:3, 3], [10, 20, 30])
    np.testing.assert_allclose(scene.obj_to_point_cloud @ scene.point_cloud_to_obj, np.eye(4))

    assert isinstance(scene.rooms[0], BIMNetRoom)
    assert [element.guid for element in scene.rooms[0].elements] == ["wall-guid"]
    assert [element.guid for element in scene.rooms[0].bounding_elements] == ["door-guid"]


def test_labeled_point_cloud_can_be_aligned_and_filtered(tmp_path):
    _write_scene(tmp_path)
    scene = BIMNetDataset(tmp_path)["1px"]

    cloud = scene.point_cloud(aligned=True, include_labels=[0])

    assert isinstance(cloud, PointCloud)
    np.testing.assert_allclose(cloud.xyz, [[11, 22, 33]])
    np.testing.assert_allclose(cloud.rgb, [[1, 128 / 255, 0]])
    np.testing.assert_array_equal(cloud.semantic_labels, [0])
    assert cloud.metadata["coordinate_frame"] == "obj"
    assert cloud.metadata["label_names"][0] == "wall"


def test_optional_assets_are_reported_without_blocking_scene_index(tmp_path):
    _write_scene(tmp_path, with_point_cloud=False)
    scene = BIMNetDataset(tmp_path)["1px"]

    assert scene.availability == {
        "ifc": True,
        "matrix": True,
        "obj": True,
        "obj_wall_filled": False,
        "point_cloud": False,
        "rvt": False,
        "rooms": True,
    }


def test_wall_filled_alias_and_filename_fallback(tmp_path):
    _write_scene(tmp_path, split="test", scene_id="d7n", with_point_cloud=False)
    filled_dir = tmp_path / "obj_wall_filled" / "test" / "d7n2"
    filled_dir.mkdir(parents=True)
    filename = "IFCWINDOW-IFC#31-RVT#301-CurvedTrue-el3.31_guid.obj"
    (filled_dir / filename).write_text("v 0 0 0\n", encoding="utf-8")

    scene = BIMNetDataset(tmp_path, split="test")["d7n"]
    elements = scene.elements(wall_filled=True)

    assert scene.wall_filled_obj_dir == filled_dir
    assert scene.has_wall_filled_mesh
    assert len(elements) == 1
    assert elements[0].ifc_type == "IFCWINDOW"
    assert elements[0].ifc_id == 31
    assert elements[0].curved


def test_obj_and_ifc_meshes_can_be_returned_in_point_cloud_coordinates(tmp_path):
    _write_scene(tmp_path)
    scene = BIMNetDataset(tmp_path)["1px"]

    with patch.object(
        BIMNetElement,
        "mesh",
        side_effect=lambda: o3d.geometry.TriangleMesh.create_box(),
    ):
        original_obj = scene.mesh(coordinates="original")
        registered_obj = scene.mesh(coordinates="point_cloud")

    with patch(
        "s3dis_sam3d.bimnet.load_ifc_mesh",
        side_effect=lambda *_args, **_kwargs: o3d.geometry.TriangleMesh.create_box(),
    ):
        original_ifc = scene.mesh(source="ifc", coordinates="original")
        registered_ifc = scene.mesh(source="ifc", coordinates="point_cloud")

    np.testing.assert_allclose(original_obj.get_min_bound(), [0, 0, 0])
    np.testing.assert_allclose(registered_obj.get_min_bound(), [-10, -20, -30])
    np.testing.assert_allclose(original_ifc.get_min_bound(), [0, 0, 0])
    np.testing.assert_allclose(registered_ifc.get_min_bound(), [-10, -20, -31])


def test_mesh_registration_uses_source_specific_coordinate_chain(tmp_path):
    _write_scene(tmp_path)
    scene = BIMNetDataset(tmp_path)["1px"]

    expected_ifc_to_obj = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    np.testing.assert_allclose(scene.ifc_to_obj, expected_ifc_to_obj)
    np.testing.assert_allclose(
        scene.mesh_to_point_cloud_transform("ifc"),
        scene.obj_to_point_cloud @ expected_ifc_to_obj,
    )
    np.testing.assert_allclose(
        scene.mesh_to_point_cloud_transform("obj"),
        scene.obj_to_point_cloud,
    )


def test_render_pairs_registered_mesh_with_matterport_frame(tmp_path):
    _write_scene(tmp_path)
    scene = BIMNetDataset(tmp_path)["1px"]
    rgb_path = tmp_path / "frame.jpg"
    depth_path = tmp_path / "frame.png"
    Image.fromarray(np.full((4, 6, 3), 128, dtype=np.uint8)).save(rgb_path)
    Image.fromarray(np.full((4, 6), 4000, dtype=np.uint16)).save(depth_path)
    frame = MatterportFrame(
        scene_id=scene.matterport_scan_id,
        panorama_id="panorama",
        camera_index=0,
        yaw_index=1,
        rgb_path=rgb_path,
        depth_path=depth_path,
        intrinsics=np.array([[5, 0, 3], [0, 5, 2], [0, 0, 1]], dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
    )
    output_path = tmp_path / "render.png"

    def fake_render(_geometries, **_options):
        return (
            np.zeros((4, 6, 3), dtype=np.float32),
            np.full((4, 6), 0.5, dtype=np.float32),
        )

    with (
        patch.object(
            BIMNetScene,
            "mesh",
            return_value=o3d.geometry.TriangleMesh.create_box(),
        ) as load_mesh,
        patch("s3dis_sam3d.bimnet.render_geometries", side_effect=fake_render) as render,
    ):
        result = scene.render_frame(frame, source="ifc")

    assert isinstance(result, BIMNetFrameRender)
    assert isinstance(result, FrameRender)
    assert result.bimnet_scene_id == "1px"
    assert result.matterport_scene_id == scene.matterport_scan_id
    assert result.frame_id == frame.frame_id
    assert result.mesh_source == "ifc"
    assert result.rendered_image_path is None
    assert result.rendered_depth_path is None
    np.testing.assert_allclose(result.source_depth, 1.0)
    np.testing.assert_allclose(result.rendered_image, 0.0)
    np.testing.assert_allclose(result.rendered_depth, 0.5)
    assert result.save(output_path) is result
    assert result.rendered_image_path == output_path
    assert result.rendered_depth_path == tmp_path / "render_depth.png"
    with Image.open(result.rendered_depth_path) as depth_image:
        np.testing.assert_array_equal(
            np.asarray(depth_image, dtype=np.uint16),
            np.full((4, 6), 2000, dtype=np.uint16),
        )
    load_mesh.assert_called_once_with(
        source="ifc",
        wall_filled=False,
        include_types=None,
        coordinates="point_cloud",
    )
    options = render.call_args.kwargs
    assert (options["width"], options["height"]) == (6, 4)
    assert options["render_depth"]
    np.testing.assert_allclose(options["intrinsics"], frame.intrinsics)
    np.testing.assert_allclose(options["world_to_camera"], frame.world_to_camera)


def test_show_uses_automatic_or_matterport_frame_view(tmp_path):
    _write_scene(tmp_path)
    scene = BIMNetDataset(tmp_path)["1px"]
    frame = MatterportFrame(
        scene_id=scene.matterport_scan_id,
        panorama_id="panorama",
        camera_index=0,
        yaw_index=1,
        rgb_path=tmp_path / "missing.jpg",
        depth_path=tmp_path / "frame.png",
        intrinsics=np.array([[5, 0, 3], [0, 5, 2], [0, 0, 1]], dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
    )
    Image.fromarray(np.full((4, 6), 4000, dtype=np.uint16)).save(frame.depth_path)
    mesh = o3d.geometry.TriangleMesh.create_box()

    with (
        patch.object(BIMNetScene, "mesh", return_value=mesh) as load_mesh,
        patch("s3dis_sam3d.bimnet.visualize_point_clouds") as visualize,
    ):
        scene.show()
        automatic_options = visualize.call_args.kwargs
        scene.show(frame)
        frame_options = visualize.call_args.kwargs

    assert load_mesh.call_args_list == [
        call(coordinates="point_cloud"),
        call(coordinates="point_cloud"),
    ]
    assert automatic_options == {
        "window_name": "BIMNet | train/1px",
        "mesh_show_back_face": True,
    }
    assert (frame_options["width"], frame_options["height"]) == (6, 4)
    assert frame.frame_id in frame_options["window_name"]
    intrinsics, extrinsic = frame_options["set_parameters"]
    np.testing.assert_allclose(intrinsics, frame.intrinsics)
    np.testing.assert_allclose(extrinsic, frame.world_to_camera)


def test_render_rejects_frame_from_another_matterport_scan(tmp_path):
    _write_scene(tmp_path)
    scene = BIMNetDataset(tmp_path)["1px"]
    frame = MatterportFrame(
        scene_id="another-scan",
        panorama_id="panorama",
        camera_index=0,
        yaw_index=0,
        rgb_path=tmp_path / "missing.jpg",
        depth_path=tmp_path / "missing.png",
        intrinsics=np.eye(3),
        camera_to_world=np.eye(4),
    )

    with pytest.raises(ValueError, match="another-scan"):
        scene.render_frame(frame)
    with (
        patch.object(BIMNetScene, "mesh") as load_mesh,
        pytest.raises(ValueError, match="another-scan"),
    ):
        scene.show(frame)
    load_mesh.assert_not_called()
