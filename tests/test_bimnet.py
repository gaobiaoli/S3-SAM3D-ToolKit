import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d

from s3dis_sam3d import BIMNetDataset, BIMNetElement, BIMNetRoom, BIMNetScene, PointCloud


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

    dataset = BIMNetDataset(tmp_path)

    assert len(dataset) == 2
    assert isinstance(dataset["train/1px"], BIMNetScene)
    assert dataset["1px"] is dataset["train/1px"]
    assert [scene.scene_id for scene in dataset.split("test")] == ["7y3"]
    assert dataset.scenes_for_scan("7y3sRwLe3Va") == (dataset["7y3"],)
    assert dataset["1px"].matterport_scan_id == "1pXnuDYAj8r"


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
