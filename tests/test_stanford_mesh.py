from pathlib import Path

import numpy as np

from s3dis_sam3d import StanfordSemanticMesh


def test_semantic_obj_is_split_into_structural_rooms(tmp_path: Path):
    obj = tmp_path / "semantic.obj"
    obj.write_text(
        """\
v 0 0 0
v 1 0 0
v 1 0 1
v 0 0 1
usemtl floor_1_office_1_1
f 1 2 3 4
usemtl chair_1_office_1_1
f 1 2 3
usemtl wall_1_conferenceRoom_2_1
f 1 3 4
""",
        encoding="utf-8",
    )

    dataset = StanfordSemanticMesh(obj, sample_points=200, seed=7)

    assert [room.name for room in dataset] == ["conferenceRoom_2", "office_1"]
    room = dataset.room("Area_1/office_1")
    assert room.key == "Area_1/office_1"
    assert room.triangles.shape == (2, 3)
    np.testing.assert_array_equal(room.face_labels, [1, 1])
    np.testing.assert_allclose(room.vertices.min(axis=0), [0, -1, 0])
    np.testing.assert_allclose(room.vertices.max(axis=0), [1, 0, 0])

    cloud = room.point_cloud(include_classes=("floor",))
    assert cloud.xyz.shape == (200, 3)
    np.testing.assert_array_equal(np.unique(cloud.semantic_labels), [1])
    assert cloud.metadata["label_names"][1] == "floor"


def test_semantic_obj_rejects_a_different_area(tmp_path: Path):
    obj = tmp_path / "semantic.obj"
    obj.write_text(
        """\
v 0 0 0
v 1 0 0
v 0 1 0
usemtl wall_1_office_1_2
f 1 2 3
""",
        encoding="utf-8",
    )

    try:
        StanfordSemanticMesh(obj, area="Area_1")
    except ValueError as error:
        assert "not in Area_1" in str(error)
    else:
        raise AssertionError("expected an Area mismatch error")
