from pathlib import Path

import numpy as np

from s3dis_sam3d import S3DISDataset, S23Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"


def test_bundled_s3dis_dataset():
    dataset = S3DISDataset(DATASET_ROOT / "s3dis")
    assert [room.key for room in dataset.rooms] == ["Area_4/hallway_5"]
    room = dataset.room("Area_4/hallway_5")
    cloud = room.point_cloud()
    clutter = room.instance("clutter_2")
    assert len(cloud.xyz) == 85855
    assert len(room.instances) == 8
    assert len(clutter.point_cloud.xyz) == 986


def test_bundled_s23dis_dataset():
    dataset = S23Dataset(DATASET_ROOT / "2d3ds" / "area_1")
    assert dataset.list_rooms() == [("hallway_2", 1)]
    room = dataset.room("hallway_2")
    frame = room.get_frame(1)
    assert frame.uuid == "c6655e566bca4de983fc3affab814587"
    point_map = frame.point_map()
    cloud = room.reconstruct(frame_id=1, stride=16, voxel_size=None)
    assert point_map.shape == (1080, 1080, 3)
    assert len(cloud.xyz) == 4624
    np.testing.assert_allclose(
        frame.camera_to_world[:3, 3],
        [-11.938932, 5.047044, 1.389766],
    )
