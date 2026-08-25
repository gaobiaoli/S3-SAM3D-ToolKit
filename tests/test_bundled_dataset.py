from pathlib import Path

import numpy as np

from s3dis_sam3d import S3DISDataset, S23Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"


def test_bundled_s3dis_dataset():
    dataset = S3DISDataset(DATASET_ROOT / "s3dis")
    assert dataset.list_rooms() == ["Area_4/hallway_5"]
    room = dataset.load_room("Area_4/hallway_5")
    clutter = dataset.object_cloud("Area_4/hallway_5", "clutter_2")
    assert len(room.xyz) == 85855
    assert len(room.metadata["instances"]) == 8
    assert len(clutter.xyz) == 986


def test_bundled_s23dis_dataset():
    dataset = S23Dataset(DATASET_ROOT / "2d3ds" / "area_1")
    assert dataset.list_rooms() == [("hallway_2", 1)]
    frame = dataset.get_frame("hallway_2", 1)
    assert frame.uuid == "c6655e566bca4de983fc3affab814587"
    point_map = dataset.point_map("hallway_2", 1)
    cloud = dataset.reconstruct("hallway_2", frame_id=1, stride=16, voxel_size=None)
    assert point_map.shape == (1080, 1080, 3)
    assert len(cloud.xyz) == 4624
    np.testing.assert_allclose(
        dataset.camera_to_world("hallway_2", 1)[:3, 3],
        [-11.938932, 5.047044, 1.389766],
    )
