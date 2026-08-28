"""Inspect and visualize the bundled 2D-3D-S frame."""

from s3dis_sam3d import S23Dataset
from s3dis_sam3d.config import MINIMAL_S23DIS_ROOT

DATASET_ROOT = MINIMAL_S23DIS_ROOT / "area_1"
ROOM = "hallway_2"
FRAME_ID = 1


if __name__ == "__main__":
    dataset = S23Dataset(DATASET_ROOT)
    frame = dataset.get_frame(ROOM, FRAME_ID)
    print(f"frame: {frame.stem}")
    print(f"intrinsics:\n{frame.intrinsics}")
    print(f"camera_to_world:\n{frame.camera_to_world}")
    dataset.room(ROOM).visualize(
        frame_id=FRAME_ID,
        stride=16,
        voxel_size=None,
        point_size=2.0,
    )
