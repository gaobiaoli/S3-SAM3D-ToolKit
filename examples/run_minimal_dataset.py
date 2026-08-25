"""Run the bundled minimal S3DIS and 2D-3D-S datasets end to end."""

from s3dis_sam3d import S3DISDataset, S23Dataset
from s3dis_sam3d.config import BUNDLED_DATASET_ROOT, OUTPUT_ROOT
from s3dis_sam3d.io import write_ply

DATASET_ROOT = BUNDLED_DATASET_ROOT
DEMO_OUTPUT_ROOT = OUTPUT_ROOT / "minimal_dataset"
DEMO_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

s3dis = S3DISDataset(DATASET_ROOT / "s3dis")
s3dis_room = s3dis.load_room("Area_4/hallway_5")
clutter = s3dis.object_cloud("Area_4/hallway_5", "clutter_2")
write_ply(DEMO_OUTPUT_ROOT / "s3dis_hallway_5.ply", s3dis_room)
write_ply(DEMO_OUTPUT_ROOT / "s3dis_clutter_2.ply", clutter)

s23dis = S23Dataset(DATASET_ROOT / "2d3ds" / "area_1")
frame = s23dis.get_frame("hallway_2", frame_id=1)
point_map = s23dis.point_map("hallway_2", frame_id=1)
reconstructed = s23dis.reconstruct(
    "hallway_2",
    frame_id=1,
    stride=16,
    voxel_size=None,
)
write_ply(DEMO_OUTPUT_ROOT / "2d3ds_hallway_2.ply", reconstructed)

print("Minimal datasets ran successfully")
print(f"S3DIS rooms: {s3dis.list_rooms()}")
print(f"S3DIS room/clutter points: {len(s3dis_room.xyz)}/{len(clutter.xyz)}")
print(f"2D-3D-S frame: {frame.stem}")
print(f"2D-3D-S point map/reconstruction: {point_map.shape}/{len(reconstructed.xyz)} points")
print(f"Outputs: {DEMO_OUTPUT_ROOT}")
