"""Minimal 2D-3D-S -> SAM3D -> posed GLB loading example."""

from pathlib import Path

from s3dis_sam3d import S23Dataset
from s3dis_sam3d.config import s23dis_area
from s3dis_sam3d.sam3d import GLBMesh, SAM3DClient

AREA = s23dis_area("Area_1")
MASK = Path("mask.png")
OUTPUT = Path("outputs")
ROOM, FRAME_ID = "office_1", 10

dataset = S23Dataset(AREA)
frame = dataset.get_frame(ROOM, FRAME_ID)
result = SAM3DClient("https://your-sam3d-server/infer").infer(
    frame.rgb_path,
    request_id=f"{ROOM}_{FRAME_ID}",
    output_dir=OUTPUT,
    mask_path=MASK,
    depth_path=frame.depth_path,
    intrinsics=frame.intrinsics,
    return_mask=True,
)

pose_path = result.optimized_pose_path or result.pose_path
asset = GLBMesh.load_posed(
    result.glb_path,
    pose_path,
    pose_space="auto",
)
asset.export(OUTPUT / f"{result.request_id}_posed.glb")
print(f"loaded transform:\n{asset.transform_matrix}")
