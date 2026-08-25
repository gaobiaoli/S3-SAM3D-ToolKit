"""Render one calibrated IFC region from a 2D-3D-S regular camera pose."""

from s3dis_sam3d import BIMSyncDataset, S23Dataset
from s3dis_sam3d.config import OUTPUT_ROOT

AREA = "Area_1"
REGION = "office_11"
ROOM = "office_11"
FRAME_ID = 0
SHOW = False
OUTPUT_PATH = (
    OUTPUT_ROOT
    / "bimsync_renders"
    / AREA
    / REGION
    / f"{ROOM}_frame_{FRAME_ID}.png"
)


def main():
    s23dis = S23Dataset(area=AREA, image_type="regular")
    bimsync = BIMSyncDataset(area=AREA)
    result = bimsync.render_regular_frame(
        REGION,
        s23dis,
        FRAME_ID,
        OUTPUT_PATH,
        room=ROOM,
        show=SHOW,
    )
    print(f"Selected UUID: {result.uuid}")
    print(f"Source image: {result.source_image_path}")
    print(f"Source depth: {result.source_depth_path}")
    print(f"Rendered image: {result.rendered_image_path}")
    print(f"Rendered depth: {result.rendered_depth_path}")


if __name__ == "__main__":
    main()
