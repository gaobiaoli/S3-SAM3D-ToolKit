"""Generate one S3DIS-instance annotation JSON for every 2D-3D-S frame."""

import json
from pathlib import Path

from s3dis_sam3d.annotations import RoomImageBatchAnnotator
from s3dis_sam3d.config import S3DIS_ROOT, s23dis_area

S23_AREA = s23dis_area("Area_3")
IMAGE_TYPE = "regular"
ROOM_NAME = "Area_3/office_1"
SELECTED_CLASSES = ["chair", "table", "sofa", "clutter", "bookcase"]
MIN_PIXELS = 20
MAX_INSTANCE_POINTS = 200_000
SAVE_EMPTY = False
OUTPUT_DIR = Path("outputs/room_image_annotations/area_3_office_1")


def main():
    annotator = RoomImageBatchAnnotator(
        S3DIS_ROOT,
        S23_AREA,
        image_type=IMAGE_TYPE,
    )
    summary = annotator.annotate_room_images(
        ROOM_NAME,
        selected_classes=SELECTED_CLASSES,
        output_dir=OUTPUT_DIR,
        min_pixels=MIN_PIXELS,
        max_instance_points=MAX_INSTANCE_POINTS,
        save_empty=SAVE_EMPTY,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
