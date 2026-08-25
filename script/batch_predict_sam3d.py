"""Run SAM3D inference for filtered room-image annotations."""

import json
from pathlib import Path

from s3dis_sam3d import S23Dataset
from s3dis_sam3d.annotations import RoomImageAnnotationsParser
from s3dis_sam3d.config import s23dis_area
from s3dis_sam3d.sam3d import SAM3DBatchPredictor, SAM3DClient

# Edit these values before running this file.
ANNOTATIONS_DIR = Path("outputs/room_image_annotations/area_1_office_8")
IMAGE_ROOT = None
S23_AREA = s23dis_area("Area_1")
IMAGE_TYPE = "regular"
SAM3D_URL = "http://127.0.0.1:8000/infer"
OUTPUT_DIR = Path("outputs/sam3d_inference/area_1_office_8_depth_op")

INCLUDE_CLASSES = None
EXCLUDE_CLASSES = ["clutter"]
INSTANCES = None
MIN_BBOX_IOU = 0.5
MAX_OCCLUSION_RATIO = 0.5
MIN_PIXELS = 20
SKIP_WRAPPED_PANO = True

USE_DEPTH = True
RETURN_MASK = True
OPTIMIZE_POSE = True
OPTIMIZE_ITERATIONS = 300
SEED = 42


def main():
    annotations = RoomImageAnnotationsParser(ANNOTATIONS_DIR, image_root=IMAGE_ROOT)
    dataset = S23Dataset(S23_AREA, image_type=IMAGE_TYPE)
    predictor = SAM3DBatchPredictor(
        SAM3DClient(SAM3D_URL),
        dataset,
        annotations,
    )
    summary = predictor.predict(
        OUTPUT_DIR,
        include_classes=INCLUDE_CLASSES,
        exclude_classes=EXCLUDE_CLASSES,
        instances=INSTANCES,
        min_bbox_iou=MIN_BBOX_IOU,
        max_occlusion_ratio=MAX_OCCLUSION_RATIO,
        min_pixels=MIN_PIXELS,
        skip_wrapped_pano=SKIP_WRAPPED_PANO,
        use_depth=USE_DEPTH,
        return_mask=RETURN_MASK,
        optimize_pose=OPTIMIZE_POSE,
        optimize_iterations=OPTIMIZE_ITERATIONS,
        seed=SEED,
        cache_dir=OUTPUT_DIR,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
