"""Batch-calibrate BIMSync IFC regions to S3DIS."""

from s3dis_sam3d import BIMSyncDataset, S3DISDataset
from s3dis_sam3d.config import CONFIG, bimsync_calibration_dir

AREA = "Area_1"
SCENES = None  # Set to None to process every matching Area scene.
OUTPUT_DIR = bimsync_calibration_dir(AREA)
SAVE_VISUALIZATION = False
SHOW_VISUALIZATION = False
IFC_VISUALIZATION_SAMPLES = 1_000_000


def main():
    bimsync = BIMSyncDataset(CONFIG.require("bimsync_root"), area=AREA)
    s3dis = S3DISDataset()
    summary = bimsync.calibrate_scenes(
        s3dis,
        OUTPUT_DIR,
        SCENES,
        visualize=SAVE_VISUALIZATION,
        visualization_options={
            "show": SHOW_VISUALIZATION,
            "ifc_samples": IFC_VISUALIZATION_SAMPLES,
        },
    )

    print(
        f"Completed: {len(summary['success'])} succeeded, "
        f"{len(summary['errors'])} failed"
    )
    print(f"Summary: {OUTPUT_DIR / 'calibration_summary.json'}")


if __name__ == "__main__":
    main()
