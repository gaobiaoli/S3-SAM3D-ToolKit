"""Export calibrated BIMSync IFC regions as triangle meshes."""

from s3dis_sam3d import BIMSyncDataset
from s3dis_sam3d.config import OUTPUT_ROOT, bimsync_calibration_dir

AREA = "Area_1"
CALIBRATION_DIR = bimsync_calibration_dir(AREA)
OUTPUT_DIR = OUTPUT_ROOT / "bimsync_meshes" / AREA
MESH_EXTENSION = ".ply"  # .ply, .obj, .stl, .off, .glb ...
REGION = "office_11"  # Set to None to export the complete Area.


def main():
    dataset = BIMSyncDataset(
        area=AREA,
        calibration_dir=CALIBRATION_DIR,
    )
    if REGION is None:
        outputs = dataset.export_meshes(OUTPUT_DIR, extension=MESH_EXTENSION)
    else:
        outputs = [
            dataset.region(REGION).export(
                OUTPUT_DIR / f"{REGION}{MESH_EXTENSION}",
            )
        ]
    print(f"Exported {len(outputs)} calibrated mesh(es) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
