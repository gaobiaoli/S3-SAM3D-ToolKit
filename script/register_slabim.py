"""Recover SLABIM v1 poses; verified existing results are reused."""

from s3dis_sam3d.config import CONFIG, slabim_registration_dir
from s3dis_sam3d.slabim import recover_region

REGIONS = None  # None: every region under sensor_data; or ["5F_Region2", ...].
OUTPUT_DIR = slabim_registration_dir()


def main():
    root = CONFIG.require("slabim_root")
    regions = REGIONS
    if regions is None:
        regions = sorted(p.name for p in (root / "sensor_data").glob("*F_Region*") if p.is_dir())
    if not regions:
        raise ValueError(f"No SLABIM regions found under {root / 'sensor_data'}")
    for region in regions:
        result = recover_region(root, region, OUTPUT_DIR)
        print(f"{region}: {result['lidar_frames']} LiDAR, {result['camera_frames']} camera poses")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
