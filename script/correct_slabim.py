"""Generate SLABIM v2 by applying one map/BIM correction per region to v1."""

from s3dis_sam3d.config import CONFIG, slabim_registration_dir
from s3dis_sam3d.slabim import correct_region
from s3dis_sam3d.slabim.correction import BIMSurface

REGIONS = None  # None: all v1 regions; or ["5F_Region2", ...].
REFERENCE_DIR = slabim_registration_dir("v1")
OUTPUT_DIR = slabim_registration_dir("v2")


def main():
    root = CONFIG.require("slabim_root")
    regions = REGIONS
    if regions is None:
        regions = sorted(p.name for p in REFERENCE_DIR.glob("*F_Region*") if p.is_dir())
    if not regions:
        raise ValueError(f"No v1 regions found under {REFERENCE_DIR}")
    surface, current_floor = None, None
    for region in regions:
        floor = region.split("_")[0]
        if floor != current_floor:
            surface = BIMSurface(root / "BIM" / floor / "mesh")
            current_floor = floor
        print(f"Correcting {region} ...", flush=True)
        result = correct_region(root, region, REFERENCE_DIR, OUTPUT_DIR, surface=surface)
        change = result["correction"]
        print(
            f"{region}: tilt {change['tilt_deg']:.4f} deg, "
            f"camera shift {change['camera_shift_cm_median']:.2f} cm, "
            f"residual {result['before']['median_cm']:.2f} -> "
            f"{result['after']['median_cm']:.2f} cm",
            flush=True,
        )
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
