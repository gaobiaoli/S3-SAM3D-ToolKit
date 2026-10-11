"""Batch-refine BIMNet → Matterport3D registrations (one folder per scene)."""

from s3dis_sam3d import MP3D_BIMDataset
from s3dis_sam3d.config import bimnet_registration_dir

SCENES = None  # Set to None to process every BIMNet scene with a Matterport scan.
OUTPUT_DIR = bimnet_registration_dir()


def main():
    # Register from BIMNet's own mat_pc2obj, never on top of a refined one.
    dataset = MP3D_BIMDataset(refined_registration=False)
    summary = dataset.register_scenes(OUTPUT_DIR, SCENES)
    print(
        f"Completed: {len(summary['success'])} succeeded, "
        f"{len(summary['errors'])} failed"
    )
    print(f"Summary: {OUTPUT_DIR / 'registration_summary.json'}")


if __name__ == "__main__":
    main()
