"""Calibrate BIMSync IFC rooms to an upright Stanford semantic mesh."""

import argparse
import json
from pathlib import Path

from s3dis_sam3d import BIMSyncDataset, StanfordSemanticMesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-obj", type=Path, required=True)
    parser.add_argument("--bimsync-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--area", default="Area_1")
    parser.add_argument("--rooms", nargs="*")
    parser.add_argument("--sample-points", type=int, default=8_000)
    args = parser.parse_args()

    stanford = StanfordSemanticMesh(
        args.semantic_obj,
        area=args.area,
        sample_points=args.sample_points,
    )
    bimsync = BIMSyncDataset(args.bimsync_root, area=args.area)
    summary = bimsync.calibrate_scenes(
        stanford,
        args.output_dir,
        scenes=args.rooms,
        ifc_samples=args.sample_points,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir.resolve()),
                "accepted": len(summary["success"]),
                "rejected": len(summary["errors"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
