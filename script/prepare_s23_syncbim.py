#!/usr/bin/env python3

"""Prepare Area_1 with S23-derived SyncBIM depth in the PriorBIMDA schema."""

import argparse
import json
from pathlib import Path

import numpy as np
from prepare_area1_prior import (
    AREA,
    DA3_REFERENCE_FOCAL,
    EXPECTED_FRAME_COUNTS,
    EXPECTED_TOTAL_FRAMES,
    ROOM_SPLITS,
    TARGET_SHAPE,
    atomic_savez,
    canonical_frame_key,
    load_gt_depth,
    relative_path,
    validate_sample,
    write_jsonl,
)

from s3dis_sam3d.config import s23dis_area
from s3dis_sam3d.mde import DA3Predictor
from s3dis_sam3d.s23dis import S23Dataset
from s3dis_sam3d.syncbim import SyncBIMScene


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare S23 Area_1 with room-level SyncBIM depth."
    )
    parser.add_argument("--s23-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--syncbim-sample-points", type=int, default=100_000)
    return parser.parse_args()


def collect_frames(dataset):
    available = set(dataset.scene_ids)
    required = {room for rooms in ROOM_SPLITS.values() for room in rooms}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Missing S23 rooms: {missing}")

    records = []
    camera_owners = {}
    for split in ("train", "val", "test"):
        for room in ROOM_SPLITS[split]:
            for frame in dataset.get_scene(room).frames:
                if not frame.has_depth:
                    continue
                records.append((split, room, frame))
                camera_owners.setdefault(frame.uuid, set()).add(split)

    leaked = {
        camera: owners for camera, owners in camera_owners.items() if len(owners) > 1
    }
    if leaked:
        raise ValueError(f"Camera UUID leakage: {list(leaked.items())[:10]}")

    counts = {
        split: sum(item[0] == split for item in records)
        for split in ("train", "val", "test")
    }
    if counts != EXPECTED_FRAME_COUNTS or len(records) != EXPECTED_TOTAL_FRAMES:
        raise ValueError(f"Area_1 frame population differs from PriorBIMDA: {counts}")
    return records


def main():
    args = parse_args()
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")
    if args.syncbim_sample_points < 1000:
        raise ValueError("--syncbim-sample-points must be at least 1000")

    s23_root = args.s23_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = S23Dataset(
        area_path=s23dis_area(AREA, root=s23_root),
        projection_type="regular",
    )
    frames = collect_frames(dataset)
    da3 = DA3Predictor(
        device=args.device,
        local_files_only=args.local_files_only,
    )
    syncbim_scenes = {}
    manifests = {"train": [], "val": [], "test": []}

    for index, (split, room, frame) in enumerate(frames, start=1):
        key = canonical_frame_key(frame)
        sample_id = f"{room}/{key}"
        sample_path = output_root / "samples" / room / (key + ".npz")

        if sample_path.exists() and not args.overwrite:
            validate_sample(sample_path)
            status = "reuse"
            with np.load(sample_path, allow_pickle=False) as item:
                gt_valid_pixels = int((item["gt_valid"] > 0).sum())
                bim_hit_pixels = int((item["bim_valid"] > 0).sum())
                focal_scale = float(item["da3_focal_scale"].item())
        else:
            intrinsic = np.asarray(
                frame.intrinsics_for_size(TARGET_SHAPE),
                dtype=np.float32,
            )
            focal_px = float((intrinsic[0, 0] + intrinsic[1, 1]) / 2.0)
            focal_scale = focal_px / DA3_REFERENCE_FOCAL
            if not np.isfinite(focal_scale) or focal_scale <= 0:
                raise ValueError(sample_id + ": invalid DA3 focal scale")

            da3_depth_raw = da3.predict_raw(frame.rgb_path, TARGET_SHAPE)
            if room not in syncbim_scenes:
                syncbim_scenes[room] = SyncBIMScene(
                    dataset.get_scene(room),
                    sample_points=args.syncbim_sample_points,
                )
            bim_depth = syncbim_scenes[room].render_depth(frame, TARGET_SHAPE)
            bim_depth = np.asarray(bim_depth, dtype=np.float32)
            bim_valid = np.isfinite(bim_depth) & (bim_depth > 0)
            bim_depth = np.where(bim_valid, bim_depth, 0).astype(np.float32)
            gt_depth, gt_valid = load_gt_depth(frame.depth_path, TARGET_SHAPE)
            gt_valid_pixels = int(gt_valid.sum())
            bim_hit_pixels = int(bim_valid.sum())

            atomic_savez(
                sample_path,
                sample_schema_version=np.asarray(1, dtype=np.uint16),
                intrinsic=intrinsic,
                da3_depth_raw=da3_depth_raw.astype(np.float16),
                da3_focal_scale=np.asarray(focal_scale, dtype=np.float32),
                bim_depth=bim_depth.astype(np.float16),
                bim_valid=bim_valid.astype(np.uint8),
                gt_depth=gt_depth.astype(np.float32),
                gt_valid=gt_valid.astype(np.uint8),
            )
            status = "write"

        manifests[split].append(
            {
                "id": sample_id,
                "split": split,
                "region": room,
                "rgb": relative_path(frame.rgb_path, s23_root),
                "sample": str(sample_path.relative_to(output_root)),
                "pose": relative_path(frame.pose_path, s23_root),
                "camera_uuid": str(frame.uuid),
                "frame_number": int(frame.frame_id),
                "gt_valid_pixels": gt_valid_pixels,
                "bim_hit_pixels": bim_hit_pixels,
                "da3_focal_scale": focal_scale,
            }
        )
        if index == 1 or index % args.log_every == 0 or index == len(frames):
            print(
                f"[{index:05d}/{len(frames):05d}] {status:5s} {split:5s} {sample_id} GT={gt_valid_pixels} BIM={bim_hit_pixels}",
                flush=True,
            )

    for records in manifests.values():
        records.sort(key=lambda record: record["id"])
    all_records = sorted(
        [record for records in manifests.values() for record in records],
        key=lambda record: record["id"],
    )
    manifest_dir = output_root / "manifests"
    for split in ("train", "val", "test"):
        write_jsonl(manifest_dir / (split + ".jsonl"), manifests[split])
    write_jsonl(manifest_dir / "all.jsonl", all_records)
    metadata = {
        "schema": "PriorBIMDA Area_1",
        "prior": "S23 SyncBIM",
        "target_shape": list(TARGET_SHAPE),
        "frame_counts": {key: len(value) for key, value in manifests.items()},
        "syncbim_sample_points": args.syncbim_sample_points,
    }
    (output_root / "syncbim.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
