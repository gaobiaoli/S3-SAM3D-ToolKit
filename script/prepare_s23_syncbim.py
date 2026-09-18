#!/usr/bin/env python3

"""Prepare selected 2D-3D-S areas as train-only SyncBIM data for MyDepth."""

import argparse
import json
from pathlib import Path

import numpy as np
from prepare_area1_prior import (
    DA3_REFERENCE_FOCAL,
    TARGET_SHAPE,
    atomic_savez,
    atomic_write_text,
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

DEFAULT_AREAS = ("Area_2", "Area_3", "Area_4", "Area_5", "Area_6")
AVAILABLE_AREAS = {"2", "3", "4", "5", "5a", "5b" , "6"}

"""
python script/prepare_s23_syncbim.py \
  --s23-root /mnt/priorbimda-data/PriorBIMDA-Datasets/Stanford2D3DS/no_xyz \
  --output-root /mnt/priorbimda-data/s23_syncbim_area2_5_504 \
  --areas 2 3 4 5 \
  --device cuda \
  --local-files-only

"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare selected S23 areas as MyDepth SyncBIM training data."
    )
    parser.add_argument("--s23-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--areas", nargs="+", default=DEFAULT_AREAS)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-area", type=int)
    parser.add_argument("--min-bim-hit-fraction", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--syncbim-sample-points", type=int, default=100_000)
    parser.add_argument(
        "--fill-plane",
        action="store_true",
        help="rebuild complete floor/ceiling planes from semantic instances",
    )
    return parser.parse_args()


def normalize_areas(values):
    """Accept 2, area_2 or Area_2; Area_5 expands to the released 5a/5b."""

    areas = []
    for value in values:
        name = str(value).strip().casefold().replace("-", "_")
        suffix = name.removeprefix("area_")
        if suffix not in AVAILABLE_AREAS:
            raise ValueError(
                f"Invalid S23 area: {value!r}; choose from Area_2 to Area_5"
            )
        if suffix == "5":
            candidates = ("Area_5a", "Area_5b")
        else:
            candidates = (f"Area_{suffix}",)
        for area in candidates:
            if area not in areas:
                areas.append(area)
    return tuple(areas)


def collect_frames(dataset, frame_stride, max_frames):
    frames = []
    for scene in dataset.scenes:
        valid = [frame for frame in scene.frames if frame.has_depth]
        frames.extend((scene.name, frame) for frame in valid[::frame_stride])
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def sample_record(area, room, frame, sample_path, output_root, s23_root, values):
    key = canonical_frame_key(frame)
    area_key = area.casefold()
    return {
        "id": f"{area_key}/{room}/{key}",
        "split": "train",
        "region": f"{area_key}/{room}",
        "area": area_key,
        "training_source": "syncbim",
        "rgb": relative_path(frame.rgb_path, s23_root),
        "sample": str(sample_path.relative_to(output_root)),
        "pose": relative_path(frame.pose_path, s23_root),
        "camera_uuid": str(frame.uuid),
        "frame_number": int(frame.frame_id),
        **values,
    }


def main():
    args = parse_args()
    if args.frame_stride < 1 or args.log_every < 1:
        raise ValueError("--frame-stride and --log-every must be positive")
    if args.max_frames_per_area is not None and args.max_frames_per_area < 1:
        raise ValueError("--max-frames-per-area must be positive")
    if args.syncbim_sample_points < 1000:
        raise ValueError("--syncbim-sample-points must be at least 1000")
    if not 0.0 <= args.min_bim_hit_fraction <= 1.0:
        raise ValueError("--min-bim-hit-fraction must be in [0, 1]")

    areas = normalize_areas(args.areas)
    s23_root = args.s23_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "syncbim.json"
    if metadata_path.is_file() and not args.overwrite:
        previous = json.loads(metadata_path.read_text("utf-8"))
        if bool(previous.get("fill_plane", False)) != args.fill_plane:
            raise ValueError(
                "--fill-plane differs from the cached dataset; use --overwrite "
                "or a new --output-root"
            )
    da3 = DA3Predictor(
        device=args.device,
        local_files_only=args.local_files_only,
    )

    records = []
    area_statistics = {}
    for area in areas:
        area_key = area.casefold()
        dataset = S23Dataset(
            area_path=s23dis_area(area, root=s23_root),
            projection_type="regular",
        )
        frames = collect_frames(
            dataset,
            args.frame_stride,
            args.max_frames_per_area,
        )
        syncbim_scenes = {}
        room_failures = {}
        area_records = []
        skipped_low_coverage = 0

        for index, (room, frame) in enumerate(frames, start=1):
            key = canonical_frame_key(frame)
            sample_path = output_root / "samples" / area_key / room / f"{key}.npz"

            if sample_path.exists() and not args.overwrite:
                validate_sample(sample_path)
                status = "reuse"
                with np.load(sample_path, allow_pickle=False) as item:
                    gt_valid_pixels = int((item["gt_valid"] > 0).sum())
                    bim_hit_pixels = int((item["bim_valid"] > 0).sum())
                    focal_scale = float(item["da3_focal_scale"].item())
            else:
                if room not in syncbim_scenes and room not in room_failures:
                    try:
                        syncbim_scenes[room] = SyncBIMScene(
                            dataset.get_scene(room),
                            sample_points=args.syncbim_sample_points,
                            fill_plane=args.fill_plane,
                        )
                    except ValueError as error:
                        room_failures[room] = str(error)
                        print(f"[{area}] skip {room}: {error}", flush=True)
                if room in room_failures:
                    continue

                intrinsic = np.asarray(
                    frame.intrinsics_for_size(TARGET_SHAPE),
                    dtype=np.float32,
                )
                focal_px = float((intrinsic[0, 0] + intrinsic[1, 1]) / 2.0)
                focal_scale = focal_px / DA3_REFERENCE_FOCAL
                if not np.isfinite(focal_scale) or focal_scale <= 0:
                    raise ValueError(f"{area_key}/{room}/{key}: invalid DA3 focal scale")

                bim_depth = syncbim_scenes[room].render_depth(frame, TARGET_SHAPE)
                bim_depth = np.asarray(bim_depth, dtype=np.float32)
                bim_valid = np.isfinite(bim_depth) & (bim_depth > 0)
                bim_depth = np.where(bim_valid, bim_depth, 0).astype(np.float32)
                bim_hit_pixels = int(bim_valid.sum())
                if float(bim_valid.mean()) < args.min_bim_hit_fraction:
                    skipped_low_coverage += 1
                    status = "skip"
                    if (
                        index == 1
                        or index % args.log_every == 0
                        or index == len(frames)
                    ):
                        print(
                            f"[{area} {index:05d}/{len(frames):05d}] "
                            f"{status:5s} {room}/{key} BIM={bim_hit_pixels}",
                            flush=True,
                        )
                    continue

                da3_depth_raw = da3.predict_raw(frame.rgb_path, TARGET_SHAPE)
                gt_depth, gt_valid = load_gt_depth(frame.depth_path, TARGET_SHAPE)
                gt_valid_pixels = int(gt_valid.sum())
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

            if bim_hit_pixels / (TARGET_SHAPE[0] * TARGET_SHAPE[1]) < (
                args.min_bim_hit_fraction
            ):
                skipped_low_coverage += 1
                continue
            area_records.append(
                sample_record(
                    area,
                    room,
                    frame,
                    sample_path,
                    output_root,
                    s23_root,
                    {
                        "gt_valid_pixels": gt_valid_pixels,
                        "bim_hit_pixels": bim_hit_pixels,
                        "da3_focal_scale": focal_scale,
                    },
                )
            )
            if index == 1 or index % args.log_every == 0 or index == len(frames):
                print(
                    f"[{area} {index:05d}/{len(frames):05d}] "
                    f"{status:5s} {room}/{key} "
                    f"GT={gt_valid_pixels} BIM={bim_hit_pixels}",
                    flush=True,
                )

        area_records.sort(key=lambda record: record["id"])
        records.extend(area_records)
        area_statistics[area_key] = {
            "frames_considered": len(frames),
            "samples": len(area_records),
            "skipped_low_coverage": skipped_low_coverage,
            "room_failures": room_failures,
        }

    records.sort(key=lambda record: record["id"])
    manifest_dir = output_root / "manifests"
    write_jsonl(manifest_dir / "train.jsonl", records)
    write_jsonl(manifest_dir / "val.jsonl", [])
    write_jsonl(manifest_dir / "test.jsonl", [])
    write_jsonl(manifest_dir / "all.jsonl", records)
    metadata = {
        "schema": "MyDepth S23PriorBIMDataset",
        "prior": "S23 SyncBIM",
        "split": "train-only",
        "areas": list(areas),
        "target_shape": list(TARGET_SHAPE),
        "samples": len(records),
        "frame_stride": args.frame_stride,
        "minimum_bim_hit_fraction": args.min_bim_hit_fraction,
        "syncbim_sample_points": args.syncbim_sample_points,
        "fill_plane": args.fill_plane,
        "area_statistics": area_statistics,
    }
    atomic_write_text(
        metadata_path,
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
    )
    print(f"Wrote {len(records)} train-only samples to {manifest_dir / 'train.jsonl'}")


if __name__ == "__main__":
    main()
