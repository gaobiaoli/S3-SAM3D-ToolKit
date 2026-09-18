#!/usr/bin/env python3

"""Prepare ScanNet 25k as train-only MyDepth data, matching S23 SyncBIM.

The fake BIM is a simplified wall/floor/ceiling shell fitted from the labeled
scan mesh, not a ground-truth CAD model.  It is rendered in the same aligned
world coordinates as the frame poses. RGB/labels are sampled on the registered
depth grid, then all modalities use 640x480 -> 672x504 -> a centered 504x504 crop.
Processed RGB and labels are saved with the same camera grid as the cached
depths. Do not apply raw sensor extrinsics again to the official 25k release.
Optional --scene-list restricts the run to selected ScanNet scenes.

Example:
    python script/prepare_scannet_syncbim.py \
        --scannet-root /home/bgao491/ScanNet \
        --output-root /mnt/priorbimda-data/scannet_syncbim_504 \
        --device cuda --local-files-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from prepare_area1_prior import (
    TARGET_SHAPE,
    atomic_savez,
    atomic_write_text,
    validate_sample,
    write_jsonl,
)

from s3dis_sam3d.mde import DA3Predictor, da3_processed_geometry
from s3dis_sam3d.scannet import ScanNetDataset
from s3dis_sam3d.syncbim import SyncBIMScene


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare ScanNet 25k with the same sample schema as S23 SyncBIM."
    )
    parser.add_argument("--scannet-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", help="Only prepare these scene IDs.")
    parser.add_argument("--scene-list", type=Path, help="Scene IDs, one per line.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--max-frames-per-scene", type=int)
    parser.add_argument("--min-bim-hit-fraction", type=float, default=0.2)
    parser.add_argument("--syncbim-sample-points", type=int, default=100_000)
    parser.add_argument(
        "--fill-plane",
        action="store_true",
        help="rebuild complete floor/ceiling planes from semantic instances",
    )
    parser.add_argument("--no-axis-align", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args(argv)


def resize_crop_geometry(source_shape, target_shape=TARGET_SHAPE):
    height, width = source_shape
    target_height, target_width = target_shape
    scale = max(target_height / height, target_width / width)
    resized = (round(height * scale), round(width * scale))
    top = (resized[0] - target_height) // 2
    left = (resized[1] - target_width) // 2
    return resized, top, left


def resize_crop(image, interpolation, target_shape=TARGET_SHAPE):
    resized, top, left = resize_crop_geometry(image.shape[:2], target_shape)
    image = cv2.resize(image, resized[::-1], interpolation=interpolation)
    height, width = target_shape
    return np.ascontiguousarray(image[top:top + height, left:left + width])


def training_intrinsics(frame, target_shape=TARGET_SHAPE):
    """Depth-grid intrinsics after uniform resize and center crop."""
    height, width = frame.image_shape
    resized, top, left = resize_crop_geometry((height, width), target_shape)
    scale_x, scale_y = resized[1] / width, resized[0] / height
    intrinsic = frame.intrinsics.copy()
    intrinsic[0] *= scale_x
    intrinsic[1] *= scale_y
    intrinsic[0, 2] = (frame.intrinsics[0, 2] + 0.5) * scale_x - 0.5 - left
    intrinsic[1, 2] = (frame.intrinsics[1, 2] + 0.5) * scale_y - 0.5 - top
    return intrinsic


def load_gt_depth(frame, target_shape=TARGET_SHAPE):
    """Resize/crop released depth without filling holes or averaging boundaries."""
    depth = resize_crop(frame.depth, cv2.INTER_NEAREST_EXACT, target_shape)
    valid = np.isfinite(depth) & (depth > 0)
    depth[~valid] = 0
    return depth, valid


def da3_focal_scale(intrinsic):
    """DA3 sees the final cropped RGB, whose intrinsics are saved in the NPZ."""
    _, _, _, focal_scale = da3_processed_geometry(TARGET_SHAPE, intrinsic)
    return focal_scale


def prepare_images(frame, rgb_path, label_path):
    rgb = resize_crop(frame.rgb, cv2.INTER_CUBIC)
    rgb = np.rint(rgb.clip(0, 255)).astype(np.uint8)
    labels = resize_crop(frame.semantic_labels.astype(np.float32), cv2.INTER_NEAREST_EXACT)
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(labels.astype(np.uint8)).save(label_path)


def collect_frames(scene, frame_stride=1, max_frames=None):
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    frames = [frame for frame in scene.frames if frame.has_depth][::frame_stride]
    return frames if max_frames is None else frames[:max_frames]


def sample_record(scene, frame, sample_path, output_root, values, rgb_path):
    key = f"{frame.frame_id:06d}"
    return {
        "id": f"scannet/{scene}/{key}",
        "split": "train",
        "region": f"scannet/{scene}",
        "area": "scannet",
        "dataset": "scannet",
        "training_source": "syncbim",
        # MyDepth resolves relative RGB paths against its Stanford root, so
        # keep an absolute path to the processed crop, not the source RGB.
        "rgb": str(rgb_path),
        "sample": str(sample_path.relative_to(output_root)),
        "pose": str(frame.pose_path),
        "frame_number": int(frame.frame_id),
        **values,
    }


def _selected_scenes(args):
    selected = set(args.scenes or ())
    if args.scene_list is not None:
        selected.update(
            line.split("#", 1)[0].strip()
            for line in args.scene_list.read_text("utf-8").splitlines()
            if line.split("#", 1)[0].strip()
        )
    if args.scene_list is not None and not selected:
        raise ValueError("--scene-list contains no scene IDs")
    return sorted(selected) if selected else None


def main(argv=None):
    args = parse_args(argv)
    for name in ("frame_stride", "log_every", "max_scenes", "max_frames_per_scene"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.syncbim_sample_points < 1000:
        raise ValueError("--syncbim-sample-points must be at least 1000")
    if not 0 <= args.min_bim_hit_fraction <= 1:
        raise ValueError("--min-bim-hit-fraction must be in [0, 1]")

    selected = _selected_scenes(args)
    dataset = ScanNetDataset(
        args.scannet_root,
        axis_align=not args.no_axis_align,
        scene_ids=selected,
    )
    if selected is not None:
        missing = set(selected) - set(dataset.scene_ids)
        if missing:
            raise ValueError(f"selected scenes have no usable RGB-D frames: {sorted(missing)}")
    scenes = list(dataset.scenes)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    if not scenes:
        raise ValueError("no usable ScanNet scenes found")

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
    records = []
    scene_statistics = {}
    da3 = None  # Reusing prepared samples does not need to load the DA3 model.
    processed = 0
    total_frames = sum(
        len(collect_frames(scene, args.frame_stride, args.max_frames_per_scene)) for scene in scenes
    )
    print(
        f"ScanNet: {len(scenes)} scenes, {total_frames} selected frames, "
        f"{dataset.invalid_pose_count} invalid poses skipped; train-only output",
        flush=True,
    )
    for scene in scenes:
        frames = collect_frames(scene, args.frame_stride, args.max_frames_per_scene)
        syncbim = None
        failure = None
        statistics = {
            "frames_considered": len(frames),
            "samples": 0,
            "skipped_low_coverage": 0,
        }
        for frame in frames:
            processed += 1
            key = f"{frame.frame_id:06d}"
            sample_path = output_root / "samples" / "scannet" / scene.name / f"{key}.npz"
            image_dir = output_root / "images" / "scannet" / scene.name
            rgb_path = image_dir / "color" / f"{key}.png"
            label_path = image_dir / "label" / f"{key}.png"
            intrinsic = training_intrinsics(frame)
            focal_scale = da3_focal_scale(intrinsic)
            if sample_path.is_file() and not args.overwrite:
                validate_sample(sample_path)
                for path in (rgb_path, label_path):
                    if not path.is_file():
                        raise ValueError(f"{path}: missing processed image; reprepare with --overwrite")
                    with Image.open(path) as image:
                        if (image.height, image.width) != TARGET_SHAPE:
                            raise ValueError(f"{path}: invalid crop size; reprepare with --overwrite")
                with np.load(sample_path, allow_pickle=False) as item:
                    if not np.allclose(item["intrinsic"], intrinsic, rtol=1e-6, atol=1e-5):
                        raise ValueError(
                            f"{sample_path}: sample does not use the cropped camera grid; "
                            "reprepare with --overwrite"
                        )
                    values = {
                        "gt_valid_pixels": int((item["gt_valid"] > 0).sum()),
                        "bim_hit_pixels": int((item["bim_valid"] > 0).sum()),
                        "da3_focal_scale": float(item["da3_focal_scale"].item()),
                    }
                if not np.isclose(values["da3_focal_scale"], focal_scale, rtol=1e-5):
                    raise ValueError(
                        f"{sample_path}: DA3 focal scale does not match the RGB input; "
                        "reprepare with --overwrite"
                    )
                status = "reuse"
            else:
                if syncbim is None and failure is None:
                    try:
                        syncbim = SyncBIMScene(
                            scene,
                            sample_points=args.syncbim_sample_points,
                            fill_plane=args.fill_plane,
                        )
                        statistics["fake_bim"] = dict(syncbim.statistics)
                    except (ValueError, FileNotFoundError) as error:
                        failure = str(error)
                        statistics["failure"] = failure
                        print(f"[ScanNet] skip {scene.name}: {failure}", flush=True)
                if failure is not None:
                    continue

                # Open3D samples (u + 0.5, v + 0.5); our image centers are (u, v).
                render_intrinsic = intrinsic.copy()
                render_intrinsic[:2, 2] += 0.5
                bim_depth = np.asarray(
                    syncbim.render_depth(frame, TARGET_SHAPE, intrinsics=render_intrinsic),
                    dtype=np.float32,
                )
                bim_valid = np.isfinite(bim_depth) & (bim_depth > 0)
                bim_depth = np.where(bim_valid, bim_depth, 0).astype(np.float32)
                if float(bim_valid.mean()) < args.min_bim_hit_fraction:
                    statistics["skipped_low_coverage"] += 1
                    continue

                prepare_images(frame, rgb_path, label_path)
                if da3 is None:
                    da3 = DA3Predictor(
                        device=args.device,
                        local_files_only=args.local_files_only,
                    )
                da3_depth_raw = np.asarray(
                    da3.predict_raw(rgb_path, TARGET_SHAPE), dtype=np.float32
                )
                if da3_depth_raw.shape != TARGET_SHAPE:
                    raise ValueError(f"{scene.name}/{key}: unexpected DA3 depth shape")
                gt_depth, gt_valid = load_gt_depth(frame, TARGET_SHAPE)
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
                values = {
                    "gt_valid_pixels": int(gt_valid.sum()),
                    "bim_hit_pixels": int(bim_valid.sum()),
                    "da3_focal_scale": focal_scale,
                }
                status = "write"

            if values["bim_hit_pixels"] / np.prod(TARGET_SHAPE) < args.min_bim_hit_fraction:
                statistics["skipped_low_coverage"] += 1
                continue
            records.append(sample_record(scene.name, frame, sample_path, output_root, values, rgb_path))
            statistics["samples"] += 1
            if processed == 1 or processed % args.log_every == 0 or processed == total_frames:
                print(
                    f"[ScanNet {processed:05d}/{total_frames:05d}] {status:5s} "
                    f"{scene.name}/{key} GT={values['gt_valid_pixels']} "
                    f"BIM={values['bim_hit_pixels']}",
                    flush=True,
                )
        scene_statistics[scene.name] = statistics
        # Do not retain scene raycasters while moving through all 1513 scans.
        del syncbim

    records.sort(key=lambda record: record["id"])
    manifest_dir = output_root / "manifests"
    write_jsonl(manifest_dir / "train.jsonl", records)
    write_jsonl(manifest_dir / "val.jsonl", [])
    write_jsonl(manifest_dir / "test.jsonl", [])
    write_jsonl(manifest_dir / "all.jsonl", records)
    metadata = {
        "schema": "MyDepth S23PriorBIMDataset",
        "prior": "ScanNet SyncBIM",
        "split": "train-only",
        "scannet_root": str(dataset.root),
        "scenes": [scene.name for scene in scenes],
        "target_shape": list(TARGET_SHAPE),
        "preprocessing": "registered depth grid -> aspect-preserving resize -> center crop",
        "samples": len(records),
        "frame_stride": args.frame_stride,
        "minimum_bim_hit_fraction": args.min_bim_hit_fraction,
        "syncbim_sample_points": args.syncbim_sample_points,
        "fill_plane": args.fill_plane,
        "axis_aligned": dataset.axis_align,
        "invalid_poses_skipped": dataset.invalid_pose_count,
        "scene_statistics": scene_statistics,
    }
    atomic_write_text(
        metadata_path,
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
    )
    print(f"Wrote {len(records)} train-only samples to {manifest_dir / 'train.jsonl'}")


if __name__ == "__main__":
    main()
