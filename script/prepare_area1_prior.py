#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


#
# export S3_SAM3D_TOOLKIT_ROOT=/home/bgao491/S3-SAM3D-ToolKit

# python script/prepare_area1_prior.py \
#   --s23-root /home/bgao491/Stanford2D3DS/no_xyz \
#   --bimsync-root /home/bgao491/BIMSyn \
#   --calibration-dir /home/bgao491/S3-SAM3D-ToolKit/dataset/ifc_to_s3dis/Area_1_upright_v2 \
#   --output-root /mnt/priorbimda-data/area1_priorbimda_504


# ================================================================
# Optional local S3-SAM3D-ToolKit checkout
# ================================================================

TOOLKIT_ROOT = os.environ.get("S3_SAM3D_TOOLKIT_ROOT")

if TOOLKIT_ROOT:
    toolkit_src = Path(TOOLKIT_ROOT).expanduser().resolve() / "src"
    if str(toolkit_src) not in sys.path:
        sys.path.insert(0, str(toolkit_src))

from s3dis_sam3d.mde import (
    DA3_CANONICAL_FOCAL,
    DA3Predictor,
)
from s3dis_sam3d.s23_bim import S23_BIMDataset


# ================================================================
# Fixed PriorBIMDA Area_1 protocol
# ================================================================

AREA = "Area_1"
TARGET_SHAPE = (504, 504)

DA3_REFERENCE_FOCAL = DA3_CANONICAL_FOCAL


# Exact room-disjoint split used by PriorBIMDA.
ROOM_SPLITS = {
    "train": (
        "WC_1",
        "conferenceRoom_2",
        "copyRoom_1",
        "hallway_1",
        "hallway_3",
        "hallway_4",
        "hallway_6",
        "hallway_7",
        "office_1",
        "office_2",
        "office_3",
        "office_4",
        "office_5",
        "office_7",
        "office_8",
        "office_9",
        "office_10",
        "office_11",
        "office_12",
        "office_13",
        "office_15",
        "office_17",
        "office_18",
        "office_19",
        "office_21",
        "office_25",
        "office_26",
        "office_27",
        "office_29",
        "pantry_1",
    ),
    "val": (
        "hallway_2",
        "hallway_5",
        "office_6",
        "office_14",
        "office_24",
        "office_28",
        "office_31",
    ),
    "test": (
        "conferenceRoom_1",
        "hallway_8",
        "office_16",
        "office_20",
        "office_22",
        "office_23",
        "office_30",
    ),
}

EXPECTED_FRAME_COUNTS = {
    "train": 7013,
    "val": 1673,
    "test": 1641,
}

EXPECTED_TOTAL_FRAMES = 10327

_NEAREST = getattr(
    cv2,
    "INTER_NEAREST_EXACT",
    cv2.INTER_NEAREST,
)


# ================================================================
# CLI
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Stanford 2D-3D-S Area_1 + BIMSyn data "
            "for PriorBIMDA-style training."
        )
    )

    parser.add_argument(
        "--s23-root",
        type=Path,
        required=True,
        help=(
            "Stanford2D3DS/no_xyz root. "
            "It should contain area_1/."
        ),
    )

    parser.add_argument(
        "--bimsync-root",
        type=Path,
        required=True,
        help="BIMSyn / IFC root used by BIMSyncDataset.",
    )

    parser.add_argument(
        "--calibration-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing "
            "*_ifc_to_s3dis_transform.npy for Area_1."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="e.g. cuda, cuda:0, cpu. Default: CUDA if available.",
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the pinned DA3 checkpoint to exist locally.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    return parser.parse_args()


# ================================================================
# General I/O
# ================================================================

def atomic_savez(path: Path, **payload):
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(
                handle,
                **payload,
            )
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_path, path)

    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_path, path)

    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, records):
    text = "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in records
    )
    atomic_write_text(path, text)


# ================================================================
# Stanford frame convention
# ================================================================

def canonical_frame_key(frame):
    """
    camera_xxx_office_1_frame_0_domain_rgb.png
                        ->
    camera_xxx_office_1_frame_0
    """

    name = frame.rgb_path.name
    suffix = "_domain_rgb.png"

    if not name.endswith(suffix):
        raise ValueError(
            f"Unexpected Stanford RGB filename: {name}"
        )

    return name[:-len(suffix)]


def relative_path(path: Path, root: Path):
    path = path.expanduser().resolve()
    root = root.expanduser().resolve()

    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# ================================================================
# Official 2D-3D-S depth
# ================================================================

def load_gt_depth(
    depth_path: Path,
    target_shape=TARGET_SHAPE,
):
    """
    Match PriorBIMDA official_all_valid protocol:

    depth = uint16 / 512
    65535 = invalid
    every other positive depth = valid

    Resize with nearest-neighbour.
    """

    raw = cv2.imread(
        str(depth_path),
        cv2.IMREAD_UNCHANGED,
    )

    if raw is None:
        raise RuntimeError(
            f"Cannot read depth: {depth_path}"
        )

    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError(
            f"Expected uint16 depth PNG, got "
            f"{raw.dtype} {raw.shape}: {depth_path}"
        )

    height, width = target_shape

    if raw.shape != target_shape:
        raw = cv2.resize(
            raw,
            (width, height),
            interpolation=_NEAREST,
        )

    depth = (
        raw.astype(np.float32)
        / 512.0
    )

    valid = (
        (raw != np.uint16(65535))
        & np.isfinite(depth)
        & (depth > 0.0)
    )

    depth[~valid] = 0.0

    return (
        depth.astype(np.float32),
        valid,
    )


# ================================================================
# Prepared sample validation
# ================================================================

def validate_sample(path: Path):
    with np.load(
        path,
        allow_pickle=False,
    ) as item:

        required = {
            "sample_schema_version",
            "intrinsic",
            "da3_depth_raw",
            "da3_focal_scale",
            "bim_depth",
            "bim_valid",
            "gt_depth",
            "gt_valid",
        }

        missing = required - set(item.files)

        if missing:
            raise ValueError(
                f"{path}: missing {sorted(missing)}"
            )

        if item["intrinsic"].shape != (3, 3):
            raise ValueError(
                f"{path}: invalid intrinsic"
            )

        for key in (
            "da3_depth_raw",
            "bim_depth",
            "bim_valid",
            "gt_depth",
            "gt_valid",
        ):
            if item[key].shape != TARGET_SHAPE:
                raise ValueError(
                    f"{path}: {key} has shape "
                    f"{item[key].shape}"
                )

        bim_valid = item["bim_valid"] > 0
        bim_depth = item["bim_depth"]

        if np.any(
            bim_depth[~bim_valid] != 0
        ):
            raise ValueError(
                f"{path}: invalid BIM depth "
                "pixels must be zero"
            )


# ================================================================
# Dataset integrity
# ================================================================

def collect_frames(dataset):
    records = []

    available_s23 = {
        scene.name
        for scene in dataset.s23_dataset.scenes
    }

    available_bim = {
        name.casefold()
        for name in dataset.scene_ids
    }

    required_rooms = {
        room
        for rooms in ROOM_SPLITS.values()
        for room in rooms
    }

    missing_s23 = sorted(
        required_rooms - available_s23
    )

    missing_bim = sorted(
        room
        for room in required_rooms
        if room.casefold() not in available_bim
    )

    if missing_s23:
        raise ValueError(
            f"Missing S23 rooms: {missing_s23}"
        )

    if missing_bim:
        raise ValueError(
            f"Missing calibrated BIM rooms: "
            f"{missing_bim}"
        )

    camera_owners = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        for room in ROOM_SPLITS[split]:
            s23_scene = (
                dataset.s23_dataset.get_scene(room)
            )

            for frame in s23_scene.frames:
                if not frame.has_depth:
                    continue

                records.append(
                    (split, room, frame)
                )

                camera_owners.setdefault(
                    frame.uuid,
                    set(),
                ).add(split)

    leaked = {
        camera: owners
        for camera, owners
        in camera_owners.items()
        if len(owners) > 1
    }

    if leaked:
        raise ValueError(
            "Camera UUID leakage across splits: "
            f"{list(leaked.items())[:10]}"
        )

    counts = {
        split: sum(
            record_split == split
            for record_split, _, _
            in records
        )
        for split in (
            "train",
            "val",
            "test",
        )
    }

    if counts != EXPECTED_FRAME_COUNTS:
        raise ValueError(
            "Area_1 frame population differs "
            "from PriorBIMDA.\n"
            f"Expected: {EXPECTED_FRAME_COUNTS}\n"
            f"Found:    {counts}"
        )

    if len(records) != EXPECTED_TOTAL_FRAMES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_FRAMES} "
            f"frames, got {len(records)}"
        )

    return records


# ================================================================
# Main
# ================================================================

def main():
    args = parse_args()

    if args.log_every < 1:
        raise ValueError(
            "--log-every must be positive"
        )

    s23_root = (
        args.s23_root
        .expanduser()
        .resolve()
    )

    bimsync_root = (
        args.bimsync_root
        .expanduser()
        .resolve()
    )

    calibration_dir = (
        args.calibration_dir
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Paired Area_1 dataset
    #
    # min_* values are irrelevant here because we deliberately
    # do NOT use S23_BIMDataset's filtered sample index.
    # Every official frame is retained.
    # ------------------------------------------------------------

    paired = S23_BIMDataset(
        area=AREA,
        s23_root=s23_root,
        bimsync_root=bimsync_root,
        calibration_dir=calibration_dir,
        min_gt_valid_fraction=0.0,
        min_bim_hit_fraction=0.0,
        full_scene=True,
    )

    frames = collect_frames(paired)

    print(
        "Area_1 population verified:",
        {
            key: EXPECTED_FRAME_COUNTS[key]
            for key in (
                "train",
                "val",
                "test",
            )
        },
        flush=True,
    )

    # ------------------------------------------------------------
    # DA3 is loaded lazily.
    # Existing samples therefore do not trigger model loading.
    # ------------------------------------------------------------

    da3 = DA3Predictor(
        device=args.device,
        local_files_only=(
            args.local_files_only
        ),
    )

    manifests = {
        "train": [],
        "val": [],
        "test": [],
    }

    total = len(frames)

    for global_index, (
        split,
        room,
        frame,
    ) in enumerate(
        frames,
        start=1,
    ):

        key = canonical_frame_key(frame)

        sample_id = f"{room}/{key}"

        sample_path = (
            output_root
            / "samples"
            / room
            / f"{key}.npz"
        )

        # --------------------------------------------------------
        # Reuse
        # --------------------------------------------------------

        if (
            sample_path.exists()
            and not args.overwrite
        ):
            validate_sample(sample_path)

            status = "reuse"

            with np.load(
                sample_path,
                allow_pickle=False,
            ) as item:
                gt_valid_pixels = int(
                    (item["gt_valid"] > 0).sum()
                )
                bim_hit_pixels = int(
                    (item["bim_valid"] > 0).sum()
                )
                focal_scale = float(
                    item[
                        "da3_focal_scale"
                    ].item()
                )

        else:
            # ----------------------------------------------------
            # Intrinsic at exactly 504 x 504
            # ----------------------------------------------------

            intrinsic = np.asarray(
                frame.intrinsics_for_size(
                    TARGET_SHAPE
                ),
                dtype=np.float32,
            )

            if (
                intrinsic.shape != (3, 3)
                or not np.isfinite(
                    intrinsic
                ).all()
            ):
                raise ValueError(
                    f"{sample_id}: invalid K"
                )

            focal_px = float(
                (
                    intrinsic[0, 0]
                    + intrinsic[1, 1]
                )
                / 2.0
            )

            focal_scale = (
                focal_px
                / DA3_REFERENCE_FOCAL
            )

            if (
                not np.isfinite(
                    focal_scale
                )
                or focal_scale <= 0
            ):
                raise ValueError(
                    f"{sample_id}: invalid "
                    "DA3 focal scale"
                )

            # ----------------------------------------------------
            # DA3
            #
            # Keep RAW DA3 value here.
            #
            # Training:
            #
            #   D_DA3_metric =
            #       da3_depth_raw
            #       * da3_focal_scale
            #
            # This matches the old PriorBIMDA convention.
            # ----------------------------------------------------

            da3_depth_raw = (
                da3.predict_raw(
                    frame.rgb_path,
                    TARGET_SHAPE,
                )
            )

            # ----------------------------------------------------
            # Global Area_1 structural BIM
            #
            # full_scene=True:
            # all 44 calibrated structural IFC rooms are merged
            # into one fixed Area_1 scene.
            #
            # No GT filtering.
            # No per-frame alignment.
            # No trust filtering.
            # ----------------------------------------------------

            bim_scene = (
                paired.bimsync_dataset.scene(
                    room
                )
            )

            bim_depth = np.asarray(
                paired._render_depth(
                    bim_scene,
                    frame,
                    size=TARGET_SHAPE,
                ),
                dtype=np.float32,
            )

            if (
                bim_depth.shape
                != TARGET_SHAPE
            ):
                raise ValueError(
                    f"{sample_id}: BIM shape "
                    f"{bim_depth.shape}"
                )

            bim_valid = (
                np.isfinite(bim_depth)
                & (bim_depth > 0.0)
            )

            bim_depth = np.where(
                bim_valid,
                bim_depth,
                0.0,
            ).astype(np.float32)

            # ----------------------------------------------------
            # Official GT
            # ----------------------------------------------------

            gt_depth, gt_valid = (
                load_gt_depth(
                    frame.depth_path,
                    TARGET_SHAPE,
                )
            )

            gt_valid_pixels = int(
                gt_valid.sum()
            )

            bim_hit_pixels = int(
                bim_valid.sum()
            )

            # ----------------------------------------------------
            # Store
            #
            # Keep GT float32.
            # DA3/BIM use float16 as in previous preparation
            # to control Area_1 storage/I/O.
            # ----------------------------------------------------

            atomic_savez(
                sample_path,

                sample_schema_version=np.asarray(
                    1,
                    dtype=np.uint16,
                ),

                intrinsic=intrinsic.astype(
                    np.float32
                ),

                da3_depth_raw=(
                    da3_depth_raw.astype(
                        np.float16
                    )
                ),

                da3_focal_scale=np.asarray(
                    focal_scale,
                    dtype=np.float32,
                ),

                bim_depth=bim_depth.astype(
                    np.float16
                ),

                bim_valid=bim_valid.astype(
                    np.uint8
                ),

                gt_depth=gt_depth.astype(
                    np.float32
                ),

                gt_valid=gt_valid.astype(
                    np.uint8
                ),
            )

            status = "write"

        # --------------------------------------------------------
        # Manifest
        # --------------------------------------------------------

        record = {
            "id": sample_id,
            "split": split,
            "region": room,

            "rgb": relative_path(
                frame.rgb_path,
                s23_root,
            ),

            "sample": str(
                sample_path.relative_to(
                    output_root
                )
            ),

            "pose": relative_path(
                frame.pose_path,
                s23_root,
            ),

            "camera_uuid": str(
                frame.uuid
            ),

            "frame_number": int(
                frame.frame_id
            ),

            "gt_valid_pixels": (
                gt_valid_pixels
            ),

            "bim_hit_pixels": (
                bim_hit_pixels
            ),

            "da3_focal_scale": (
                focal_scale
            ),
        }

        manifests[split].append(record)

        if (
            global_index == 1
            or global_index
            % args.log_every == 0
            or global_index == total
        ):
            print(
                f"[{global_index:05d}/"
                f"{total:05d}] "
                f"{status:5s} "
                f"{split:5s} "
                f"{sample_id} "
                f"GT={gt_valid_pixels} "
                f"BIM={bim_hit_pixels}",
                flush=True,
            )

    # ------------------------------------------------------------
    # Stable ordering
    # ------------------------------------------------------------

    for split in manifests:
        manifests[split].sort(
            key=lambda record: record["id"]
        )

    all_records = sorted(
        [
            record
            for split_records
            in manifests.values()
            for record in split_records
        ],
        key=lambda record: record["id"],
    )

    # ------------------------------------------------------------
    # Final exact-count guard
    # ------------------------------------------------------------

    for split, expected in (
        EXPECTED_FRAME_COUNTS.items()
    ):
        actual = len(
            manifests[split]
        )

        if actual != expected:
            raise RuntimeError(
                f"{split}: expected "
                f"{expected}, got {actual}"
            )

    # ------------------------------------------------------------
    # Manifests
    # ------------------------------------------------------------

    manifest_dir = (
        output_root / "manifests"
    )

    for split in (
        "train",
        "val",
        "test",
    ):
        write_jsonl(
            manifest_dir
            / f"{split}.jsonl",
            manifests[split],
        )

    write_jsonl(
        manifest_dir / "all.jsonl",
        all_records,
    )

    



if __name__ == "__main__":
    main()
