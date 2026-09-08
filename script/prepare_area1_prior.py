#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Local repositories
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRIORBIMDA_SRC = PROJECT_ROOT / "src"
if str(PRIORBIMDA_SRC) not in sys.path:
    sys.path.insert(0, str(PRIORBIMDA_SRC))

TOOLKIT_ROOT = Path(
    os.environ.get(
        "S3_SAM3D_TOOLKIT_ROOT",
        PROJECT_ROOT.parent / "S3-SAM3D-ToolKit",
    )
).expanduser().resolve()

if (TOOLKIT_ROOT / "src").is_dir():
    toolkit_src = TOOLKIT_ROOT / "src"
    if str(toolkit_src) not in sys.path:
        sys.path.insert(0, str(toolkit_src))


from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data.preparation import DA3PredictionProvider
from bim_priorda3.data.splits import resolve_annotation_splits

from s3dis_sam3d.bimsync import (
    BIMSyncDataset,
    STRUCTURAL_IFC_TYPES,
)
from s3dis_sam3d.rendering import MeshRaycaster
from s3dis_sam3d.s23dis import S23Dataset


# Include a couple of structural IFC classes not currently present in the
# toolkit default. Furniture/FurnishingElement remains excluded.
AREA1_IFC_TYPES = frozenset(
    STRUCTURAL_IFC_TYPES
    | {
        "IfcCurtainWall",
        "IfcRoof",
    }
)

PROTOCOL = "s3-sam3d-area1-global-structural-hit-only-v1"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the minimal Area_1 dataset required by "
            "PriorBIMDA BIMDomainDataset."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "stanford_area1_priorda_v11_bim_domain_6plus12epoch_20260908.yaml",
    )

    parser.add_argument(
        "--s23-root",
        type=Path,
        default=None,
        help=(
            "2D-3D-S no_xyz root. "
            "Defaults to data.source_root in the config."
        ),
    )

    parser.add_argument(
        "--bimsync-root",
        type=Path,
        default=PROJECT_ROOT.parent / "BIMSyn",
        help="Root containing BIM_model/ifc or the IFC files.",
    )

    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=(
            TOOLKIT_ROOT
            / "dataset"
            / "ifc_to_s3dis"
            / "Area_1"
        ),
        help=(
            "Directory containing "
            "*_ifc_to_s3dis_transform.npy."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Prepared dataset root. "
            "Defaults to data.processed_root."
        ),
    )

    parser.add_argument(
        "--da3-cache-root",
        type=Path,
        default=None,
        help=(
            "Optional existing schema-v2 DA3 cache. "
            "Missing predictions are inferred locally."
        ),
    )

    parser.add_argument(
        "--da3-process-res",
        type=int,
        default=504,
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the pinned DA3 model to already exist locally.",
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


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def atomic_savez(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)

    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary, path)

    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary, path)

    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------
# Frame convention
# ---------------------------------------------------------------------

def canonical_frame_key(frame) -> str:
    """
    PriorBIMDA convention:

        camera_xxx_office_1_frame_0_domain_rgb.png
                    ↓
        camera_xxx_office_1_frame_0
    """

    suffix = "_domain_rgb.png"
    name = frame.rgb_path.name

    if not name.endswith(suffix):
        raise ValueError(
            f"Unexpected regular 2D-3D-S RGB filename: {name}"
        )

    return name[:-len(suffix)]


# ---------------------------------------------------------------------
# Existing sample validation
# ---------------------------------------------------------------------

def validate_existing_sample(
    path: Path,
    target_shape: tuple[int, int],
) -> None:

    with np.load(path, allow_pickle=False) as item:

        required = {
            "intrinsic",
            "base_depth",
            "bim_depth",
            "bim_valid",
        }

        missing = sorted(required - set(item.files))

        if missing:
            raise ValueError(
                f"{path}: missing keys {missing}"
            )

        if item["intrinsic"].shape != (3, 3):
            raise ValueError(
                f"{path}: intrinsic must be 3x3"
            )

        for key in (
            "base_depth",
            "bim_depth",
            "bim_valid",
        ):
            if item[key].shape != target_shape:
                raise ValueError(
                    f"{path}: {key} shape="
                    f"{item[key].shape}, "
                    f"expected={target_shape}"
                )

        bim_depth = item["bim_depth"].astype(
            np.float32
        )

        bim_valid = item["bim_valid"] > 0

        if not np.isfinite(bim_depth).all():
            raise ValueError(
                f"{path}: bim_depth contains non-finite values"
            )

        if np.any(bim_depth[~bim_valid] != 0):
            raise ValueError(
                f"{path}: invalid BIM pixels "
                "must have zero depth"
            )


# ---------------------------------------------------------------------
# Global Area_1 BIM
# ---------------------------------------------------------------------

def build_global_raycaster(
    bimsync: BIMSyncDataset,
    rooms: list[str],
) -> MeshRaycaster:

    meshes = []

    for index, room in enumerate(
        rooms,
        start=1,
    ):

        scene = bimsync.scene(room)

        if not scene.is_calibrated:
            raise ValueError(
                f"{room}: IFC calibration is missing"
            )

        calibration = scene.calibration

        if (
            calibration is None
            or calibration.shape != (4, 4)
            or not np.isfinite(calibration).all()
        ):
            raise ValueError(
                f"{room}: IFC calibration "
                "must be a finite 4x4 matrix"
            )

        print(
            f"[mesh {index:02d}/{len(rooms):02d}] "
            f"{room}",
            flush=True,
        )

        mesh = scene.mesh(
            include_types=AREA1_IFC_TYPES,
            calibrated=True,
        )

        meshes.append(mesh)

    if not meshes:
        raise RuntimeError(
            "No Area_1 IFC meshes were built"
        )

    full_mesh = meshes[0]

    for mesh in meshes[1:]:
        full_mesh += mesh

    full_mesh.remove_duplicated_vertices()
    full_mesh.remove_degenerate_triangles()
    full_mesh.remove_unreferenced_vertices()

    return MeshRaycaster(full_mesh)


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------

def manifest_record(
    *,
    frame,
    room: str,
    sample_path: Path,
    source_root: Path,
    output_root: Path,
    da3_source: str,
    bim_hit_pixels: int,
) -> dict:

    key = canonical_frame_key(frame)

    image_path = (
        frame.rgb_path
        .expanduser()
        .resolve()
    )

    pose_path = (
        frame.pose_path
        .expanduser()
        .resolve()
    )

    try:
        image_relative = (
            image_path.relative_to(source_root)
        )
    except ValueError:
        image_relative = (
            Path("area_1")
            / "data"
            / "rgb"
            / image_path.name
        )

    return {
        "id": f"{room}/{key}",
        "region": room,

        "dataset":
            "Stanford2D3DS/Area_1+BIMSyn",

        "image": str(image_path),

        "image_relative_to_source":
            str(image_relative),

        "sample":
            str(sample_path.resolve()),

        "sample_relative_to_processed":
            str(
                sample_path.relative_to(
                    output_root
                )
            ),

        "pose":
            str(pose_path),

        "camera_uuid":
            str(frame.uuid),

        "frame_number":
            int(frame.frame_id),

        "frame_index":
            int(frame.frame_id),

        "da3_source":
            da3_source,

        "bim_hit_pixels":
            int(bim_hit_pixels),

        "preparation_protocol":
            PROTOCOL,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    if args.da3_process_res < 1:
        raise ValueError(
            "--da3-process-res must be positive"
        )

    if args.log_every < 1:
        raise ValueError(
            "--log-every must be positive"
        )

    cfg = load_config(args.config)

    # -------------------------------------------------------------
    # Resolution
    # -------------------------------------------------------------

    target_shape = (
        int(cfg.data.target_height),
        int(cfg.data.target_width),
    )

    target_height, target_width = target_shape

    if target_shape != (504, 504):
        print(
            "WARNING: current experiment was "
            f"designed for 504x504, got {target_shape}",
            flush=True,
        )

    # -------------------------------------------------------------
    # 2D-3D-S
    # -------------------------------------------------------------

    if args.s23_root is not None:
        source_root = (
            args.s23_root
            .expanduser()
            .resolve()
        )
    else:
        source_root = resolve_project_path(
            cfg,
            cfg.data.source_root,
        )

    area_root = (
        source_root
        / "area_1"
    )

    if not area_root.is_dir():
        raise FileNotFoundError(
            "2D-3D-S Area_1 directory "
            f"not found: {area_root}"
        )

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    if args.output_root is not None:
        output_root = (
            args.output_root
            .expanduser()
            .resolve()
        )
    else:
        output_root = resolve_project_path(
            cfg,
            cfg.data.processed_root,
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # BIM
    # -------------------------------------------------------------

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

    # Important:
    # reproduce canonical PriorBIMDA manifest ordering.
    regions = sorted(
        str(room)
        for room in cfg.data.regions
    )

    if len(regions) != len(set(regions)):
        raise ValueError(
            "data.regions contains duplicate rooms"
        )

    s23 = S23Dataset(
        area_path=area_root,
        projection_type="regular",
    )

    bimsync = BIMSyncDataset(
        root=bimsync_root,
        area="Area_1",
        calibration_dir=calibration_dir,
    )

    s23_rooms = {
        room.name
        for room in s23.rooms
    }

    ifc_rooms = {
        scene.name
        for scene in bimsync.scenes
    }

    missing_s23 = sorted(
        set(regions) - s23_rooms
    )

    missing_ifc = sorted(
        set(regions) - ifc_rooms
    )

    if missing_s23:
        raise ValueError(
            "Configured rooms missing "
            f"from 2D-3D-S: {missing_s23}"
        )

    if missing_ifc:
        raise ValueError(
            "Configured rooms missing "
            f"from IFC: {missing_ifc}"
        )

    # -------------------------------------------------------------
    # Build ONE fixed Area_1 BIM
    # -------------------------------------------------------------

    print(
        "Building one fixed Area_1 "
        f"structural BIM from {len(regions)} rooms...",
        flush=True,
    )

    raycaster = build_global_raycaster(
        bimsync,
        regions,
    )

    # -------------------------------------------------------------
    # DA3 provider
    #
    # Current training config does not need these fields because
    # training only consumes the already prepared base_depth.
    # Preparation supplies them explicitly.
    # -------------------------------------------------------------

    if args.da3_cache_root is not None:

        cfg.data.da3_cache_roots = {
            "area_1": str(
                args.da3_cache_root
                .expanduser()
                .resolve()
            )
        }

    else:

        cfg.data.da3_cache_roots = {}

    cfg.data.da3_process_res = (
        int(args.da3_process_res)
    )

    cfg.data.da3_require_pinned_revision = True

    cfg.data.da3_local_files_only = (
        bool(args.local_files_only)
    )

    da3_cache_write = (
        output_root
        / "da3_cache"
        / "area_1"
    )

    provider = DA3PredictionProvider(
        cfg,
        "area_1",
        da3_cache_write,
    )

    # -------------------------------------------------------------
    # Frames
    # -------------------------------------------------------------

    records = []
    processed = 0

    for room_index, room in enumerate(
        regions,
        start=1,
    ):

        s23_room = s23.room(room)

        # S23Room normally sorts by frame_id/uuid.
        # PriorBIMDA historically sorted by the original frame key.
        # Preserve that ordering for split reproducibility.
        frames = sorted(
            s23_room.frames,
            key=canonical_frame_key,
        )

        sample_dir = (
            output_root
            / "samples"
            / room
        )

        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"[room {room_index:02d}/"
            f"{len(regions):02d}] "
            f"{room}: {len(frames)} frames",
            flush=True,
        )

        for frame_index, frame in enumerate(
            frames,
            start=1,
        ):

            key = canonical_frame_key(
                frame
            )

            sample_path = (
                sample_dir
                / f"{key}.npz"
            )

            # -------------------------------------------------
            # Reuse
            # -------------------------------------------------

            if (
                sample_path.exists()
                and not args.overwrite
            ):

                validate_existing_sample(
                    sample_path,
                    target_shape,
                )

                with np.load(
                    sample_path,
                    allow_pickle=False,
                ) as item:

                    bim_hit_pixels = int(
                        np.count_nonzero(
                            item["bim_valid"]
                        )
                    )

                da3_source = "existing"

            # -------------------------------------------------
            # Generate
            # -------------------------------------------------

            else:

                prediction = (
                    provider.get_with_provenance(
                        frame.rgb_path,
                        target_shape,
                    )
                )

                # IMPORTANT:
                #
                # Save raw canonical-focal DA3Metric depth.
                #
                # Do NOT apply:
                #
                #   mean(fx, fy) / 300
                #
                # here.
                #
                # Dense4/BIMDomainDataset applies that correction
                # during loading.

                base_depth = np.asarray(
                    prediction.depth,
                    dtype=np.float32,
                )

                if (
                    base_depth.shape
                    != target_shape
                ):
                    raise ValueError(
                        f"{room}/{key}: "
                        f"DA3 shape "
                        f"{base_depth.shape} "
                        f"!= {target_shape}"
                    )

                if (
                    not np.isfinite(
                        base_depth
                    ).all()
                    or np.any(
                        base_depth <= 0
                    )
                ):
                    raise ValueError(
                        f"{room}/{key}: "
                        "DA3 depth must be "
                        "finite and positive"
                    )

                # -------------------------------------------------
                # Correct 504x504 camera intrinsics
                # -------------------------------------------------

                intrinsic = np.asarray(
                    frame.intrinsics_for_size(
                        target_shape
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
                        f"{room}/{key}: "
                        "invalid scaled intrinsics"
                    )

                # -------------------------------------------------
                # Global Area_1 BIM raycast
                # -------------------------------------------------

                bim_depth = raycaster.depth(
                    intrinsic,
                    frame.world_to_camera,
                    target_width,
                    target_height,
                ).astype(
                    np.float32
                )

                # Hit-only BIM support.
                # No GT filtering.
                # No 0.2-5 m cutoff.
                # No confidence/trust filtering.

                bim_valid = (
                    np.isfinite(bim_depth)
                    & (bim_depth > 0)
                )

                bim_depth[
                    ~bim_valid
                ] = 0.0

                # -------------------------------------------------
                # Minimal current BIMDomainDataset schema
                # -------------------------------------------------

                atomic_savez(
                    sample_path,

                    intrinsic=(
                        intrinsic.astype(
                            np.float32
                        )
                    ),

                    base_depth=(
                        base_depth.astype(
                            np.float16
                        )
                    ),

                    bim_depth=(
                        bim_depth.astype(
                            np.float16
                        )
                    ),

                    bim_valid=(
                        bim_valid.astype(
                            np.uint8
                        )
                    ),
                )

                bim_hit_pixels = int(
                    bim_valid.sum()
                )

                da3_source = (
                    prediction.source
                )

            # -------------------------------------------------
            # Manifest
            # -------------------------------------------------

            records.append(
                manifest_record(
                    frame=frame,
                    room=room,
                    sample_path=sample_path,
                    source_root=source_root,
                    output_root=output_root,
                    da3_source=da3_source,
                    bim_hit_pixels=bim_hit_pixels,
                )
            )

            processed += 1

            if (
                frame_index == 1
                or frame_index
                % args.log_every == 0
                or frame_index
                == len(frames)
            ):

                print(
                    f"  [{frame_index:04d}/"
                    f"{len(frames):04d}] "
                    f"{key} | "
                    f"BIM hits="
                    f"{bim_hit_pixels}",
                    flush=True,
                )

    # -------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------

    manifest_path = (
        output_root
        / "manifest.jsonl"
    )

    manifest_text = "".join(
        json.dumps(
            record,
            sort_keys=True,
        )
        + "\n"
        for record in records
    )

    atomic_write_text(
        manifest_path,
        manifest_text,
    )

    # -------------------------------------------------------------
    # Verify population against immutable room split annotation
    # -------------------------------------------------------------

    annotation_path = resolve_project_path(
        cfg,
        cfg.data.split_annotation,
    )

    split_resolution = (
        resolve_annotation_splits(
            records,
            annotation_path,
        )
    )

    actual_split_fingerprint = (
        split_resolution
        .provenance[
            "fingerprint_sha256"
        ]
    )

    expected_split_fingerprint = str(
        cfg.data.get(
            "split_fingerprint_sha256",
            "",
        )
    ).strip()

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,

        "area": "Area_1",

        "target_shape":
            list(target_shape),

        "record_count":
            len(records),

        "rooms":
            regions,

        "ifc_types":
            sorted(AREA1_IFC_TYPES),

        "manifest":
            str(manifest_path),

        "split_annotation":
            str(annotation_path),

        "split_annotation_sha256":
            split_resolution
            .provenance[
                "annotation_raw_sha256"
            ],

        "split_fingerprint_sha256":
            actual_split_fingerprint,

        "configured_split_fingerprint_sha256":
            (
                expected_split_fingerprint
                or None
            ),

        "da3_model":
            str(cfg.data.da3_model),

        "da3_revision":
            str(cfg.data.da3_revision),

        "da3_process_res":
            int(args.da3_process_res),
    }

    atomic_write_text(
        output_root
        / "preparation_summary.json",

        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    # -------------------------------------------------------------
    # Done
    # -------------------------------------------------------------

    print()

    print(
        f"Prepared {processed} "
        "Area_1 samples",
        flush=True,
    )

    print(
        f"manifest: {manifest_path}",
        flush=True,
    )

    print(
        "split fingerprint: "
        f"{actual_split_fingerprint}",
        flush=True,
    )

    if (
        expected_split_fingerprint
        and expected_split_fingerprint
        != actual_split_fingerprint
    ):

        print(
            "\nNOTE: the new local preparation "
            "does not have the same dataset "
            "fingerprint as the currently pinned "
            "materialization.\n"
            "\nSet:\n"
            "\n"
            "data:\n"
            "  split_fingerprint_sha256: "
            f"{actual_split_fingerprint}\n"
            "\n"
            "before training with this dataset.",
            flush=True,
        )


if __name__ == "__main__":
    main()