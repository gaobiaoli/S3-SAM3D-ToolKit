"""Recover the frozen v1 reference from Livox ROS1 frames and SLAM-global PCDs.

No additional map/BIM or photometric calibration is fitted here.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import closing, contextmanager
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from ..config import slabim_registration_dir
from .poses import (
    interpolate_camera_poses,
    load_pose_table,
    map_to_bim,
    save_pose_table,
    smooth_poses,
)

METHOD_PARAMETERS = {
    "voxel_m": 0.2,
    "first_frame_schedule": [[1.0, 60], [0.5, 60], [0.2, 80], [0.1, 80]],
    "later_schedule": [[0.35, 25]],
    "max_time_difference_s": 0.01,
    "smoothing_window": 5,
}
RESULT_FILES = ("lidar_pose_to_bim.txt", "camera_pose_to_bim.txt", "camera_frames.txt")
# Preserve the original v1 benchmark selection; do not silently add these frames.
EXCLUDED_IMAGES = {"4F_Region3": {40, 42, 44, 46, 48, 50, 54}}
_HEADER_LOCK = Lock()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summary(output):
    path = output / "summary.json"
    if not path.exists():
        return {"version": "v1", "method_parameters": METHOD_PARAMETERS, "regions": {}}
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("version") != "v1" or summary.get("method_parameters") != METHOD_PARAMETERS:
        raise ValueError("Results are not frozen SLABIM v1; use a separate output directory")
    return summary


def _indexed_files(directory, suffix, count):
    paths = sorted(directory.glob(f"*{suffix}"), key=lambda path: int(path.stem))
    if [int(path.stem) for path in paths] != list(range(count)):
        raise ValueError(f"File indices do not align with timestamps: {directory}")
    return paths


def _source_hashes(root, source):
    return {
        "lidar_timestamps": _sha256(source / "points/timestamps.txt"),
        "map_to_bim": _sha256(source / "points/pose_frame_to_bim.txt"),
        "camera_timestamps": _sha256(source / "images/timestamps.txt"),
        "camera_to_lidar": _sha256(root / "calibration_files/cam_to_lidar.txt"),
    }


def _fingerprint(source, pcds, bags, hashes):
    # Stat large immutable sensor files; hash the small calibration/time files.
    stats = hashlib.sha256()
    for path in [*pcds, *bags]:
        stat = path.stat()
        stats.update(f"{path.relative_to(source)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return {
        "source_sha256": hashes,
        "sensor_stat_sha256": stats.hexdigest(),
        "method_parameters": METHOD_PARAMETERS,
    }


@contextmanager
def _bag_header_compat():
    """Accept SLABIM's two-space ROS1 magic while opening a bag, without editing it."""
    import rosbags.rosbag1.reader as reader_module

    def match(pattern, string, *args, **kwargs):
        if pattern == r"#ROSBAG V(\d+).(\d+)\n":
            pattern = r"#ROSBAG +V(\d+)\.(\d+)\n"
        return re.match(pattern, string, *args, **kwargs)

    with _HEADER_LOCK:
        original = reader_module.re
        reader_module.re = SimpleNamespace(match=match)
        try:
            yield
        finally:
            reader_module.re = original


def _livox_messages(bags):
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:
        raise ImportError(
            'Pose recovery requires pip install -e ".[slabim]" (Python >=3.10)'
        ) from error

    for bag in bags:
        reader = AnyReader([bag])
        with _bag_header_compat():
            reader.open()
        try:
            connections = [c for c in reader.connections if c.topic == "/livox/lidar"]
            if not connections:
                raise ValueError(f"No /livox/lidar topic: {bag}")
            for connection, _, data in reader.messages(connections=connections):
                message = reader.deserialize(data, connection.msgtype)
                points = np.array([(p.x, p.y, p.z) for p in message.points], dtype=float)
                yield float(message.timebase) / 1e9, points
        finally:
            reader.close()


def _read_points(path):
    import open3d as o3d

    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def _register_frame(local, target, estimate, first):
    import open3d as o3d

    clouds = []
    for points in (local, target):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.remove_non_finite_points()
        cloud = cloud.voxel_down_sample(METHOD_PARAMETERS["voxel_m"])
        if not cloud.has_points():
            raise ValueError("Empty LiDAR/SLAM point cloud")
        clouds.append(cloud)
    schedule = METHOD_PARAMETERS["first_frame_schedule" if first else "later_schedule"]
    for distance, iterations in schedule:
        fit = o3d.pipelines.registration.registration_icp(
            clouds[0],
            clouds[1],
            distance,
            estimate,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
        )
        estimate = fit.transformation
    if fit.fitness <= 0 or not np.isfinite(estimate).all() or not np.isfinite(fit.inlier_rmse):
        raise RuntimeError("ICP failed to recover a valid pose")
    return estimate, float(fit.fitness), float(fit.inlier_rmse)


def _camera_samples(source, times, region):
    image_times = np.loadtxt(source / "images/timestamps.txt", ndmin=1)
    if not np.isfinite(image_times).all() or not np.all(np.diff(image_times) > 0):
        raise ValueError("Invalid image timestamps")
    selected, excluded = [], []
    images = sorted((source / "images/data").glob("*.png"), key=lambda p: int(p.stem))
    if len({int(p.stem) for p in images}) != len(images):
        raise ValueError("Duplicate image indices")
    for path in images:
        index = int(path.stem)
        if not 0 <= index < len(image_times):
            raise ValueError(f"Image lacks timestamp: {path}")
        stamp = float(image_times[index])
        if index in EXCLUDED_IMAGES.get(region, set()):
            reason = "historical v1 exclusion"
        elif stamp < times[0] or stamp > times[-1]:
            reason = "outside LiDAR time range"
        else:
            selected.append((path.name, stamp))
            continue
        excluded.append({"file": path.name, "reason": reason})
    if not selected:
        raise ValueError(f"No camera frames covered by the LiDAR trajectory: {region}")
    return selected, excluded


def _save_checkpoint(path, fingerprint, transforms, recovered, fitness, rmse):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        fingerprint=json.dumps(fingerprint, sort_keys=True),
        transforms=transforms,
        recovered=recovered,
        fitness=fitness,
        rmse=rmse,
    )
    temporary.replace(path)


def _recover_lidar(times, pcds, bags, checkpoint, fingerprint):
    count = len(times)
    transforms = np.repeat(np.eye(4)[None], count, axis=0)
    recovered = np.zeros(count, dtype=bool)
    fitness, rmse = np.full(count, np.nan), np.full(count, np.nan)
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if json.loads(str(saved["fingerprint"])) != fingerprint:
                raise ValueError("Checkpoint source fingerprint changed; remove it to restart")
            transforms, recovered, fitness, rmse = (
                saved[name].copy() for name in ("transforms", "recovered", "fitness", "rmse")
            )
        if (
            transforms.shape != (count, 4, 4)
            or recovered.shape != (count,)
            or recovered.dtype != np.bool_
            or fitness.shape != (count,)
            or rmse.shape != (count,)
            or not np.isfinite(transforms).all()
            or not np.isfinite(fitness[recovered]).all()
            or not np.isfinite(rmse[recovered]).all()
        ):
            raise ValueError("Invalid recovery checkpoint")
    done = int(recovered.sum())
    if done == count:
        return transforms, fitness, rmse
    estimate = np.eye(4)
    try:
        with (
            closing(_livox_messages(bags)) as messages,
            tqdm(total=count, initial=done, desc=checkpoint.stem) as progress,
        ):
            for stamp, local in messages:
                right = int(np.searchsorted(times, stamp))
                choices = [i for i in (right - 1, right) if 0 <= i < count]
                index = min(choices, key=lambda i: abs(times[i] - stamp))
                if recovered[index] or abs(times[index] - stamp) > 0.01:
                    continue
                if index > 0 and recovered[index - 1]:
                    estimate = transforms[index - 1]
                estimate, fit, error = _register_frame(
                    local, _read_points(pcds[index]), estimate, first=done == 0
                )
                transforms[index], fitness[index], rmse[index] = estimate, fit, error
                recovered[index] = True
                done += 1
                progress.update()
                if done % 100 == 0:
                    _save_checkpoint(checkpoint, fingerprint, transforms, recovered, fitness, rmse)
                if done == count:
                    break
    finally:
        _save_checkpoint(checkpoint, fingerprint, transforms, recovered, fitness, rmse)
    if not recovered.all():
        raise RuntimeError(f"{int((~recovered).sum())} LiDAR poses missing; checkpoint retained")
    return transforms, fitness, rmse


def recover_region(root, region, output_dir=None):
    """Recover one region with frozen v1 parameters, or verify and reuse its result.

    ``root`` contains ``sensor_data`` and ``calibration_files``. Raw data is read-only.
    Successful output has three text files per region and one shared summary.
    Incomplete recovery keeps one resumable checkpoint in ``output_dir/.cache``.
    """
    root = Path(root).expanduser().resolve()
    output = Path(output_dir or slabim_registration_dir()).expanduser().resolve()
    raw = (root / "sensor_data").resolve()
    if output == raw or raw in output.parents or output in raw.parents:
        raise ValueError("Raw data and results must use separate directories")
    if not re.fullmatch(r"\d+F_Region\d+", region):
        raise ValueError(f"Invalid region name: {region}")
    source = raw / region
    summary = _summary(output)
    hashes = _source_hashes(root, source)
    if region in summary["regions"]:
        record = summary["regions"][region]
        if record["source_sha256"] != hashes:
            raise ValueError("Source calibration/timestamps changed; use separate results")
        for name in RESULT_FILES:
            if _sha256(output / region / name) != record["sha256"][name]:
                raise ValueError(f"Result checksum mismatch: {region}/{name}")
        return record

    times = np.loadtxt(source / "points/timestamps.txt", ndmin=1)
    if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Invalid LiDAR timestamps")
    pcds = _indexed_files(source / "points/data", ".pcd", len(times))
    bags = sorted((source / "rosbag").glob("*.bag"), key=lambda p: int(p.stem.split("_")[-1]))
    if not bags:
        raise ValueError(f"No ROS1 bags found: {source}")
    map_transform = map_to_bim(source / "points/pose_frame_to_bim.txt", times)
    selected, excluded = _camera_samples(source, times, region)
    extrinsic = np.loadtxt(root / "calibration_files/cam_to_lidar.txt")
    fingerprint = _fingerprint(source, pcds, bags, hashes)
    cache = output / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = cache / f"{region}.npz"
    transforms, fitness, rmse = _recover_lidar(times, pcds, bags, checkpoint, fingerprint)
    lidar = map_transform[None] @ smooth_poses(transforms)
    camera_times = np.array([stamp for _, stamp in selected])
    camera = interpolate_camera_poses(times, lidar, camera_times, extrinsic)
    destination = output / region
    destination.mkdir(exist_ok=True)
    save_pose_table(destination / RESULT_FILES[0], times, lidar)
    save_pose_table(destination / RESULT_FILES[1], camera_times, camera)
    (destination / RESULT_FILES[2]).write_text(
        "\n".join(name for name, _ in selected) + "\n", encoding="utf-8"
    )
    # Validate the final tables before marking the region complete.
    for name in RESULT_FILES[:2]:
        load_pose_table(destination / name)
    record = {
        "lidar_frames": len(times),
        "camera_frames": len(selected),
        "excluded_images": excluded,
        "source_sha256": hashes,
        "median_icp_fitness": float(np.median(fitness)),
        "median_icp_rmse_m": float(np.median(rmse)),
        "sha256": {name: _sha256(destination / name) for name in RESULT_FILES},
    }
    summary["regions"][region] = record
    _write_json(output / "summary.json", summary)
    checkpoint.unlink()
    if not any(cache.iterdir()):
        cache.rmdir()
    return record
