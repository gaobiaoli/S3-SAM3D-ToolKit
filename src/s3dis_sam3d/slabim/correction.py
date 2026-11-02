"""SLABIM v2: one robust 6DoF map-to-BIM correction per region (Claude G1).

Only the global LiDAR map and BIM geometry determine the correction. Both
LiDAR and camera poses are left-multiplied by it, always starting from v1.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import slabim_registration_dir
from .poses import load_pose_table, map_to_bim, save_pose_table
from .recovery import RESULT_FILES, _sha256, _source_hashes, _summary, _write_json

CORRECTION_PARAMETERS = {
    "voxel_m": 0.05,
    "fit_classes": ["walls", "floors", "columns"],
    "distance_gates_m": [0.30, 0.15, 0.08],
    "iterations_per_gate": 6,
    "huber_m": 0.03,
    "dof": 6,
    "max_points": 3_000_000,
    "seed": 0,
    "bootstrap_tile_m": 10.0,
    "bootstrap_resamples": 200,
    "remove_zero_area_triangles": True,
}


class BIMSurface:
    """Closest triangles on a floor's published walls, floors and columns."""

    def __init__(self, directory, nthreads=8):
        import open3d as o3d

        try:
            from plyfile import PlyData
        except ImportError as error:
            raise ImportError('BIM correction requires pip install -e ".[slabim]"') from error

        self.directory = Path(directory).resolve()
        self.nthreads = nthreads
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=nthreads)
        vertices, triangles, labels = [], [], []
        self.source_sha256 = {}
        self.degenerate_triangles_removed = {}
        offset = 0
        # Match the original triangle/quad parser; Open3D can return partial Rhino meshes.
        for name in sorted(CORRECTION_PARAMETERS["fit_classes"]):
            path = self.directory / f"{name}.ply"
            ply = PlyData.read(path)
            v = np.column_stack([ply["vertex"][axis] for axis in ("x", "y", "z")]).astype(
                np.float32
            )
            polygons = ply["face"]["vertex_indices"]
            sizes = np.fromiter(map(len, polygons), dtype=np.int32)
            if not np.isfinite(v).all() or np.any((sizes < 3) | (sizes > 4)):
                raise ValueError(f"Invalid BIM triangle/quad mesh: {path}")
            faces = []
            for size in np.unique(sizes):
                selected = np.stack(polygons[sizes == size]).astype(np.int32)
                for j in range(1, int(size) - 1):
                    faces.append(selected[:, [0, j, j + 1]])
            f = np.concatenate(faces) if faces else np.empty((0, 3), np.int32)
            if len(f) != int(np.sum(sizes - 2)) or (len(f) and (f.min() < 0 or f.max() >= len(v))):
                raise ValueError(f"Invalid BIM face indices: {path}")
            # Zero-area Rhino faces have undefined normals. Check in the same
            # float32 geometry used by raycasting; keep every nondegenerate face.
            cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
            valid = np.linalg.norm(cross, axis=1) > 0
            self.degenerate_triangles_removed[name] = int((~valid).sum())
            f = f[valid]
            vertices.append(v)
            triangles.append(f + offset)
            labels.extend([name] * len(f))
            offset += len(v)
            self.source_sha256[f"{name}.ply"] = _sha256(path)
        v = np.concatenate(vertices).astype(np.float32)
        f = np.concatenate(triangles).astype(np.uint32)
        if not len(f):
            raise ValueError(f"No structural BIM triangles: {directory}")
        self.triangles = len(f)
        self.labels = np.asarray(labels)
        self.scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(f))

    def closest(self, points):
        import open3d as o3d

        result = self.scene.compute_closest_points(
            o3d.core.Tensor(np.asarray(points, np.float32)), nthreads=self.nthreads
        )
        target = result["points"].numpy().astype(float)
        normals = result["primitive_normals"].numpy().astype(float)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-9)
        return target, normals, self.labels[result["primitive_ids"].numpy()]


def _apply(transform, points):
    return points @ transform[:3, :3].T + transform[:3, 3]


def _normal_equations(points, target, normals, center):
    residual = np.sum((points - target) * normals, axis=1)
    jacobian = np.column_stack((np.cross(points - center, normals), normals))
    weights = np.minimum(1.0, CORRECTION_PARAMETERS["huber_m"] / np.abs(residual).clip(1e-9))
    weighted = jacobian * weights[:, None]
    return weighted.T @ jacobian, -weighted.T @ residual


def _step(matrix, vector, center):
    motion = np.linalg.solve(matrix, vector)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(motion[:3]).as_matrix()
    transform[:3, 3] = center - transform[:3, :3] @ center + motion[3:]
    return transform


def _fit(points, surface):
    center = points.mean(axis=0)
    transform = np.eye(4)
    for gate in CORRECTION_PARAMETERS["distance_gates_m"]:
        for _ in range(CORRECTION_PARAMETERS["iterations_per_gate"]):
            current = _apply(transform, points)
            target, normals, _ = surface.closest(current)
            keep = np.linalg.norm(current - target, axis=1) < gate
            matrix, vector = _normal_equations(current[keep], target[keep], normals[keep], center)
            transform = _step(matrix, vector, center) @ transform
            if not np.isfinite(transform).all():
                raise ValueError("Nonfinite whole-region correction")
    return transform, center


def _angles(transform):
    rotation = transform[:3, :3]
    return {
        "rotation_deg": float(np.degrees(Rotation.from_matrix(rotation).magnitude())),
        "tilt_deg": float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1)))),
        "yaw_deg": float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
    }


def _residuals(points, target, normals):
    keep = np.linalg.norm(points - target, axis=1) < 0.30
    residual = np.abs(np.sum((points[keep] - target[keep]) * normals[keep], axis=1)) * 100
    if not len(residual):
        raise ValueError("No map points within 30 cm of the BIM")
    return {
        "points_within_30cm": int(keep.sum()),
        "share_within_30cm": float(keep.mean()),
        "median_cm": float(np.median(residual)),
        "p90_cm": float(np.percentile(residual, 90)),
    }


def _floor_slope(points, corrected, target, normals, labels):
    keep = (
        (labels == "floors")
        & (np.linalg.norm(corrected - target, axis=1) < 0.08)
        & (np.abs(normals[:, 2]) > 0.95)
    )
    result = {"points": int(keep.sum()), "before_deg": None, "after_deg": None}
    if keep.sum() < 3:
        return result
    for key, values in (("before_deg", points), ("after_deg", corrected)):
        xy = values[keep, :2]
        design = np.column_stack((xy - xy.mean(axis=0), np.ones(len(xy))))
        coefficient, _, rank, _ = np.linalg.lstsq(
            design, values[keep, 2] - target[keep, 2], rcond=None
        )
        if rank == 3:
            result[key] = float(np.degrees(np.arctan(np.linalg.norm(coefficient[:2]))))
    return result


def _bootstrap(points, target, normals, center, transform, cameras, rng):
    keep = np.linalg.norm(points - target, axis=1) < 0.08
    selected, target, normals = points[keep], target[keep], normals[keep]
    tile = np.floor(selected[:, :2] / CORRECTION_PARAMETERS["bootstrap_tile_m"]).astype(np.int64)
    _, indices = np.unique(tile, axis=0, return_inverse=True)
    count = int(indices.max()) + 1
    # Summing per-tile normal equations equals weighting every point by its tile's
    # resample count, without allocating 200 copies of the full Jacobian.
    matrices, vectors = [], []
    for i in range(count):
        mask = indices == i
        matrix, vector = _normal_equations(selected[mask], target[mask], normals[mask], center)
        matrices.append(matrix)
        vectors.append(vector)
    matrices, vectors = np.array(matrices), np.array(vectors)
    samples = []
    for _ in range(CORRECTION_PARAMETERS["bootstrap_resamples"]):
        weights = np.bincount(rng.integers(0, count, count), minlength=count)
        try:
            step = _step(np.einsum("i,ijk->jk", weights, matrices), weights @ vectors, center)
        except np.linalg.LinAlgError:
            continue
        candidate = step @ transform
        angles = _angles(candidate)
        shift = np.median(np.linalg.norm(_apply(candidate, cameras) - cameras, axis=1)) * 100
        samples.append([angles["rotation_deg"], angles["tilt_deg"], float(shift)])
    intervals = np.percentile(samples, [2.5, 97.5], axis=0).T.tolist() if samples else [None] * 3
    return {
        "tiles": count,
        "resamples_requested": CORRECTION_PARAMETERS["bootstrap_resamples"],
        "resamples_valid": len(samples),
        "rotation_deg_95": intervals[0],
        "tilt_deg_95": intervals[1],
        "camera_shift_cm_median_95": intervals[2],
    }


def _overlap(first, second):
    return first == second or first in second.parents or second in first.parents


def correct_region(root, region, reference_dir=None, output_dir=None, *, surface=None):
    """Create or verify v2 from frozen v1; never refit individual sensor frames.

    Output contains the same three text files as v1. One shared summary stores
    corrections, conditional geometry checks, bootstrap intervals and checksums.
    """
    import open3d as o3d

    root = Path(root).expanduser().resolve()
    reference = Path(reference_dir or slabim_registration_dir("v1")).expanduser().resolve()
    output = Path(output_dir or slabim_registration_dir("v2")).expanduser().resolve()
    if any(
        _overlap(output, path)
        for path in (reference, (root / "sensor_data").resolve(), (root / "BIM").resolve())
    ):
        raise ValueError("v1, v2, and raw data must use separate directories")
    if not re.fullmatch(r"\d+F_Region\d+", region):
        raise ValueError(f"Invalid region name: {region}")
    frozen = _summary(reference)
    original = frozen["regions"][region]
    for name in RESULT_FILES:
        if _sha256(reference / region / name) != original["sha256"][name]:
            raise ValueError(f"v1 result checksum mismatch: {region}/{name}")
    source = root / "sensor_data" / region
    if _source_hashes(root, source) != original["source_sha256"]:
        raise ValueError("v1 source timestamps/calibration changed")
    times, lidar = load_pose_table(reference / region / RESULT_FILES[0])
    camera_times, camera = load_pose_table(reference / region / RESULT_FILES[1])
    official = np.loadtxt(source / "map/pose_map_to_bim.txt")
    if official.shape != (4, 4) or not np.allclose(
        official, map_to_bim(source / "points/pose_frame_to_bim.txt", times), atol=1e-4, rtol=0
    ):
        raise ValueError("Official map-to-BIM and v1 trajectory map-to-BIM differ")
    floor = region.split("_")[0]
    mesh_directory = root / "BIM" / floor / "mesh"
    sources = {
        str(path.relative_to(root)): _sha256(path)
        for path in [
            source / "map/data/uncolorized.ply",
            source / "map/pose_map_to_bim.txt",
            *(mesh_directory / f"{name}.ply" for name in CORRECTION_PARAMETERS["fit_classes"]),
        ]
    }
    summary = {
        "version": "v2",
        "reference_version": "v1",
        "method_parameters": CORRECTION_PARAMETERS,
        "regions": {},
    }
    if (output / "summary.json").exists():
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if (
            summary.get("version") != "v2"
            or summary.get("method_parameters") != CORRECTION_PARAMETERS
        ):
            raise ValueError("Existing v2 method differs; use a separate output directory")
    if region in summary["regions"]:
        record = summary["regions"][region]
        if record["source_sha256"] != sources or record["v1_sha256"] != original["sha256"]:
            raise ValueError("Existing v2 source fingerprint changed")
        for name in RESULT_FILES:
            if _sha256(output / region / name) != record["sha256"][name]:
                raise ValueError(f"v2 result checksum mismatch: {region}/{name}")
        return record
    if (output / region).exists():
        raise ValueError(f"Unregistered output directory already exists: {output / region}")
    if surface is None:
        surface = BIMSurface(mesh_directory)
    elif surface.directory != mesh_directory.resolve():
        raise ValueError("BIM surface belongs to a different floor")
    for name, digest in surface.source_sha256.items():
        if sources[str((mesh_directory / name).relative_to(root))] != digest:
            raise ValueError("BIM mesh changed after constructing the cached surface")
    cloud = o3d.io.read_point_cloud(str(source / "map/data/uncolorized.ply"))
    cloud.remove_non_finite_points()
    cloud = cloud.voxel_down_sample(CORRECTION_PARAMETERS["voxel_m"])
    points = _apply(official, np.asarray(cloud.points))
    del cloud
    total = len(points)
    if total < 6:
        raise ValueError("Global LiDAR map has too few points")
    rng = np.random.default_rng(CORRECTION_PARAMETERS["seed"])
    if total > CORRECTION_PARAMETERS["max_points"]:
        points = points[rng.choice(total, CORRECTION_PARAMETERS["max_points"], replace=False)]
    target, normals, _ = surface.closest(points)
    before = _residuals(points, target, normals)
    correction, center = _fit(points, surface)
    current = _apply(correction, points)
    target, normals, labels = surface.closest(current)
    shift = np.linalg.norm(_apply(correction, camera[:, :3, 3]) - camera[:, :3, 3], axis=1) * 100
    record = {
        "lidar_frames": len(times),
        "camera_frames": len(camera_times),
        "excluded_images": original["excluded_images"],
        "correction_to_bim": correction.tolist(),
        "map_to_bim": (correction @ official).tolist(),
        "correction": {
            **_angles(correction),
            "camera_shift_cm_median": float(np.median(shift)),
            "camera_shift_cm_max": float(shift.max()),
        },
        "map_points_downsampled": total,
        "map_points_used": len(points),
        "bim_triangles_used": surface.triangles,
        "degenerate_bim_triangles_removed": surface.degenerate_triangles_removed,
        "before": before,
        "after": _residuals(current, target, normals),
        "floor_slope": _floor_slope(points, current, target, normals, labels),
        "bootstrap": _bootstrap(
            current, target, normals, center, correction, camera[:, :3, 3], rng
        ),
        "source_sha256": sources,
        "v1_sha256": original["sha256"],
    }
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{region}-", dir=output) as temporary:
        staged = Path(temporary) / region
        staged.mkdir()
        for name, stamps, poses in (
            (RESULT_FILES[0], times, lidar),
            (RESULT_FILES[1], camera_times, camera),
        ):
            save_pose_table(staged / name, stamps, correction[None] @ poses)
            load_pose_table(staged / name)
        shutil.copyfile(reference / region / RESULT_FILES[2], staged / RESULT_FILES[2])
        record["sha256"] = {name: _sha256(staged / name) for name in RESULT_FILES}
        staged.rename(output / region)
    summary["regions"][region] = record
    _write_json(output / "summary.json", summary)
    return record
