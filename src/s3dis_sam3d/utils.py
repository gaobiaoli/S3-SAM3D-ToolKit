"""Dataset-independent geometry and image math."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


def euler_xyz_to_matrix(euler):
    rx, ry, rz = np.asarray(euler)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_matrix = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_matrix = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_matrix = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz_matrix @ ry_matrix @ rx_matrix


def camera_to_world_from_pose(pose):
    rotation = np.asarray(pose["camera_rt_matrix"])[:3, :3]
    adjustment = euler_xyz_to_matrix(pose["final_camera_rotation"])
    transform = np.eye(4)
    transform[:3, :3] = adjustment @ rotation @ adjustment
    transform[:3, 3] = pose["camera_location"]
    return transform


def project_pinhole_points(points, intrinsics):
    points = np.asarray(points, dtype=np.float32)
    depth = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (depth > 1e-6)
    points, depth = points[valid], depth[valid]
    intrinsics = np.asarray(intrinsics)
    u = intrinsics[0, 0] * points[:, 0] / depth + intrinsics[0, 2]
    v = intrinsics[1, 1] * points[:, 1] / depth + intrinsics[1, 2]
    return np.column_stack((u, v)).astype(np.float32), depth.astype(np.float32)


def project_pano_points(points, image_shape):
    points = np.asarray(points, dtype=np.float32)
    height, width = image_shape
    depth = np.linalg.norm(points, axis=1)
    valid = np.isfinite(points).all(axis=1) & (depth > 1e-6)
    points, depth = points[valid], depth[valid]
    theta = np.arctan2(points[:, 0], points[:, 2])
    phi = np.arctan2(-points[:, 1], np.hypot(points[:, 0], points[:, 2]))
    u = np.mod((theta + np.pi) / (2 * np.pi) * width, width)
    v = (0.5 - phi / np.pi) * height
    return np.column_stack((u, v)).astype(np.float32), depth.astype(np.float32)


def prepare_mask(mask, shape=None):
    if mask is None:
        return None
    if isinstance(mask, (str, Path)):
        mask = np.asarray(Image.open(mask))
    elif isinstance(mask, Image.Image):
        mask = np.asarray(mask)
    else:
        mask = np.asarray(mask)

    if mask.ndim == 3:
        rgb = np.any(mask[..., :3] > 0, axis=-1)
        mask = mask[..., 3] > 0 if mask.shape[2] == 4 and not rgb.any() else rgb
    else:
        mask = mask > 0
    if shape is not None and mask.shape != shape:
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize(
                shape[::-1], Image.Resampling.NEAREST
            )
        ) > 0
    return mask


def backproject_regular(
    depth,
    intrinsics,
    stride=1,
    depth_min=0.1,
    depth_max=10.0,
    mask=None,
):
    height, width = depth.shape
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > 0)
    if depth_min is not None:
        valid &= z >= depth_min
    if depth_max is not None:
        valid &= z <= depth_max
    mask = prepare_mask(mask, depth.shape)
    if mask is not None:
        valid &= mask[ys, xs]

    u, v, z = xs[valid], ys[valid], z[valid]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    return points.astype(np.float32), ys[valid], xs[valid]


def backproject_pano(
    depth,
    stride=1,
    depth_min=0.1,
    depth_max=10.0,
    mask=None,
):
    height, width = depth.shape
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    radius = depth[ys, xs]
    valid = np.isfinite(radius) & (radius > 0)
    if depth_min is not None:
        valid &= radius >= depth_min
    if depth_max is not None:
        valid &= radius <= depth_max
    mask = prepare_mask(mask, depth.shape)
    if mask is not None:
        valid &= mask[ys, xs]

    u, v, radius = xs[valid], ys[valid], radius[valid]
    theta = u / width * 2 * np.pi - np.pi
    phi = np.pi / 2 - v / height * np.pi
    points = np.column_stack(
        (
            radius * np.cos(phi) * np.sin(theta),
            -radius * np.sin(phi),
            radius * np.cos(phi) * np.cos(theta),
        )
    )
    return points.astype(np.float32), ys[valid], xs[valid]


def points_from_global_xyz(xyz, rgb, stride=4, mask=None, return_indices=False):
    ys, xs = np.mgrid[0 : xyz.shape[0] : stride, 0 : xyz.shape[1] : stride]
    points = xyz[ys, xs]
    colors = rgb[ys, xs]
    valid = np.isfinite(points).all(axis=-1) & ~np.all(np.abs(points) < 1e-8, axis=-1)
    mask = prepare_mask(mask, xyz.shape[:2])
    if mask is not None:
        valid &= mask[ys, xs]
    result = points[valid], colors[valid]
    if return_indices:
        return *result, ys[valid], xs[valid]
    return result


def transformed_geometry(geometry, transform):
    result = copy.deepcopy(geometry)
    if hasattr(result, "apply_transform"):
        result.apply_transform(transform)
    else:
        result.transform(transform)
    return result


def preprocess_registration_cloud(cloud, voxel_size):
    cloud = cloud.voxel_down_sample(voxel_size) if voxel_size else copy.deepcopy(cloud)
    radius = max(voxel_size * 3, 0.05) if voxel_size else 0.1
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60)
    )
    return cloud


def initial_registration_transform(source, target, yaw_deg, with_scaling=False):
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    angle = math.radians(yaw_deg)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    scale = 1.0
    if with_scaling:
        source_extent = np.ptp(source_points @ rotation.T, axis=0)
        target_extent = np.ptp(target_points, axis=0)
        valid = (source_extent > 1e-6) & (target_extent > 1e-6)
        scale = float(np.median(target_extent[valid] / source_extent[valid]))

    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target_center - transform[:3, :3] @ source_center
    if not with_scaling:
        transform[2, 3] = target_points[:, 2].min() - source_points[:, 2].min()
    return transform


def run_icp(
    source,
    target,
    initial,
    thresholds,
    max_iterations,
    with_scaling=False,
):
    transform = initial
    stages = []
    estimator = (
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True)
        if with_scaling
        else o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    for threshold in thresholds:
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            float(threshold),
            transform,
            estimator,
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations
            ),
        )
        transform = result.transformation
        stages.append(
            {
                "threshold": float(threshold),
                "fitness": float(result.fitness),
                "rmse": float(result.inlier_rmse),
            }
        )
    return transform, stages
