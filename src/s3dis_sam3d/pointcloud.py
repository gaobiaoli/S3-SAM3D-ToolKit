from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import open3d as o3d

from .models import BoundingBox3D, GLBMesh, PointCloud


def random_downsample(cloud, max_points, seed=42):
    if len(cloud.xyz) <= max_points:
        return cloud
    indices = np.random.default_rng(seed).choice(len(cloud.xyz), max_points, replace=False)
    return cloud.select(indices)


def voxel_downsample(cloud, voxel_size):
    if not voxel_size or not len(cloud.xyz):
        return cloud

    keys = np.floor(cloud.xyz / voxel_size).astype(np.int64)
    _, first, groups = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )

    # Keep voxels in input order instead of np.unique's lexicographic order.
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    groups = remap[groups]
    counts = np.bincount(groups)

    def mean(values):
        return np.column_stack(
            [np.bincount(groups, weights=column) / counts for column in values.T]
        )

    xyz = mean(cloud.xyz)
    rgb = None if cloud.rgb is None else mean(cloud.rgb)

    semantic = cloud.semantic_labels
    instance = cloud.instance_labels
    labels = [values for values in (semantic, instance) if values is not None]
    if labels:
        # Vote on the complete label tuple so semantic and instance labels remain
        # paired. Ties are resolved deterministically by the smaller tuple.
        candidates, votes = np.unique(
            np.column_stack((groups, *labels)), axis=0, return_counts=True
        )
        ranked = np.lexsort((np.arange(len(votes)), -votes, candidates[:, 0]))
        ranked = ranked[
            np.r_[True, np.diff(candidates[ranked, 0]) != 0]
        ]
        voted = candidates[ranked, 1:]
        if semantic is not None and instance is not None:
            semantic, instance = voted[:, 0], voted[:, 1]
        elif semantic is not None:
            semantic = voted[:, 0]
        else:
            instance = voted[:, 0]

    return PointCloud(xyz, rgb, semantic, instance, dict(cloud.metadata))


def transform_points(points, transform):
    transform = np.asarray(transform)
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def transform_point_cloud(cloud, transform):
    return PointCloud(
        transform_points(cloud.xyz, transform),
        None if cloud.rgb is None else cloud.rgb.copy(),
        None if cloud.semantic_labels is None else cloud.semantic_labels.copy(),
        None if cloud.instance_labels is None else cloud.instance_labels.copy(),
        dict(cloud.metadata),
    )


def to_open3d_point_cloud(cloud):
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(cloud.xyz)
    if cloud.rgb is not None:
        geometry.colors = o3d.utility.Vector3dVector(np.clip(cloud.rgb, 0, 1))
    return geometry


def bounding_box_to_open3d(bbox, color=(1.0, 0.1, 0.1)):
    box = o3d.geometry.AxisAlignedBoundingBox(bbox.min_bound, bbox.max_bound)
    lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
    lines.paint_uniform_color(color)
    return lines


def to_open3d_geometry(geometry):
    """Convert toolkit objects while leaving native Open3D geometries unchanged."""
    if isinstance(geometry, PointCloud):
        return to_open3d_point_cloud(geometry)
    if isinstance(geometry, GLBMesh):
        return geometry.get()
    if isinstance(geometry, BoundingBox3D):
        return bounding_box_to_open3d(geometry)
    return geometry

def set_camera(vis,
               width=1080,
               height=1080,
                fx=935.307,
                fy=935.307,
                cx=539.5,
                cy=539.5,
                extrinsic=None,
               ):
    if extrinsic is None:
        extrinsic = np.eye(4)
    ctr =  vis.get_view_control()
    param = o3d.camera.PinholeCameraParameters()

    # 创建内参对象
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    param.intrinsic = intrinsic
    param.extrinsic = extrinsic

    # 核心：必须 allow_fovy_change=True，否则 Open3D 会用它自己的默认缩放
    ctr.convert_from_pinhole_camera_parameters(param,allow_arbitrary=True)
    return vis

def visualize_point_clouds(
    geometries: Sequence,
    bounding_boxes: Sequence[BoundingBox3D] = (),
    window_name="Open3D",
    width=1920,
    height=1080,
    point_size=2.0,
    background_color=(1, 1, 1),
    show_coordinate_frame=False,
    coordinate_frame_size=0.5,
    get_parameters=False,
    set_parameters=None,
    camera_view=False,
    save_path=None,
    depth_path=None,
    depth_scale=1000.0,
    show=True,
    mesh_show_back_face=False,
):
    """Visualize mixed point clouds, meshes, boxes, and native Open3D geometries."""
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name=window_name,
        width=width,
        height=height,
        visible=show,
    )

    for geometry in geometries:
        visualizer.add_geometry(to_open3d_geometry(geometry))
    for bbox in bounding_boxes:
        visualizer.add_geometry(bounding_box_to_open3d(bbox))

    if show_coordinate_frame:
        visualizer.add_geometry(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=coordinate_frame_size)
        )

    if set_parameters is not None:
        intrinsic, extrinsic = set_parameters
        intrinsic = np.asarray(intrinsic)
        parameters = o3d.camera.PinholeCameraParameters()
        parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
        )
        parameters.extrinsic = np.asarray(extrinsic)
        visualizer.get_view_control().convert_from_pinhole_camera_parameters(
            parameters,
            allow_arbitrary=True,
        )

    elif camera_view:
        set_camera(visualizer)

    render = visualizer.get_render_option()
    render.point_size = point_size
    render.background_color = np.asarray(background_color)
    render.mesh_show_back_face = mesh_show_back_face
    visualizer.poll_events()
    visualizer.update_renderer()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        visualizer.capture_screen_image(str(save_path), do_render=True)

    if depth_path is not None:
        depth_path = Path(depth_path)
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        visualizer.capture_depth_image(
            str(depth_path),
            do_render=True,
            depth_scale=depth_scale,
        )

    if show:
        visualizer.run()

    if get_parameters:
        parameters = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
        print("intrinsic =", parameters.intrinsic.intrinsic_matrix.tolist())
        print("extrinsic =", parameters.extrinsic.tolist())
    visualizer.destroy_window()
    return save_path
