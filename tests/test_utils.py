import numpy as np
import open3d as o3d

from s3dis_sam3d.utils import (
    backproject_regular,
    camera_to_world_from_pose,
    euler_xyz_to_matrix,
    initial_registration_transform,
    project_pinhole_points,
    transformed_geometry,
)


def test_camera_math_is_dataset_independent():
    rotation = euler_xyz_to_matrix([0, 0, np.pi / 2])
    np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-7)

    pose = {
        "camera_rt_matrix": np.eye(3),
        "final_camera_rotation": [0, 0, 0],
        "camera_location": [1, 2, 3],
    }
    transform = camera_to_world_from_pose(pose)
    np.testing.assert_allclose(transform[:3, 3], [1, 2, 3])

    mesh = o3d.geometry.TriangleMesh.create_box()
    transformed = transformed_geometry(mesh, transform)
    np.testing.assert_allclose(mesh.get_center(), [0.5, 0.5, 0.5])
    np.testing.assert_allclose(transformed.get_center(), [1.5, 2.5, 3.5])


def test_pinhole_projection_and_backprojection_round_trip():
    intrinsics = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    depth = np.ones((2, 2), dtype=np.float32)
    points, ys, xs = backproject_regular(depth, intrinsics)
    pixels, projected_depth = project_pinhole_points(points, intrinsics)

    np.testing.assert_allclose(pixels, np.column_stack((xs, ys)))
    np.testing.assert_allclose(projected_depth, 1)


def test_registration_helpers_are_dataset_independent():
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[0, 0, 0], [1, 2, 1]]))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[3, 4, 1], [4, 6, 2]]))
    transform = initial_registration_transform(source, target, yaw_deg=0)
    np.testing.assert_allclose(transform[:3, 3], [3, 4, 1])
