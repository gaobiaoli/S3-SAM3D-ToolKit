import numpy as np
import open3d as o3d

from s3dis_sam3d import BoundingBox3D, PointCloud
from s3dis_sam3d import pointcloud as pointcloud_module
from s3dis_sam3d.pointcloud import (
    random_downsample,
    to_open3d_geometry,
    transform_point_cloud,
    visualize_point_clouds,
    voxel_downsample,
)


def _cloud() -> PointCloud:
    return PointCloud(
        xyz=np.array([[0, 0, 0], [0.01, 0.01, 0.01], [0.2, 0, 0]], dtype=np.float32),
        rgb=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        semantic_labels=np.array([10, 11, 12]),
        instance_labels=np.array([20, 21, 22]),
        metadata={"name": "sample"},
    )


def test_random_downsample_is_deterministic_and_preserves_attributes():
    first = random_downsample(_cloud(), 2, seed=7)
    second = random_downsample(_cloud(), 2, seed=7)
    np.testing.assert_array_equal(first.xyz, second.xyz)
    np.testing.assert_array_equal(first.semantic_labels, second.semantic_labels)
    assert len(first.xyz) == 2


def test_voxel_downsample_uses_voxel_centroids_and_mean_colors():
    cloud = _cloud()
    result = voxel_downsample(cloud, 0.1)
    np.testing.assert_allclose(result.xyz, [[0.005, 0.005, 0.005], [0.2, 0, 0]])
    np.testing.assert_allclose(result.rgb, [[0.5, 0.5, 0], [0, 0, 1]])
    np.testing.assert_array_equal(result.semantic_labels, [10, 12])
    np.testing.assert_array_equal(result.instance_labels, [20, 22])
    assert result.metadata == {"name": "sample"}
    assert result.metadata is not cloud.metadata


def test_voxel_downsample_votes_on_semantic_instance_pairs():
    cloud = PointCloud(
        xyz=np.arange(27, dtype=np.float32).reshape(-1, 3) / 100,
        semantic_labels=np.array([1, 1, 1, 1, 2, 2, 2, 3, 3]),
        instance_labels=np.array([10, 10, 10, 20, 20, 20, 20, 20, 20]),
    )

    result = voxel_downsample(cloud, 1.0)

    # Independent votes would produce the nonexistent pair (1, 20).
    np.testing.assert_array_equal(result.semantic_labels, [1])
    np.testing.assert_array_equal(result.instance_labels, [10])


def test_transform_point_cloud_does_not_mutate_input():
    cloud = _cloud()
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    result = transform_point_cloud(cloud, transform)
    np.testing.assert_allclose(result.xyz, cloud.xyz + [1, 2, 3])
    np.testing.assert_allclose(cloud.xyz[0], [0, 0, 0])
    assert result.metadata == cloud.metadata


def test_to_open3d_geometry_supports_toolkit_and_native_objects():
    cloud = to_open3d_geometry(_cloud())
    mesh = o3d.geometry.TriangleMesh.create_box()
    bbox = BoundingBox3D.from_points([[0, 0, 0], [1, 1, 1]])

    assert isinstance(cloud, o3d.geometry.PointCloud)
    assert to_open3d_geometry(mesh) is mesh
    assert isinstance(to_open3d_geometry(bbox), o3d.geometry.LineSet)


def test_visualize_point_clouds_accepts_mixed_geometry_and_saves_image(
    monkeypatch, tmp_path
):
    added = []
    state = {"ran": False, "captured": None, "depth": None, "camera": None}

    class RenderOptions:
        point_size = None
        background_color = None

    class ViewControl:
        def convert_from_pinhole_camera_parameters(self, parameters, allow_arbitrary):
            state["camera"] = (parameters, allow_arbitrary)

    class Visualizer:
        def create_window(self, **kwargs):
            self.window_options = kwargs

        def add_geometry(self, geometry):
            added.append(geometry)

        def get_render_option(self):
            return RenderOptions()

        def get_view_control(self):
            return ViewControl()

        def poll_events(self):
            pass

        def update_renderer(self):
            pass

        def capture_screen_image(self, path, do_render):
            state["captured"] = (path, do_render)

        def capture_depth_image(self, path, do_render, depth_scale):
            state["depth"] = (path, do_render, depth_scale)

        def run(self):
            state["ran"] = True

        def destroy_window(self):
            pass

    monkeypatch.setattr(pointcloud_module.o3d.visualization, "Visualizer", Visualizer)
    mesh = o3d.geometry.TriangleMesh.create_box()
    bbox = BoundingBox3D.from_points([[0, 0, 0], [1, 1, 1]])

    output = tmp_path / "preview" / "registration.png"
    depth_output = tmp_path / "preview" / "registration_depth.png"
    intrinsic = np.array([[900, 0, 960], [0, 901, 540], [0, 0, 1]])
    extrinsic = np.eye(4)
    result = visualize_point_clouds(
        [_cloud(), mesh],
        bounding_boxes=[bbox],
        show_coordinate_frame=False,
        save_path=output,
        depth_path=depth_output,
        depth_scale=2000,
        show=False,
        set_parameters=(intrinsic, extrinsic),
    )

    assert isinstance(added[0], o3d.geometry.PointCloud)
    assert added[1] is mesh
    assert isinstance(added[2], o3d.geometry.LineSet)
    assert result == output
    assert state["captured"] == (str(output), True)
    assert state["depth"] == (str(depth_output), True, 2000)
    assert not state["ran"]
    parameters, allow_arbitrary = state["camera"]
    assert parameters.intrinsic.width == 1920
    assert parameters.intrinsic.height == 1080
    np.testing.assert_allclose(parameters.intrinsic.intrinsic_matrix, intrinsic)
    np.testing.assert_allclose(parameters.extrinsic, extrinsic)
    assert allow_arbitrary
