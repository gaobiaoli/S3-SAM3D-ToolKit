import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import open3d as o3d
from PIL import Image

from s3dis_sam3d import (
    BIMSyncDataset,
    BIMSyncFrameRender,
    BIMSyncRegistration,
    BIMSyncScene,
    FrameRender,
    PointCloud,
)
from s3dis_sam3d import bimsync as bimsync_module
from s3dis_sam3d.pointcloud import transform_points


class _S3DISRoom:
    def __init__(self, cloud, key="Area_1/office_1"):
        self.cloud = cloud
        self.key = key
        self.calls = []

    def point_cloud(self, include_classes=None):
        self.calls.append(include_classes)
        return self.cloud


class _S3DIS:
    def __init__(self, room):
        self.selected_room = room
        area, name = room.key.split("/", 1)
        self.rooms = [SimpleNamespace(area=area, name=name)]

    def room(self, key):
        self.selected_room.key = str(key)
        return self.selected_room


class BIMSyncDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        area = self.root / "Area_1"
        area.mkdir()
        (area / "office_1.ifc").touch()
        (area / "office_2.ifc").touch()
        self.dataset = BIMSyncDataset(self.root)
        self.scene = self.dataset.scene("office_1")

    def tearDown(self):
        self.temp.cleanup()

    def _registration(self, transform=None):
        transform = np.eye(4) if transform is None else transform
        return BIMSyncRegistration(
            "Area_1",
            self.scene.name,
            self.scene.path,
            "Area_1/office_1",
            transform,
            np.linalg.inv(transform),
            0.9,
            0.02,
            0,
            [],
        )

    def test_scene_selection_matching_and_registration_persistence(self):
        self.assertEqual(
            [scene.key for scene in self.dataset.scenes],
            ["Area_1/office_1", "Area_1/office_2"],
        )
        self.assertIsInstance(self.scene, BIMSyncScene)
        for old_name in ("BIMSyncRegion", "IFCRegion", "IFCRegistration", "IFCFrameRender"):
            self.assertFalse(hasattr(bimsync_module, old_name))
        self.assertEqual(self.dataset.scene("Area_1/office_2").name, "office_2")

        s3dis = SimpleNamespace(
            rooms=[
                SimpleNamespace(area="Area_1", name="office_2"),
                SimpleNamespace(area="Area_2", name="office_1"),
            ]
        )
        self.assertEqual(
            [scene.name for scene in self.dataset.matching_scenes(s3dis)],
            ["office_2"],
        )

        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]
        json_path, npy_path = self.scene.save_registration(
            self._registration(transform),
            self.root / "registration",
        )
        self.assertTrue(json_path.is_file())
        np.testing.assert_allclose(np.load(npy_path), transform)
        np.testing.assert_allclose(self.scene.calibration, transform)
        self.assertTrue(self.scene.is_calibrated)

        loaded = BIMSyncDataset(
            self.root,
            calibration_dir=self.root / "registration",
        )
        np.testing.assert_allclose(loaded.scene("office_1").calibration, transform)
        for name in (
            "load_mesh",
            "export_mesh",
            "estimate_ifc_to_s3dis",
            "save_registration",
            "transformed_mesh",
            "visualize_registration",
            "render_regular_frame",
            "resolve_scene",
            "list_scenes",
            "get_calibration",
            "set_calibration",
        ):
            self.assertFalse(hasattr(self.dataset, name))

    def test_default_root_and_calibration_come_from_config(self):
        calibration_dir = self.root / "registration"
        calibration_dir.mkdir()
        np.save(
            calibration_dir / "office_1_ifc_to_s3dis_transform.npy",
            np.eye(4),
        )

        with (
            patch.object(
                bimsync_module.CONFIG,
                "require",
                return_value=self.root,
            ),
            patch.object(
                bimsync_module,
                "bimsync_calibration_dir",
                return_value=calibration_dir,
            ),
        ):
            dataset = BIMSyncDataset()
        self.assertEqual(
            [scene.key for scene in dataset.scenes],
            ["Area_1/office_1", "Area_1/office_2"],
        )
        self.assertTrue(dataset.scene("office_1").is_calibrated)

    def test_missing_default_calibration_directory_fails(self):
        with (
            patch.object(
                bimsync_module.CONFIG,
                "require",
                return_value=self.root,
            ),
            patch.object(
                bimsync_module,
                "bimsync_calibration_dir",
                return_value=self.root / "missing_calibrations",
            ),
            self.assertRaisesRegex(
                FileNotFoundError,
                "BIMSync calibration directory not found",
            ),
        ):
            BIMSyncDataset()

    def test_mesh_uses_saved_or_explicit_transform_once(self):
        saved = np.eye(4)
        saved[:3, 3] = [10, 0, 0]
        requested = np.eye(4)
        requested[:3, 3] = [0, 2, 0]
        self.scene.set_calibration(saved)

        with patch.object(
            BIMSyncScene,
            "_raw_mesh",
            side_effect=lambda *args, **kwargs: o3d.geometry.TriangleMesh.create_box(),
        ):
            calibrated = self.scene.mesh()
            transformed = self.scene.mesh(calibrated=False, transform=requested)

        np.testing.assert_allclose(np.asarray(calibrated.vertices).min(axis=0), [10, 0, 0])
        np.testing.assert_allclose(np.asarray(transformed.vertices).min(axis=0), [0, 2, 0])

        output = self.root / "office_1.ply"
        with (
            patch.object(
                BIMSyncScene,
                "_raw_mesh",
                return_value=o3d.geometry.TriangleMesh.create_box(),
            ),
            patch.object(o3d.io, "write_triangle_mesh", return_value=True) as write,
        ):
            self.assertEqual(self.scene.export(output), output)
        write.assert_called_once()

    def test_raw_mesh_preserves_ifcopenshell_meter_coordinates(self):
        model = SimpleNamespace(
            by_type=lambda _: [SimpleNamespace(Representation=object())]
        )
        settings = SimpleNamespace(USE_WORLD_COORDS="use-world-coords")
        settings.set = lambda *_: None
        shape = SimpleNamespace(
            geometry=SimpleNamespace(
                verts=[0, 0, 0, 5, 0, 0, 0, 3, 2],
                faces=[0, 1, 2],
            )
        )

        with (
            patch("ifcopenshell.open", return_value=model),
            patch("ifcopenshell.geom.settings", return_value=settings),
            patch("ifcopenshell.geom.create_shape", return_value=shape),
        ):
            mesh = self.scene._raw_mesh()

        np.testing.assert_allclose(mesh.get_min_bound(), [0, 0, 0])
        np.testing.assert_allclose(mesh.get_max_bound(), [5, 3, 2])

    def test_registration_direction(self):
        mesh = o3d.geometry.TriangleMesh.create_box(1.0, 2.0, 0.7)
        o3d.utility.random.seed(7)
        ifc_points = np.asarray(mesh.sample_points_uniformly(3000).points)
        expected = np.array(
            [
                [0, -1, 0, 4],
                [1, 0, 0, -2],
                [0, 0, 1, 0.5],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        room = _S3DISRoom(PointCloud(transform_points(ifc_points, expected)))

        with patch.object(BIMSyncScene, "mesh", return_value=mesh):
            result = self.scene.register(
                room,
                ifc_samples=3000,
                voxel_size=0.03,
                thresholds=(0.3, 0.1, 0.03),
                max_iterations=100,
                seed=7,
            )

        np.testing.assert_allclose(
            result.ifc_to_s3dis @ result.s3dis_to_ifc,
            np.eye(4),
            atol=1e-8,
        )
        np.testing.assert_allclose(result.ifc_to_s3dis, expected, atol=0.03)
        self.assertTrue(room.calls)

    def test_registration_with_scaling(self):
        mesh = o3d.geometry.TriangleMesh.create_box(1.0, 2.0, 0.7)
        o3d.utility.random.seed(11)
        ifc_points = np.asarray(mesh.sample_points_uniformly(5000).points)
        expected = np.array(
            [
                [0, -1.2, 0, 4],
                [1.2, 0, 0, -2],
                [0, 0, 1.2, 0.5],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        room = _S3DISRoom(PointCloud(transform_points(ifc_points, expected)))

        with patch.object(BIMSyncScene, "mesh", return_value=mesh):
            result = self.scene.register(
                room,
                ifc_samples=5000,
                voxel_size=0.03,
                thresholds=(0.3, 0.1, 0.03),
                max_iterations=100,
                seed=11,
                with_scaling=True,
            )

        np.testing.assert_allclose(result.ifc_to_s3dis, expected, atol=0.04)
        self.assertAlmostEqual(result.scale, 1.2, places=2)

    def test_batch_calibration_uses_scene_behavior(self):
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]
        registration = self._registration(transform)
        output = self.root / "calibrations"
        room = _S3DISRoom(PointCloud(np.zeros((1, 3))))
        s3dis = _S3DIS(room)

        with (
            patch.object(BIMSyncScene, "register", return_value=registration),
            patch.object(BIMSyncScene, "visualize_registration") as visualize,
        ):
            summary = self.dataset.calibrate_scenes(
                s3dis,
                output,
                ["office_1"],
                visualize=True,
                progress=False,
            )

        self.assertIn("office_1", summary["success"])
        self.assertTrue((output / "calibration_summary.json").is_file())
        self.assertTrue(
            (output / "office_1" / "office_1_ifc_to_s3dis_transform.npy").is_file()
        )
        np.testing.assert_allclose(self.scene.calibration, transform)
        visualize.assert_called_once()

    def test_visualize_registration_uses_scene_mesh(self):
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]
        registration = self._registration(transform)
        cloud = PointCloud(np.zeros((1, 3)))
        room = _S3DISRoom(cloud)
        mesh = o3d.geometry.TriangleMesh.create_box()
        output = self.root / "registration.png"

        with (
            patch.object(BIMSyncScene, "mesh", return_value=mesh) as load_mesh,
            patch(
                "s3dis_sam3d.bimsync.visualize_point_clouds",
                return_value=output,
            ) as view,
        ):
            result = self.scene.visualize_registration(registration, room, output)

        load_mesh.assert_called_once_with(calibrated=False, transform=transform)
        args, kwargs = view.call_args
        self.assertIs(args[0][0], cloud)
        self.assertIsInstance(args[0][1], o3d.geometry.PointCloud)
        np.testing.assert_allclose(np.asarray(args[0][1].colors)[0], [1.0, 0.15, 0.05])
        self.assertEqual(kwargs["save_path"], output)
        self.assertEqual(result, output)

    def test_render_frame_uses_frame_camera_and_calibrated_mesh(self):
        self.scene.set_calibration(np.eye(4))
        image_path = self.root / "frame.png"
        Image.new("RGB", (640, 480)).save(image_path)
        source_depth_path = self.root / "frame_depth.png"
        Image.fromarray(np.full((480, 640), 512, dtype=np.uint16)).save(
            source_depth_path
        )
        intrinsic = np.array([[500, 0, 320], [0, 501, 240], [0, 0, 1]])
        extrinsic = np.eye(4)
        frame = SimpleNamespace(
            room="office_1",
            frame_id=7,
            uuid="camera_uuid",
            rgb_path=image_path,
            depth_path=source_depth_path,
            has_depth=True,
            intrinsics=intrinsic,
            world_to_camera=extrinsic,
            rgb=np.full((480, 640, 3), 0.5, dtype=np.float32),
            depth=np.ones((480, 640), dtype=np.float32),
            projection_type="regular",
        )
        mesh = o3d.geometry.TriangleMesh.create_box()
        output = self.root / "render.png"

        def render_output(*args, **kwargs):
            image = np.full((480, 640, 3), np.array([64, 128, 255]) / 255)
            depth = np.full((480, 640), 2500 / 512, dtype=np.float32)
            depth[0, 0] = 0
            return image, depth

        with (
            patch.object(BIMSyncScene, "mesh", return_value=mesh) as load_mesh,
            patch(
                "s3dis_sam3d.bimsync.render_geometries",
                side_effect=render_output,
            ) as render,
        ):
            result = self.scene.render_frame(frame)

        self.assertIsInstance(result, BIMSyncFrameRender)
        self.assertIsInstance(result, FrameRender)
        self.assertIsNone(result.rendered_image_path)
        self.assertIsNone(result.rendered_depth_path)
        load_mesh.assert_called_once_with()
        _, kwargs = render.call_args
        self.assertEqual((kwargs["width"], kwargs["height"]), (640, 480))
        np.testing.assert_allclose(kwargs["intrinsics"], intrinsic)
        np.testing.assert_allclose(kwargs["world_to_camera"], extrinsic)
        self.assertEqual(result.source_image_path, image_path)
        self.assertEqual(result.source_depth_path, source_depth_path)
        np.testing.assert_allclose(result.source_image, 0.5)
        np.testing.assert_allclose(result.source_depth, 1.0)
        np.testing.assert_allclose(
            result.rendered_image[0, 0],
            np.array([64, 128, 255]) / 255,
        )
        self.assertEqual(result.image_shape, (480, 640))
        self.assertTrue(result.has_source_depth)
        self.assertTrue(result.has_rendered_depth)
        self.assertEqual(result.rendered_depth[0, 0], 0)
        self.assertAlmostEqual(result.rendered_depth[1, 1], 2500 / 512)
        self.assertIs(result.save(output), result)
        self.assertEqual(result.rendered_image_path, output)
        with Image.open(result.rendered_depth_path) as depth_image:
            saved_depth = np.asarray(depth_image, dtype=np.uint16)
        self.assertEqual(saved_depth[0, 0], 65535)

    def test_render_depth_reuses_raycaster_and_invalidates_it_after_calibration(self):
        self.scene.set_calibration(np.eye(4))
        intrinsic = np.array([[5, 0, 3], [0, 5, 2], [0, 0, 1]], dtype=np.float32)
        extrinsic = np.eye(4, dtype=np.float32)
        frame = SimpleNamespace(
            projection_type="regular",
            image_shape=(4, 6),
            intrinsics=intrinsic,
            world_to_camera=extrinsic,
        )
        mesh = o3d.geometry.TriangleMesh.create_box()
        expected = np.full((4, 6), 2.5, dtype=np.float32)

        with (
            patch.object(BIMSyncScene, "mesh", return_value=mesh) as load_mesh,
            patch("s3dis_sam3d.bimsync.MeshRaycaster") as raycaster_type,
        ):
            raycaster_type.return_value.depth.return_value = expected
            first = self.scene.render_depth(
                frame,
                include_types=["IfcWall", "IfcSlab"],
            )
            second = self.scene.render_depth(
                frame,
                include_types=["ifcslab", "ifcwall"],
            )
            updated = np.eye(4)
            updated[0, 3] = 1
            self.scene.set_calibration(updated)
            third = self.scene.render_depth(
                frame,
                include_types=["IfcWall", "IfcSlab"],
            )

        self.assertIs(first, expected)
        self.assertIs(second, expected)
        self.assertIs(third, expected)
        self.assertEqual(load_mesh.call_count, 2)
        for mesh_call in load_mesh.call_args_list:
            self.assertEqual(mesh_call.args, (("IfcWall", "IfcSlab"),))
        self.assertEqual(raycaster_type.call_count, 2)
        self.assertEqual(raycaster_type.return_value.depth.call_count, 3)
        for depth_call in raycaster_type.return_value.depth.call_args_list:
            args = depth_call.args
            np.testing.assert_allclose(args[0], intrinsic)
            np.testing.assert_allclose(args[1], extrinsic)
            self.assertEqual(args[2:], (6, 4))


if __name__ == "__main__":
    unittest.main()
