import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import open3d as o3d
from PIL import Image

from s3dis_sam3d import BIMSyncDataset, IFCRegistration, PointCloud, S23Dataset
from s3dis_sam3d import bimsync as bimsync_module
from s3dis_sam3d.pointcloud import transform_points


class _S3DIS:
    def __init__(self, cloud):
        self.cloud = cloud
        self.calls = []

    def filter_room(self, room, include_classes=None):
        self.calls.append((room, include_classes))
        return self.cloud

    def get_region_point_cloud(self, region, area="Area_1"):
        self.calls.append((f"{area}/{region}", None))
        return self.cloud


class BIMSyncDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        area = self.root / "Area_1"
        area.mkdir()
        (area / "office_1.ifc").touch()
        (area / "office_2.ifc").touch()
        self.dataset = BIMSyncDataset(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_scan_resolve_and_save_registration(self):
        self.assertEqual(
            self.dataset.list_regions(),
            ["Area_1/office_1", "Area_1/office_2"],
        )
        self.assertEqual(self.dataset.resolve_region("Area_1/office_2").name, "office_2")
        s3dis = type(
            "S3DISRooms",
            (),
            {"list_rooms": lambda self: ["Area_1/office_2", "Area_2/office_1"]},
        )()
        self.assertEqual(self.dataset.matching_regions(s3dis), ["office_2"])

        ifc_to_s3dis = np.eye(4)
        ifc_to_s3dis[:3, 3] = [1, 2, 3]
        registration = IFCRegistration(
            "Area_1",
            "office_1",
            self.dataset.resolve_region("office_1").path,
            "Area_1/office_1",
            ifc_to_s3dis,
            np.linalg.inv(ifc_to_s3dis),
            0.9,
            0.02,
            0,
            [],
        )
        json_path, npy_path = self.dataset.save_registration(
            registration, self.root / "registration"
        )
        self.assertTrue(json_path.is_file())
        np.testing.assert_allclose(np.load(npy_path), ifc_to_s3dis)
        np.testing.assert_allclose(
            np.load(self.root / "registration" / "office_1_s3dis_to_ifc_transform.npy"),
            np.linalg.inv(ifc_to_s3dis),
        )
        np.testing.assert_allclose(
            self.dataset.get_calibration("office_1"),
            ifc_to_s3dis,
        )

        loaded = BIMSyncDataset(
            self.root,
            calibration_dir=self.root / "registration",
        )
        self.assertEqual(loaded.calibrated_regions(), ["office_1"])
        np.testing.assert_allclose(loaded.get_calibration("office_1"), ifc_to_s3dis)

    def test_default_root_comes_from_config(self):
        with (
            patch.object(bimsync_module, "BIMSYNC_ROOT", self.root),
            patch.object(
                bimsync_module,
                "bimsync_calibration_dir",
                return_value=self.root / "missing_calibrations",
            ),
        ):
            dataset = BIMSyncDataset()
        self.assertEqual(dataset.list_regions(), ["Area_1/office_1", "Area_1/office_2"])

    def test_load_mesh_applies_calibration_by_default(self):
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]
        self.dataset.set_calibration("office_1", transform)
        raw = o3d.geometry.TriangleMesh.create_box()

        with patch.object(self.dataset, "_load_raw_mesh", return_value=raw):
            calibrated = self.dataset.load_mesh("office_1")

        np.testing.assert_allclose(
            np.asarray(calibrated.vertices).min(axis=0),
            [1, 2, 3],
        )

    def test_transformed_mesh_does_not_apply_saved_calibration_twice(self):
        saved = np.eye(4)
        saved[:3, 3] = [10, 0, 0]
        requested = np.eye(4)
        requested[:3, 3] = [0, 2, 0]
        self.dataset.set_calibration("office_1", saved)
        raw = o3d.geometry.TriangleMesh.create_box()

        with patch.object(self.dataset, "_load_raw_mesh", return_value=raw):
            transformed = self.dataset.transformed_mesh("office_1", requested)

        np.testing.assert_allclose(
            np.asarray(transformed.vertices).min(axis=0),
            [0, 2, 0],
        )

    def test_estimate_ifc_to_s3dis_direction(self):
        mesh = o3d.geometry.TriangleMesh.create_box(1.0, 2.0, 0.7)
        o3d.utility.random.seed(7)
        sampled = mesh.sample_points_uniformly(3000)
        ifc_points = np.asarray(sampled.points)
        expected = np.array(
            [
                [0, -1, 0, 4],
                [1, 0, 0, -2],
                [0, 0, 1, 0.5],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        cloud = PointCloud(transform_points(ifc_points, expected))
        s3dis = _S3DIS(cloud)

        original = self.dataset.load_mesh
        self.dataset.load_mesh = lambda *args, **kwargs: mesh
        try:
            result = self.dataset.estimate_ifc_to_s3dis(
                "office_1",
                s3dis,
                ifc_samples=3000,
                voxel_size=0.03,
                thresholds=(0.3, 0.1, 0.03),
                max_iterations=100,
                seed=7,
            )
        finally:
            self.dataset.load_mesh = original

        np.testing.assert_allclose(
            result.ifc_to_s3dis @ result.s3dis_to_ifc,
            np.eye(4),
            atol=1e-8,
        )
        np.testing.assert_allclose(result.ifc_to_s3dis, expected, atol=0.03)
        self.assertEqual(s3dis.calls[0][0], "Area_1/office_1")

    def test_estimate_ifc_to_s3dis_with_scaling(self):
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
        s3dis = _S3DIS(PointCloud(transform_points(ifc_points, expected)))

        with patch.object(self.dataset, "load_mesh", return_value=mesh):
            result = self.dataset.estimate_ifc_to_s3dis(
                "office_1",
                s3dis,
                ifc_samples=5000,
                voxel_size=0.03,
                thresholds=(0.3, 0.1, 0.03),
                max_iterations=100,
                seed=11,
                with_scaling=True,
            )

        np.testing.assert_allclose(result.ifc_to_s3dis, expected, atol=0.04)
        self.assertAlmostEqual(result.scale, 1.2, places=2)

    def test_calibrate_regions_saves_summary_and_updates_calibrations(self):
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]
        registration = IFCRegistration(
            "Area_1",
            "office_1",
            self.dataset.resolve_region("office_1").path,
            "Area_1/office_1",
            transform,
            np.linalg.inv(transform),
            0.9,
            0.02,
            0,
            [],
        )
        output = self.root / "calibrations"
        cloud = PointCloud(np.zeros((1, 3)))
        s3dis = _S3DIS(cloud)

        with (
            patch.object(
                self.dataset,
                "estimate_ifc_to_s3dis",
                return_value=registration,
            ),
            patch.object(self.dataset, "visualize_registration") as visualize,
        ):
            summary = self.dataset.calibrate_regions(
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
        np.testing.assert_allclose(self.dataset.get_calibration("office_1"), transform)
        visualize.assert_called_once()

    def test_visualize_registration_transforms_ifc_and_saves_screenshot(self):
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]
        registration = IFCRegistration(
            "Area_1",
            "office_1",
            self.dataset.resolve_region("office_1").path,
            "Area_1/office_1",
            transform,
            np.linalg.inv(transform),
            0.9,
            0.02,
            0,
            [],
        )
        cloud = PointCloud(np.zeros((1, 3)))
        mesh = o3d.geometry.TriangleMesh.create_box()
        output = self.root / "registration.png"

        with (
            patch.object(self.dataset, "transformed_mesh", return_value=mesh) as transformed,
            patch("s3dis_sam3d.bimsync.visualize_point_clouds", return_value=output) as view,
        ):
            result = self.dataset.visualize_registration(registration, cloud, output)

        transformed.assert_called_once_with("office_1", transform)
        args, kwargs = view.call_args
        self.assertIs(args[0][0], cloud)
        self.assertIsInstance(args[0][1], o3d.geometry.PointCloud)
        np.testing.assert_allclose(np.asarray(args[0][1].colors)[0], [1.0, 0.15, 0.05])
        self.assertEqual(kwargs["save_path"], output)
        self.assertFalse(kwargs["show"])
        self.assertEqual(result, output)

    def test_render_regular_frame_uses_image_camera_and_calibrated_mesh(self):
        self.dataset.set_calibration("office_1", np.eye(4))
        image_path = self.root / "frame.png"
        Image.new("RGB", (640, 480)).save(image_path)
        source_depth_path = self.root / "frame_depth.png"
        Image.fromarray(np.full((480, 640), 512, dtype=np.uint16)).save(
            source_depth_path
        )
        frame = SimpleNamespace(
            frame_id=7,
            uuid="camera_uuid",
            rgb_path=image_path,
            depth_path=source_depth_path,
        )
        intrinsic = np.array([[500, 0, 320], [0, 501, 240], [0, 0, 1]])
        extrinsic = np.eye(4)

        class S23DIS:
            image_type = "regular"

            @staticmethod
            def get_frame(room, frame_id, uuid):
                return frame

            @staticmethod
            def get_k(room, frame_id, uuid):
                return intrinsic

            @staticmethod
            def world_to_camera(room, frame_id, uuid):
                return extrinsic

            @staticmethod
            def get_image(room, frame_id, uuid):
                return np.full((480, 640, 3), 0.5, dtype=np.float32)

            @staticmethod
            def get_depth(room, frame_id, uuid):
                return np.ones((480, 640), dtype=np.float32)

        mesh = o3d.geometry.TriangleMesh.create_box()
        output = self.root / "render.png"

        def render_output(*args, **kwargs):
            raw_depth = np.full((480, 640), 2500, dtype=np.uint16)
            raw_depth[0, 0] = 0
            Image.fromarray(raw_depth).save(kwargs["depth_path"])
            return output

        with (
            patch.object(self.dataset, "load_mesh", return_value=mesh) as load_mesh,
            patch(
                "s3dis_sam3d.bimsync.visualize_point_clouds",
                side_effect=render_output,
            ) as render,
        ):
            result = self.dataset.render_regular_frame(
                "office_1",
                S23DIS(),
                7,
                output,
                uuid="camera_uuid",
            )

        load_mesh.assert_called_once_with("office_1")
        _, kwargs = render.call_args
        self.assertEqual((kwargs["width"], kwargs["height"]), (640, 480))
        np.testing.assert_allclose(kwargs["set_parameters"][0], intrinsic)
        np.testing.assert_allclose(kwargs["set_parameters"][1], extrinsic)
        self.assertTrue(kwargs["mesh_show_back_face"])
        self.assertEqual(result.rendered_image_path, output)
        self.assertEqual(result.source_image_path, image_path)
        self.assertEqual(result.source_depth_path, source_depth_path)
        np.testing.assert_allclose(result.source_image, 0.5)
        np.testing.assert_allclose(result.source_depth, 1.0)
        self.assertEqual(result.rendered_depth[0, 0], 0)
        self.assertAlmostEqual(result.rendered_depth[1, 1], 2500 / 512)
        with Image.open(result.rendered_depth_path) as depth_image:
            saved_depth = np.asarray(depth_image, dtype=np.uint16)
        self.assertEqual(saved_depth[0, 0], 65535)
        np.testing.assert_allclose(
            S23Dataset.load_depth(result.rendered_depth_path),
            result.rendered_depth,
        )


if __name__ == "__main__":
    unittest.main()
