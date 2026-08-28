import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from s3dis_sam3d import S23Frame, RGBDFrame, S23Dataset, S23Room, parse_stem
from s3dis_sam3d import s23dis as s23dis_module


class S23DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        area = Path(self.temp.name) / "area_1"
        self.area = area
        for name in ("pose", "rgb", "depth"):
            (area / "data" / name).mkdir(parents=True)
        stem = "camera_0123456789abcdef_office_1_frame_7_domain"
        pose = {
            "camera_k_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
            "camera_rt_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "final_camera_rotation": [0, 0, 0],
            "camera_location": [1, 2, 3],
        }
        (area / "data" / "pose" / f"{stem}_pose.json").write_text(
            json.dumps(pose), "utf-8"
        )
        Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)).save(
            area / "data" / "rgb" / f"{stem}_rgb.png"
        )
        Image.fromarray(np.full((2, 2), 512, dtype=np.uint16)).save(
            area / "data" / "depth" / f"{stem}_depth.png"
        )
        self.dataset = S23Dataset(area)
        self.room = self.dataset.room("office_1")

    def tearDown(self):
        self.temp.cleanup()

    def test_point_map_and_world_reconstruction(self):
        self.assertEqual(self.dataset.list_rooms(), [("office_1", 1)])
        frame = self.room.get_frame(7)
        self.assertEqual(frame.uuid, "0123456789abcdef")
        point_map = frame.point_map()
        np.testing.assert_allclose(point_map[1, 1], [0.5, 0.5, 1])
        cloud = self.room.reconstruct(frame_id=7, stride=1, voxel_size=None)
        np.testing.assert_allclose(cloud.xyz[-1], [1.5, 2.5, 4])
        self.assertEqual(cloud.metadata["area"], "Area_1")

    def test_frame_owns_its_data_and_camera_properties(self):
        frame = self.room.get_frame(7)

        self.assertIsInstance(frame, S23Frame)
        self.assertEqual(frame.projection_type, "regular")
        self.assertTrue(frame.has_depth)
        self.assertFalse(frame.has_xyz)
        self.assertIs(frame.pose, frame.pose)
        self.assertEqual(frame.image_shape, (2, 2))
        np.testing.assert_allclose(frame.rgb, 128 / 255)
        np.testing.assert_allclose(frame.depth, 1)
        np.testing.assert_allclose(
            frame.intrinsics,
            [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        )
        np.testing.assert_allclose(frame.camera_to_world[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(
            frame.world_to_camera @ frame.camera_to_world,
            np.eye(4),
            atol=1e-7,
        )
        with self.assertRaises(FileNotFoundError):
            _ = frame.xyz

    def test_default_area_path_comes_from_config(self):
        with patch.object(
            s23dis_module,
            "s23dis_area",
            return_value=self.dataset.area_path,
        ):
            dataset = S23Dataset()
        self.assertEqual(dataset.list_rooms(), [("office_1", 1)])

    def test_default_uuid_and_uuid_listing(self):
        first = self.room.get_frame(7)
        second_stem = "camera_fedcba9876543210_office_1_frame_7_domain"
        (self.area / "data" / "pose" / f"{second_stem}_pose.json").write_bytes(
            first.pose_path.read_bytes()
        )
        (self.area / "data" / "rgb" / f"{second_stem}_rgb.png").write_bytes(
            first.rgb_path.read_bytes()
        )
        (self.area / "data" / "depth" / f"{second_stem}_depth.png").write_bytes(
            first.depth_path.read_bytes()
        )

        dataset = S23Dataset(self.area)
        expected = ["0123456789abcdef", "fedcba9876543210"]
        self.assertEqual(dataset.list_uuids(), expected)
        self.assertEqual(dataset.room("office_1").list_uuids(), expected)
        self.assertEqual(dataset.room("office_1").list_uuids(7), expected)
        self.assertEqual(dataset.room("office_1").get_frame(7).uuid, expected[0])

        preferred = S23Dataset(self.area, default_uuid=expected[1])
        self.assertEqual(preferred.room("office_1").get_frame(7).uuid, expected[1])

    def test_world_point_projection(self):
        frame = self.room.get_frame(7)
        pixels, depth = frame.project_world_points([[1, 2, 4], [1, 2, 2]])
        np.testing.assert_allclose(pixels, [[0, 0]])
        np.testing.assert_allclose(depth, [1])

    def test_frame_is_the_single_entry_for_frame_data(self):
        frame = self.room.get_frame(7)
        self.assertIsInstance(frame, RGBDFrame)
        np.testing.assert_allclose(frame.depth, 1)
        np.testing.assert_allclose(
            frame.intrinsics,
            [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        )
        np.testing.assert_allclose(frame.rgb, 128 / 255)
        self.assertEqual(self.room.frames, (frame,))
        for name in ("load_pose", "load_rgb", "load_depth", "load_xyz"):
            self.assertFalse(hasattr(frame, name))
        self.assertFalse(hasattr(frame, "K"))
        self.assertFalse(hasattr(frame, "frame_cloud"))
        self.assertFalse(hasattr(frame, "_backproject_frame"))
        for name in (
            "load_pose",
            "get_depth",
            "get_k",
            "get_xyz",
            "get_image",
            "camera_to_world",
            "world_to_camera",
            "get_room_frames",
            "project_world_points",
            "point_cloud",
            "point_map",
            "visualize",
            "save_ply",
            "reconstruct",
            "visualize_room",
            "save_room_ply",
        ):
            self.assertFalse(hasattr(self.dataset, name))

    def test_room_matches_matterport_scene_object_protocol(self):
        frame = self.room.get_frame(7)

        self.assertIsInstance(self.room, S23Room)
        self.assertEqual(self.room.key, "Area_1/office_1")
        self.assertEqual(self.room.projection_type, "regular")
        self.assertIs(self.dataset[0], self.room)
        self.assertIs(self.dataset["Area_1/office_1"], self.room)
        self.assertEqual(tuple(self.dataset), (self.room,))
        self.assertEqual(tuple(self.room), (frame,))
        self.assertIs(self.room[0], frame)
        self.assertIs(self.room[frame.stem], frame)

    def test_mask_path_rgb_mapping_and_room_helpers(self):
        frame = self.room.get_frame(7)
        mask_path = Path(self.temp.name) / "mask.png"
        mask = np.zeros((2, 2, 3), dtype=np.uint8)
        mask[1, 1] = 255
        Image.fromarray(mask).save(mask_path)

        cloud = frame.point_cloud(
            stride=1,
            mask={frame.stem: mask_path},
            world_coordinates=False,
        )
        self.assertEqual(len(cloud.xyz), 1)
        np.testing.assert_allclose(cloud.xyz[0], [0.5, 0.5, 1])

        area = Path(self.temp.name) / "area_1" / "data"
        second_stem = "camera_fedcba9876543210_office_1_frame_8_domain"
        (area / "pose" / f"{second_stem}_pose.json").write_bytes(
            frame.pose_path.read_bytes()
        )
        (area / "rgb" / f"{second_stem}_rgb.png").write_bytes(
            frame.rgb_path.read_bytes()
        )
        (area / "depth" / f"{second_stem}_depth.png").write_bytes(
            frame.depth_path.read_bytes()
        )
        masked_room = S23Dataset(Path(self.temp.name) / "area_1").room(
            "office_1"
        ).reconstruct(
            stride=1,
            voxel_size=None,
            mask={frame.stem: mask_path},
        )
        self.assertEqual(len(masked_room.xyz), 1)
        self.assertEqual(masked_room.metadata["frame_count"], 1)

        output = Path(self.temp.name) / "room.ply"
        self.assertEqual(
            self.room.save_ply(output, stride=1, voxel_size=None),
            output,
        )
        self.assertTrue(output.is_file())

        with patch("s3dis_sam3d.s23dis.visualize_point_clouds") as visualize:
            result = self.room.visualize(stride=1, voxel_size=None)
        self.assertEqual(len(result.xyz), 4)
        visualize.assert_called_once()

    def test_parse_asset_stem(self):
        info = parse_stem(
            "camera_0123456789abcdef_office_1_frame_7_domain_rgb_"
            "conference_table_12_optimized.json"
        )
        self.assertEqual(info["room"], "office_1")
        self.assertEqual(info["frame_id"], 7)
        self.assertEqual(info["class_name"], "conference_table")
        self.assertEqual(info["instance_id"], 12)
        self.assertEqual(info["instance_name"], "conference_table_12")

    def test_pano_point_map_uses_spherical_backprojection(self):
        area = Path(self.temp.name) / "pano_area"
        for name in ("pose", "rgb", "depth"):
            (area / "pano" / name).mkdir(parents=True)
        stem = "camera_abcdef0123456789_office_1_frame_equirectangular_domain"
        pose = {
            "camera_k_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "camera_rt_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "final_camera_rotation": [0, 0, 0],
            "camera_location": [0, 0, 0],
        }
        (area / "pano" / "pose" / f"{stem}_pose.json").write_text(
            json.dumps(pose), "utf-8"
        )
        Image.fromarray(np.full((2, 4, 3), 128, dtype=np.uint8)).save(
            area / "pano" / "rgb" / f"{stem}_rgb.png"
        )
        Image.fromarray(np.full((2, 4), 512, dtype=np.uint16)).save(
            area / "pano" / "depth" / f"{stem}_depth.png"
        )

        dataset = S23Dataset(area, projection_type="pano")
        frame = dataset.room("office_1").get_frame(0, uuid="abcdef0123456789")
        self.assertEqual(frame.projection_type, "pano")
        point_map = frame.point_map()
        np.testing.assert_allclose(point_map[1, 1], [-1, 0, 0], atol=1e-7)
        pixels, depth = frame.project_camera_points([[-1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(pixels, [[1, 1], [2, 1]], atol=1e-7)
        np.testing.assert_allclose(depth, [1, 1])

    def test_global_xyz_only_frame_is_indexed(self):
        area = Path(self.temp.name) / "xyz_area"
        for name in ("pose", "rgb", "global_xyz"):
            (area / "pano" / name).mkdir(parents=True)
        stem = "camera_abcdef0123456789_office_2_frame_equirectangular_domain"
        pose = {
            "camera_k_matrix": np.eye(3).tolist(),
            "camera_rt_matrix": np.eye(3).tolist(),
            "final_camera_rotation": [0, 0, 0],
            "camera_location": [1, 2, 3],
        }
        (area / "pano" / "pose" / f"{stem}_pose.json").write_text(
            json.dumps(pose), "utf-8"
        )
        Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)).save(
            area / "pano" / "rgb" / f"{stem}_rgb.png"
        )
        (area / "pano" / "global_xyz" / f"{stem}_global_xyz.exr").touch()

        dataset = S23Dataset(area, projection_type="pano")
        frame = dataset.room("office_2").get_frame(0, uuid="abcdef0123456789")
        xyz = np.array(
            [[[0, 0, 0], [1, 2, 4]], [[1, 3, 3], [2, 3, 4]]],
            dtype=np.float32,
        )
        with patch.object(s23dis_module, "_read_global_xyz", return_value=xyz):
            world = frame.point_cloud(stride=1, from_global_xyz=True)
            camera = frame.point_cloud(
                stride=1,
                from_global_xyz=True,
                world_coordinates=False,
            )
        self.assertEqual(len(world.xyz), 3)
        self.assertTrue(frame.has_xyz)
        self.assertFalse(frame.has_depth)
        np.testing.assert_allclose(camera.xyz[0], [0, 0, 1])
        self.assertEqual(camera.metadata["coordinate_frame"], "camera")


if __name__ == "__main__":
    unittest.main()
