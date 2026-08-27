import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from s3dis_sam3d import S3DISDataset, S3DISInstance, S3DISRoom
from s3dis_sam3d import s3dis as s3dis_module
from s3dis_sam3d.io import write_ply


class S3DISTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        room = root / "Area_1" / "office_1"
        annotations = room / "Annotations"
        annotations.mkdir(parents=True)
        (room / "office_1.txt").write_text(
            "0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 0 0 0 255\n", encoding="utf-8"
        )
        (annotations / "chair_1.txt").write_text(
            "0 0 0 255 0 0\n1 0 0 0 255 0\n", encoding="utf-8"
        )
        (annotations / "table_1.txt").write_text("0 1 0 0 0 255\n", encoding="utf-8")
        self.dataset = S3DISDataset(root)

    def tearDown(self):
        self.temp.cleanup()

    def test_scan_labels_filter_and_bbox(self):
        self.assertEqual([room.key for room in self.dataset.rooms], ["Area_1/office_1"])
        room = self.dataset.room("office_1")
        self.assertIsInstance(room, S3DISRoom)
        cloud = room.point_cloud()
        self.assertEqual(cloud.xyz.shape, (3, 3))
        chair = room.instance("chair_1")
        self.assertIsInstance(chair, S3DISInstance)
        self.assertEqual(len(chair.point_cloud.xyz), 2)
        filtered = room.point_cloud(include_classes=["table"])
        self.assertEqual(len(filtered.xyz), 1)
        self.assertEqual(chair.bbox.extent.tolist(), [1.0, 0.0, 0.0])

        instances = [item for item in room.instances if item.class_name == "chair"]
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].name, "chair_1")
        for name in (
            "load_room",
            "filter_room",
            "visualize_room",
            "object_cloud",
            "object_clouds",
            "object_bboxes",
            "get_region_point_cloud",
            "get_visualization_cloud",
            "resolve_room",
            "list_rooms",
        ):
            self.assertFalse(hasattr(self.dataset, name))

    def test_default_root_comes_from_config(self):
        with patch.object(s3dis_module, "S3DIS_ROOT", self.dataset.root):
            dataset = S3DISDataset()
        self.assertEqual(dataset.rooms[0].key, "Area_1/office_1")

    def test_ply_export(self):
        cloud = self.dataset.room(0).point_cloud()
        output = Path(self.temp.name) / "cloud.ply"
        write_ply(output, cloud)
        text = output.read_text("ascii")
        self.assertIn("element vertex 3", text)
        self.assertIn("property uchar red", text)

    def test_room_can_be_selected_by_name_or_key(self):
        room = self.dataset.room("office_1")
        self.assertIs(room, self.dataset.room("Area_1/office_1"))
        chairs = room.point_cloud(include_classes=["chair"])
        self.assertEqual(len(chairs.xyz), 2)

    def test_visualization_filter_options_ignore_missing_instances(self):
        room = self.dataset.room("Area_1/office_1")
        cloud = room.point_cloud(
            color_mode="semantic",
            exclude_classes=["floor"],
            exclude_instances=["wall_3", "wall_4"],
            ignore_missing_instances=True,
            max_points=2,
        )
        self.assertEqual(len(cloud.xyz), 2)
        self.assertEqual(cloud.rgb.shape, (2, 3))

        with patch("s3dis_sam3d.s3dis.visualize_point_clouds") as visualize:
            shown = room.visualize(color_mode="instance")
        self.assertEqual(len(shown.xyz), 3)
        visualize.assert_called_once()

    def test_text_files_are_cached_and_results_do_not_share_arrays(self):
        with patch.object(
            s3dis_module,
            "read_xyzrgb_txt",
            wraps=s3dis_module.read_xyzrgb_txt,
        ) as reader:
            room = self.dataset.room("office_1")
            first = room.point_cloud(labels=False)
            self.assertEqual(reader.call_count, 1)
            first.xyz[0, 0] = 999

            second = room.point_cloud(labels=False)
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(second.xyz[0, 0], 0)

            point_path = room.point_path
            point_path.write_text(
                point_path.read_text(encoding="utf-8") + "2 2 2 1 2 3\n",
                encoding="utf-8",
            )
            changed = room.point_cloud(labels=False)
            self.assertEqual(reader.call_count, 2)
            self.assertEqual(len(changed.xyz), 4)

            self.dataset.clear_cache()
            room.point_cloud(labels=False)
            self.assertEqual(reader.call_count, 3)


if __name__ == "__main__":
    unittest.main()
