import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from s3dis_sam3d import s3dis as s3dis_module
from s3dis_sam3d.io import write_ply
from s3dis_sam3d.s3dis import S3DISDataset


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
        self.assertEqual(self.dataset.list_rooms(), ["Area_1/office_1"])
        cloud = self.dataset.load_room("office_1")
        self.assertEqual(cloud.xyz.shape, (3, 3))
        chair = self.dataset.object_cloud("Area_1/office_1", "chair_1")
        self.assertEqual(len(chair.xyz), 2)
        filtered = self.dataset.filter_room("office_1", include_classes=["table"])
        self.assertEqual(len(filtered.xyz), 1)
        boxes = self.dataset.object_bboxes("office_1", include_classes=["chair"])
        self.assertEqual(boxes[0]["bbox"]["extent"], [1.0, 0.0, 0.0])

        objects = self.dataset.object_clouds("office_1", include_classes=["chair"])
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].metadata["name"], "chair_1")
        self.assertEqual(len(objects[0].xyz), 2)

    def test_default_root_comes_from_config(self):
        with patch.object(s3dis_module, "S3DIS_ROOT", self.dataset.root):
            dataset = S3DISDataset()
        self.assertEqual(dataset.list_rooms(), ["Area_1/office_1"])

    def test_ply_export(self):
        cloud = self.dataset.load_room(0)
        output = Path(self.temp.name) / "cloud.ply"
        write_ply(output, cloud)
        text = output.read_text("ascii")
        self.assertIn("element vertex 3", text)
        self.assertIn("property uchar red", text)

    def test_get_region_point_cloud_uses_area_1_by_default(self):
        cloud = self.dataset.get_region_point_cloud("office_1")
        self.assertEqual(len(cloud.xyz), 3)

        chairs = self.dataset.get_region_point_cloud(
            "Area_1/office_1",
            include_classes=["chair"],
        )
        self.assertEqual(len(chairs.xyz), 2)

    def test_visualization_filter_options_ignore_missing_instances(self):
        cloud = self.dataset.get_visualization_cloud(
            "Area_1/office_1",
            color_mode="semantic",
            hidden_classes=["floor"],
            hidden_instances=["wall_3", "wall_4"],
            ignore_missing_instances=True,
            max_points=2,
        )
        self.assertEqual(len(cloud.xyz), 2)
        self.assertEqual(cloud.rgb.shape, (2, 3))

    def test_text_files_are_cached_and_results_do_not_share_arrays(self):
        with patch.object(
            s3dis_module,
            "read_xyzrgb_txt",
            wraps=s3dis_module.read_xyzrgb_txt,
        ) as reader:
            first = self.dataset.load_room("office_1", labels=False)
            self.assertEqual(reader.call_count, 1)
            first.xyz[0, 0] = 999

            second = self.dataset.load_room("office_1", labels=False)
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(second.xyz[0, 0], 0)

            point_path = self.dataset.resolve_room("office_1").point_path
            point_path.write_text(
                point_path.read_text(encoding="utf-8") + "2 2 2 1 2 3\n",
                encoding="utf-8",
            )
            changed = self.dataset.load_room("office_1", labels=False)
            self.assertEqual(reader.call_count, 2)
            self.assertEqual(len(changed.xyz), 4)

            self.dataset.clear_cache()
            self.dataset.load_room("office_1", labels=False)
            self.assertEqual(reader.call_count, 3)


if __name__ == "__main__":
    unittest.main()
