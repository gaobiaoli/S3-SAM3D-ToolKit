import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from s3dis_sam3d.annotations import RoomImageAnnotationsParser


class RoomImageAnnotationsParserTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.annotations_dir = root / "annotations"
        self.image_root = root / "images_root"
        (self.image_root / "images").mkdir(parents=True)
        self.annotations_dir.mkdir()

        for name, color in (("frame_a.png", (10, 20, 30)), ("frame_b.png", (1, 2, 3))):
            Image.new("RGB", (8, 8), color).save(self.image_root / "images" / name)

        annotation = {
            "instance_name": "chair_1",
            "class_name": "chair",
            "bbox_xyxy": [2, 3, 4, 5],
            "bbox_uncropped_xyxy": [2, 3, 4, 5],
            "bbox_iou_with_uncropped": 0.5,
            "pixel_count": 20,
            "occlusion_ratio": 0.5,
        }
        self._write_record(
            "a_ann.json",
            "images\\frame_a.png",
            "uuid_a",
            [annotation],
        )
        self._write_record(
            "b_ann.json",
            "images/frame_b.png",
            "uuid_b",
            [{**annotation, "instance_name": "table_1", "class_name": "table"}],
        )
        (self.annotations_dir / "summary.json").write_text("{}", "utf-8")
        self.parser = RoomImageAnnotationsParser(
            self.annotations_dir,
            image_root=self.image_root,
        )

    def tearDown(self):
        self.temp.cleanup()

    def _write_record(self, name, image_path, uuid, annotations):
        record = {
            "image_path": image_path,
            "room": "office_1",
            "s3dis_room": "Area_1/office_1",
            "frame_id": 7,
            "uuid": uuid,
            "annotations": annotations,
        }
        (self.annotations_dir / name).write_text(json.dumps(record), "utf-8")

    def test_indexes_paths_and_ambiguity(self):
        self.assertEqual(len(self.parser), 2)
        self.assertEqual(self.parser.list_records()[0], (0, "office_1", 7, "uuid_a", 1))
        self.assertEqual(self.parser.get_record_by_uuid_frame("uuid_b", 7)["uuid"], "uuid_b")
        with self.assertRaisesRegex(ValueError, "multiple records"):
            self.parser.get_record_by_room_frame("office_1", 7)

        image_path = (self.image_root / "images" / "frame_a.png").resolve()
        self.assertEqual(
            self.parser.get_index_by_image_path(image_path, allow_basename=False),
            0,
        )
        self.assertEqual(self.parser.resolve_image_path(0), image_path)
        np.testing.assert_array_equal(self.parser.read_image(0)[0, 0], [10, 20, 30])
        np.testing.assert_array_equal(self.parser.read_image(0, as_rgb=False)[0, 0], [30, 20, 10])

    def test_filter_iteration_and_visualization(self):
        record = self.parser[0]
        selected = self.parser.filter_annotations(
            record,
            class_filter=" CHAIR ",
            min_bbox_iou=0.5,
            max_occlusion_ratio=0.5,
            min_pixels=20,
        )
        self.assertEqual([item["instance_name"] for item in selected], ["chair_1"])
        self.assertEqual(len(list(self.parser.iter_annotations(instance_filter=["chair_1"]))), 1)

        output = Path(self.temp.name) / "visualized.png"
        result = self.parser.visualize_bboxes_on_image(0, save_path=output)
        self.assertEqual(result["drawn_boxes"], 1)
        self.assertTrue(output.is_file())
        self.assertEqual(result["image_rgb"].shape, (8, 8, 3))


if __name__ == "__main__":
    unittest.main()
