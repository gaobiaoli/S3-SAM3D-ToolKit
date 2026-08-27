import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from s3dis_sam3d import s23dis as s23dis_module
from s3dis_sam3d.annotations import RoomImageBatchAnnotator


class RoomImageBatchAnnotatorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)

        room = root / "s3dis" / "Area_1" / "office_1"
        annotations = room / "Annotations"
        annotations.mkdir(parents=True)
        chair_1 = [
            (-0.2, -0.2, 2),
            (0, -0.2, 2),
            (0.2, -0.2, 2),
            (-0.2, 0, 2),
            (0.2, 0, 2),
            (-0.2, 0.2, 2),
            (0, 0.2, 2),
            (0.2, 0.2, 2),
        ]
        chair_2 = [(-0.3, -0.3, 3), (0.3, -0.3, 3), (-0.3, 0.3, 3), (0.3, 0.3, 3)]
        self._write_points(annotations / "chair_1.txt", chair_1)
        self._write_points(annotations / "chair_2.txt", chair_2)
        self._write_points(room / "office_1.txt", [*chair_1, *chair_2])

        area = root / "area_1"
        for name in ("pose", "rgb", "depth"):
            (area / "data" / name).mkdir(parents=True)
        stem = "camera_abcdef0123456789_office_1_frame_7_domain"
        pose = {
            "camera_k_matrix": [[10, 0, 5], [0, 10, 5], [0, 0, 1]],
            "camera_rt_matrix": np.eye(3).tolist(),
            "final_camera_rotation": [0, 0, 0],
            "camera_location": [0, 0, 0],
        }
        (area / "data" / "pose" / f"{stem}_pose.json").write_text(
            json.dumps(pose), encoding="utf-8"
        )
        Image.fromarray(np.full((10, 10, 3), 128, dtype=np.uint8)).save(
            area / "data" / "rgb" / f"{stem}_rgb.png"
        )
        Image.fromarray(np.full((10, 10), 1024, dtype=np.uint16)).save(
            area / "data" / "depth" / f"{stem}_depth.png"
        )

        self.output = root / "annotations"
        self.annotator = RoomImageBatchAnnotator(root / "s3dis", area)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _write_points(path, points):
        path.write_text(
            "".join(f"{x} {y} {z} 255 0 0\n" for x, y, z in points),
            encoding="utf-8",
        )

    def test_batch_reuses_frame_data_and_writes_strict_json(self):
        with patch.object(
            s23dis_module,
            "_read_depth",
            wraps=s23dis_module._read_depth,
        ) as load_depth:
            summary = self.annotator.annotate_room_images(
                "Area_1/office_1",
                selected_classes=[" CHAIR "],
                output_dir=self.output,
                min_pixels=4,
                max_instance_points=None,
                progress=False,
            )

        self.assertEqual(load_depth.call_count, 1)
        self.assertEqual(summary["instances_used"], 2)
        self.assertEqual(summary["images_saved"], 1)
        record = json.loads(Path(summary["files"][0]).read_text(encoding="utf-8"))
        self.assertEqual(record["room"], "office_1")
        self.assertEqual(record["s3dis_room"], "Area_1/office_1")
        self.assertEqual(
            [item["instance_name"] for item in record["annotations"]],
            ["chair_1", "chair_2"],
        )
        first = record["annotations"][0]
        self.assertEqual(first["bbox_xyxy"], [4, 4, 6, 6])
        self.assertEqual(first["pixel_count"], 8)
        self.assertEqual(first["occlusion_ratio"], 0)
        self.assertEqual(first["bbox_iou_with_uncropped"], 1)

    def test_save_empty_false_skips_the_frame(self):
        summary = self.annotator.annotate_room_images(
            "office_1",
            selected_classes=["sofa"],
            output_dir=self.output,
            save_empty=False,
            progress=False,
        )
        self.assertEqual(summary["instances_used"], 0)
        self.assertEqual(summary["images_saved"], 0)
        self.assertEqual(list(self.output.glob("*.json")), [])

    def test_missing_occlusion_depth_is_written_as_null(self):
        summary = self.annotator.annotate_room_images(
            "office_1",
            selected_classes=["chair"],
            output_dir=self.output,
            min_pixels=4,
            occlusion_check=False,
            progress=False,
        )
        record = json.loads(Path(summary["files"][0]).read_text(encoding="utf-8"))
        self.assertIsNone(record["annotations"][0]["occlusion_ratio"])
        self.assertEqual(list(self.output.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
