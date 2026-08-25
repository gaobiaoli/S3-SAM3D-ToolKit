import base64
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from s3dis_sam3d import S23Dataset
from s3dis_sam3d.annotations import RoomImageAnnotationsParser
from s3dis_sam3d.sam3d import SAM3DBatchPredictor, SAM3DClient


class _Response:
    @staticmethod
    def raise_for_status():
        pass

    def __init__(self, request_id):
        self.request_id = request_id

    def json(self):
        pose = {
            "object_0": {
                "rotation": [1, 0, 0, 0],
                "translation": [0, 0, 0],
                "scale": [1, 1, 1],
            }
        }
        return {
            "request_id": self.request_id,
            "glb_b64": base64.b64encode(b"mock-glb").decode("ascii"),
            "mask_png_b64": base64.b64encode(b"mock-mask").decode("ascii"),
            "pose": pose,
            "pose_optimized": pose,
        }


class _Session:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(kwargs["data"]["request_id"])


class SAM3DBatchPredictorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.area = root / "area_1"
        for name in ("pose", "rgb", "depth"):
            (self.area / "data" / name).mkdir(parents=True)
        self.stem = "camera_abcdef0123456789_office_1_frame_7_domain"
        pose = {
            "camera_k_matrix": [[10, 0, 5], [0, 10, 5], [0, 0, 1]],
            "camera_rt_matrix": np.eye(3).tolist(),
            "final_camera_rotation": [0, 0, 0],
            "camera_location": [0, 0, 0],
        }
        (self.area / "data" / "pose" / f"{self.stem}_pose.json").write_text(
            json.dumps(pose), "utf-8"
        )
        self.image_path = self.area / "data" / "rgb" / f"{self.stem}_rgb.png"
        Image.new("RGB", (10, 10)).save(self.image_path)
        Image.fromarray(np.full((10, 10), 1024, dtype=np.uint16)).save(
            self.area / "data" / "depth" / f"{self.stem}_depth.png"
        )

        annotations_dir = root / "annotations"
        annotations_dir.mkdir()
        record = {
            "image_path": str(self.image_path),
            "room": "office_1",
            "frame_id": 7,
            "uuid": "abcdef0123456789",
            "annotations": [
                {
                    "instance_name": "chair_1",
                    "class_name": "chair",
                    "bbox_xyxy": [1, 2, 5, 6],
                    "bbox_iou_with_uncropped": 0.8,
                    "pixel_count": 30,
                    "occlusion_ratio": 0.1,
                    "wraps_horizontal": False,
                },
                {
                    "instance_name": "clutter_1",
                    "class_name": "clutter",
                    "bbox_xyxy": [1, 1, 2, 2],
                    "bbox_iou_with_uncropped": 1,
                    "pixel_count": 30,
                    "occlusion_ratio": 0,
                },
            ],
        }
        (annotations_dir / f"{self.stem}_ann.json").write_text(
            json.dumps(record), "utf-8"
        )
        self.parser = RoomImageAnnotationsParser(annotations_dir)
        self.session = _Session()
        self.predictor = SAM3DBatchPredictor(
            SAM3DClient("https://example.invalid/infer", session=self.session),
            S23Dataset(self.area),
            self.parser,
        )
        self.output = root / "output"

    def tearDown(self):
        self.temp.cleanup()

    def test_prediction_filtering_camera_data_and_resume(self):
        summary = self.predictor.predict(self.output, progress=False)
        self.assertEqual(summary["annotations_seen"], 2)
        self.assertEqual(summary["selected"], 1)
        self.assertEqual(summary["success"], 1)
        self.assertEqual(len(self.session.calls), 1)
        data = self.session.calls[0][1]["data"]
        self.assertEqual(data["use_depth"], "true")
        self.assertIn("K", data)
        self.assertIn("Rt", data)
        self.assertEqual(data["return_mask"], "true")
        self.assertEqual(data["optimize_pose"], "true")

        request_id = f"{self.stem}_rgb_chair_1"
        self.assertTrue((self.output / f"{request_id}.glb").is_file())
        self.assertTrue((self.output / f"{request_id}_optimized.json").is_file())
        self.assertTrue((self.output / "batch_predict_summary.json").is_file())

        cached = self.predictor.predict(self.output, progress=False)
        self.assertEqual(cached["cached"], 1)
        self.assertEqual(len(self.session.calls), 1)

        (self.output / f"{request_id}_mask.png").unlink()
        refreshed = self.predictor.predict(self.output, progress=False)
        self.assertEqual(refreshed["success"], 1)
        self.assertEqual(len(self.session.calls), 2)


if __name__ == "__main__":
    unittest.main()
