import base64
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from s3dis_sam3d.sam3d.client import SAM3DClient


class _Response:
    status_code = 200
    text = ""

    @staticmethod
    def raise_for_status():
        pass

    @staticmethod
    def json():
        return {
            "request_id": "request_1",
            "glb_b64": base64.b64encode(b"mock-glb").decode("ascii"),
            "mask_png_b64": base64.b64encode(b"mock-png").decode("ascii"),
            "pose": {
                "object_0": {
                    "rotation": [1, 0, 0, 0],
                    "translation": [0, 0, 0],
                    "scale": [1, 1, 1],
                }
            },
        }


class _Session:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response()


class SAM3DClientTest(unittest.TestCase):
    def test_inference_writes_files_and_populates_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "image.png"
            Image.new("RGB", (2, 2)).save(image)
            session = _Session()
            client = SAM3DClient("https://example.invalid/infer", session=session)
            result = client.infer(
                image,
                request_id="request_1",
                output_dir=root / "first",
                bbox=[0, 0, 1, 1],
                return_mask=True,
                cache_dir=root / "cache",
            )
            self.assertEqual(result.glb_path.read_bytes(), b"mock-glb")
            self.assertTrue(result.pose_path.is_file())
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(session.calls[0][1]["data"]["bbox"], "[0, 0, 1, 1]")

            cached = client.infer(
                image,
                request_id="request_1",
                output_dir=root / "second",
                bbox=[0, 0, 1, 1],
                cache_dir=root / "cache",
                cache_only=True,
            )
            self.assertTrue(cached.cached)
            self.assertEqual(cached.glb_path.read_bytes(), b"mock-glb")
            self.assertEqual(len(session.calls), 1)


if __name__ == "__main__":
    unittest.main()
