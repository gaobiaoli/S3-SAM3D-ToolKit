"""UniDepthV2 metric-depth inference helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .base import MetricMDEPredictor, resize_depth
from .dav3 import da3_processed_geometry

UNIDEPTHV2_MODEL = "lpiccinelli/unidepth-v2-vitl14"
UNIDEPTHV2_REVISION = "52b349b514bd8b47642f67ac78cb7b5dc5c51dd9"
UNIDEPTHV2_OUTPUT_SIZE = 504


def _validated_intrinsics(intrinsics):
    if intrinsics is None:
        return None

    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("intrinsics must be finite with positive focal lengths")
    return matrix


class UniDepthV2Predictor(MetricMDEPredictor):
    """Lazy ViT-L predictor returning metric depth in meters.

    ``predict_raw`` infers the camera if no intrinsics are supplied. ``predict_frame``
    passes the frame's calibrated intrinsics to UniDepthV2 and returns a depth map
    with a 504-pixel long edge by default. A supplied target shape changes only
    the returned depth size; intrinsics always refer to the input RGB.
    """

    cache_depth_key = "metric_depth"

    def __init__(
        self,
        *,
        device=None,
        local_files_only=False,
        cache_root=None,
        model_name=UNIDEPTHV2_MODEL,
        revision=UNIDEPTHV2_REVISION,
        resolution_level=None,
        output_size=UNIDEPTHV2_OUTPUT_SIZE,
    ):
        super().__init__(cache_root=cache_root)
        self.device_name = device
        self.local_files_only = bool(local_files_only)
        self.model_name = str(model_name)
        self.revision = None if revision is None else str(revision)
        self.resolution_level = resolution_level
        self.output_size = int(output_size)
        if resolution_level is not None and not (0 <= resolution_level < 10):
            raise ValueError("resolution_level must be in [0, 10)")
        if self.output_size < 1:
            raise ValueError("output_size must be positive")

    def _model_cache_signature(self):
        return {
            "unidepthv2_model": self.model_name,
            "unidepthv2_revision": self.revision,
            "resolution_level": self.resolution_level,
            "output_size": self.output_size,
            "input_camera": "pinhole_or_inferred",
        }

    def _default_target_shape(self, image_path):
        return self.output_size, self.output_size

    def _load(self):
        if self.model is not None:
            return self.model

        import torch
        from unidepth.models import UniDepthV2

        device_name = self.device_name
        if device_name is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        print(f"Loading UniDepthV2 on {device}...", flush=True)

        self.model = (
            UniDepthV2.from_pretrained(
                self.model_name,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
            .to(device)
            .eval()
        )
        if self.resolution_level is not None:
            self.model.resolution_level = self.resolution_level
        return self.model

    def _infer_raw(self, image_path, target_shape, *, intrinsics=None):
        import torch

        with Image.open(image_path) as image:
            rgb = np.array(image.convert("RGB"), copy=True)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        camera = None
        if intrinsics is not None:
            camera = torch.from_numpy(intrinsics.astype(np.float32))

        prediction = self._load().infer(rgb_tensor, camera)
        depth = prediction["depth"].detach().cpu().numpy()
        if depth.ndim != 4 or depth.shape[:2] != (1, 1):
            raise ValueError(f"Unexpected UniDepthV2 depth shape: {depth.shape}")
        return depth[0, 0]

    def predict_raw(self, image_path, target_shape=None, *, cache_id=None, intrinsics=None):
        """Predict metric depth, optionally conditioned on input-image intrinsics."""
        image_path = Path(image_path)
        intrinsics = _validated_intrinsics(intrinsics)
        if target_shape is None:
            target_shape = self._default_target_shape(image_path)
        target_shape = self._validate_target_shape(target_shape)

        # The same RGB can have different predictions under different cameras.
        camera_cache_id = cache_id
        if cache_id is not None and intrinsics is not None:
            digest = hashlib.sha256(intrinsics.astype("<f8").tobytes()).hexdigest()[:20]
            camera_cache_id = f"{cache_id}#K={digest}"
        path = self.cache_path(camera_cache_id)
        depth = self._read_cache(path, target_shape)
        if depth is not None:
            return depth

        depth = np.asarray(
            self._infer_raw(image_path, target_shape, intrinsics=intrinsics),
            dtype=np.float32,
        )
        depth = resize_depth(depth, target_shape)
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError(f"MDE model produced invalid depth: {image_path}")

        self._write_cache(path, image_path, camera_cache_id, depth)
        return depth

    def predict_frame(self, frame, *, focal_correct=True):
        """Predict one calibrated frame at DA3's 504-based output resolution.

        ``focal_correct`` is accepted for DA3 call-site compatibility; UniDepthV2
        already returns metric depth with the supplied camera, so it has no effect.
        """
        height, width, _, _ = da3_processed_geometry(
            frame.image_shape,
            frame.intrinsics,
            self.output_size,
        )
        scene_id = getattr(frame, "scene_id", "frame")
        frame_id = getattr(frame, "frame_id", Path(frame.rgb_path).stem)
        return self.predict_raw(
            frame.rgb_path,
            (height, width),
            cache_id=f"{scene_id}/{frame_id}",
            intrinsics=frame.intrinsics,
        )
