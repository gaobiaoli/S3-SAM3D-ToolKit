"""MoGe 2 and MoGe 3 metric-depth inference helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .base import MetricMDEPredictor, resize_depth
from .dav3 import da3_processed_geometry

MOGE_OUTPUT_SIZE = 504
MOGE2_MODEL = "Ruicheng/moge-2-vitl"
MOGE2_REVISION = "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"
MOGE3_MODEL = "Ruicheng/moge-3-vitl"
MOGE3_REVISION = "184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be"


def _validated_intrinsics(intrinsics):
    if intrinsics is None:
        return None

    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("intrinsics must be finite with positive focal lengths")
    return matrix


def moge_fov_x(image_shape, intrinsics):
    """Convert pixel-space pinhole intrinsics to MoGe's horizontal FoV in degrees."""
    _, width = map(int, image_shape)
    if width < 1:
        raise ValueError("image dimensions must be positive")
    intrinsics = _validated_intrinsics(intrinsics)
    return float(np.degrees(2.0 * np.arctan(width / (2.0 * intrinsics[0, 0]))))


def _depth_map(prediction):
    if "depth" not in prediction:
        raise ValueError("MoGe prediction does not contain depth")
    depth = prediction["depth"].detach().float().cpu().numpy()
    while depth.ndim > 2 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Unexpected MoGe depth shape: {depth.shape}")
    return depth


class _MoGePredictor(MetricMDEPredictor):
    """Shared cached metric-depth adapter for MoGe 2 and MoGe 3."""

    cache_depth_key = "metric_depth"
    model_version = None
    default_model_name = None
    default_revision = None

    def __init__(
        self,
        *,
        device=None,
        local_files_only=False,
        cache_root=None,
        model_name=None,
        revision=None,
        output_size=MOGE_OUTPUT_SIZE,
        resolution_level=9,
        num_tokens=None,
        use_fp16=False,
    ):
        super().__init__(cache_root=cache_root)
        self.device_name = device
        self.local_files_only = bool(local_files_only)
        self.model_name = str(model_name or self.default_model_name)
        self.revision = self.default_revision if revision is None else str(revision)
        self.output_size = int(output_size)
        self.resolution_level = int(resolution_level)
        self.num_tokens = None if num_tokens is None else int(num_tokens)
        self.use_fp16 = bool(use_fp16)

        if self.output_size < 1:
            raise ValueError("output_size must be positive")
        if not 0 <= self.resolution_level <= 9:
            raise ValueError("resolution_level must be in [0, 9]")
        if self.num_tokens is not None and self.num_tokens < 1:
            raise ValueError("num_tokens must be positive")

    def _model_cache_signature(self):
        return {
            "moge_version": self.model_version,
            "moge_model": self.model_name,
            "moge_revision": self.revision,
            "output_size": self.output_size,
            "resolution_level": self.resolution_level,
            "num_tokens": self.num_tokens,
            "use_fp16": self.use_fp16,
            "apply_mask": False,
            "input_camera": "horizontal_fov_or_inferred",
            **self._extra_cache_signature(),
        }

    def _extra_cache_signature(self):
        return {}

    def _default_target_shape(self, image_path):
        return self.output_size, self.output_size

    def _model_class(self):
        if self.model_version == 3:
            from moge.model.v3 import MoGeModel
        elif self.model_version == 2:
            from moge.model.v2 import MoGeModel
        else:  # pragma: no cover - subclasses define a fixed supported version
            raise RuntimeError(f"Unsupported MoGe version: {self.model_version}")
        return MoGeModel

    def _load(self):
        if self.model is not None:
            return self.model

        import torch

        device_name = self.device_name
        if device_name is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        print(f"Loading MoGe {self.model_version} on {device}...", flush=True)

        self.model = (
            self._model_class()
            .from_pretrained(
                self.model_name,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
            .to(device)
            .eval()
        )
        return self.model

    def _infer_kwargs(self):
        kwargs = {
            "resolution_level": self.resolution_level,
            "apply_mask": False,
            "use_fp16": self.use_fp16,
        }
        if self.num_tokens is not None:
            kwargs["num_tokens"] = self.num_tokens
        return kwargs

    def _infer_raw(self, image_path, target_shape, *, intrinsics=None):
        import torch

        with Image.open(image_path) as image:
            rgb = np.array(image.convert("RGB"), copy=True)
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)

        kwargs = self._infer_kwargs()
        if intrinsics is not None:
            kwargs["fov_x"] = moge_fov_x(rgb.shape[:2], intrinsics)
        prediction = self._load().infer(image_tensor, **kwargs)
        return _depth_map(prediction)

    def predict_raw(self, image_path, target_shape=None, *, cache_id=None, intrinsics=None):
        """Predict metric depth, optionally using pixel-space pinhole intrinsics."""
        image_path = Path(image_path)
        intrinsics = _validated_intrinsics(intrinsics)
        if target_shape is None:
            target_shape = self._default_target_shape(image_path)
        target_shape = self._validate_target_shape(target_shape)

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
        """Predict a calibrated frame with a 504-pixel long edge by default."""
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


class MoGe3Predictor(_MoGePredictor):
    """MoGe 3 ViT-L metric-depth predictor with sparse refinement."""

    model_version = 3
    default_model_name = MOGE3_MODEL
    default_revision = MOGE3_REVISION

    def __init__(self, *, refine_steps=0, **kwargs):
        self.refine_steps = int(refine_steps)
        if self.refine_steps < 0:
            raise ValueError("refine_steps must be non-negative")
        super().__init__(**kwargs)

    def _extra_cache_signature(self):
        return {"refine_steps": self.refine_steps}

    def _infer_kwargs(self):
        return {**super()._infer_kwargs(), "refine_steps": self.refine_steps}


class MoGe2Predictor(_MoGePredictor):
    """MoGe 2 ViT-L metric-depth predictor."""

    model_version = 2
    default_model_name = MOGE2_MODEL
    default_revision = MOGE2_REVISION
