"""Depth Anything 3 metric-depth inference helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import MetricMDEPredictor


DA3_MODEL = "depth-anything/da3metric-large"
DA3_REVISION = "4010e39f3634a45bc60553321fb49fb760bd594e"
DA3_PROCESS_RES = 504
DA3_PROCESS_METHOD = "upper_bound_resize"
DA3_PATCH_SIZE = 14
DA3_CANONICAL_FOCAL = 300.0


def _nearest_multiple(value, multiple=DA3_PATCH_SIZE):
    down = (value // multiple) * multiple
    up = down + multiple
    return max(multiple, up if abs(up - value) <= abs(value - down) else down)


def da3_processed_geometry(
    image_shape,
    intrinsics,
    process_res=DA3_PROCESS_RES,
    canonical_focal=DA3_CANONICAL_FOCAL,
):
    """Return DA3 input shape, resized intrinsics, and metric focal scale."""
    height, width = map(int, image_shape)
    if min(height, width, process_res) < 1:
        raise ValueError("image dimensions and process_res must be positive")

    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")

    scale = process_res / float(max(height, width))
    process_height = _nearest_multiple(round(height * scale))
    process_width = _nearest_multiple(round(width * scale))

    processed = intrinsics.copy()
    processed[0] *= process_width / width
    processed[1] *= process_height / height
    focal = float((processed[0, 0] + processed[1, 1]) / 2.0)
    focal_scale = focal / float(canonical_focal)
    if not np.isfinite(focal_scale) or focal_scale <= 0:
        raise ValueError(f"invalid DA3 focal scale: {focal_scale}")

    return process_height, process_width, processed, focal_scale


class DA3Predictor(MetricMDEPredictor):
    """Lazy DA3 metric predictor with an optional lossless raw-depth cache."""

    cache_depth_key = "canonical_depth"

    def __init__(
        self,
        *,
        device=None,
        local_files_only=False,
        cache_root=None,
        model_name=DA3_MODEL,
        revision=DA3_REVISION,
        process_res=DA3_PROCESS_RES,
    ):
        super().__init__(cache_root=cache_root)

        self.device_name = device
        self.local_files_only = bool(local_files_only)
        self.model_name = str(model_name)
        self.revision = str(revision)
        self.process_res = int(process_res)
        if self.process_res < 1:
            raise ValueError("process_res must be positive")

    def _model_cache_signature(self):
        return {
            "da3_model": self.model_name,
            "da3_revision": self.revision,
            "process_res": self.process_res,
            "process_res_method": DA3_PROCESS_METHOD,
            "feature_layers": [],
        }

    def _default_target_shape(self, image_path):
        return self.process_res, self.process_res

    def _load(self):
        if self.model is not None:
            return self.model

        import torch
        from depth_anything_3.api import DepthAnything3

        device_name = self.device_name
        if device_name is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        print(f"Loading DA3 on {device}...", flush=True)

        self.model = (
            DepthAnything3.from_pretrained(
                self.model_name,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
            .to(device)
            .eval()
        )

        loaded_revision = getattr(self.model, "_commit_hash", None)
        if loaded_revision not in {None, self.revision}:
            raise RuntimeError(
                f"Unexpected DA3 revision: {loaded_revision}, expected {self.revision}"
            )
        return self.model

    def _infer_raw(self, image_path, target_shape):
        result = self._load().inference(
            [str(image_path)],
            process_res=self.process_res,
            process_res_method=DA3_PROCESS_METHOD,
            export_dir=None,
        )
        return np.asarray(result.depth[0], dtype=np.float32)

    def predict_frame(self, frame, *, focal_correct=True):
        """Predict one calibrated frame at DA3's processed resolution."""
        height, width, _, focal_scale = da3_processed_geometry(
            frame.image_shape,
            frame.intrinsics,
            self.process_res,
        )
        scene_id = getattr(frame, "scene_id", "frame")
        frame_id = getattr(frame, "frame_id", Path(frame.rgb_path).stem)
        depth = self.predict_raw(
            frame.rgb_path,
            (height, width),
            cache_id=f"{scene_id}/{frame_id}",
        )
        return depth * focal_scale if focal_correct else depth