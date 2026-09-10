"""Depth Anything 3 metric-depth inference helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

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


def _resize_depth(depth, shape):
    if depth.shape == tuple(shape):
        return depth
    import cv2

    height, width = shape
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DA3Predictor:
    """Lazy DA3 metric predictor with an optional lossless raw-depth cache."""

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
        self.device_name = device
        self.local_files_only = bool(local_files_only)
        self.cache_root = None if cache_root is None else Path(cache_root).expanduser()
        self.model_name = str(model_name)
        self.revision = str(revision)
        self.process_res = int(process_res)
        if self.process_res < 1:
            raise ValueError("process_res must be positive")
        self.model = None

    @property
    def cache_signature(self):
        return {
            "schema_version": 1,
            "da3_model": self.model_name,
            "da3_revision": self.revision,
            "process_res": self.process_res,
            "process_res_method": DA3_PROCESS_METHOD,
            "feature_layers": [],
            "array_storage": "lossless_float32_uncompressed_npz",
        }

    @property
    def cache_namespace(self):
        if self.cache_root is None:
            return None
        payload = json.dumps(
            self.cache_signature,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return self.cache_root / hashlib.sha256(payload).hexdigest()[:20]

    def cache_path(self, cache_id):
        """Return the cache path for an identifier such as ``scene/frame``."""
        if self.cache_root is None or cache_id is None:
            return None
        cache_id = str(cache_id).strip("/")
        group = cache_id.split("/", 1)[0] if "/" in cache_id else "images"
        name = hashlib.sha256(cache_id.encode()).hexdigest() + ".npz"
        return self.cache_namespace / group / name

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

    @staticmethod
    def _read_cache(path, target_shape):
        if path is None or not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as item:
            depth = np.asarray(item["canonical_depth"], dtype=np.float32)
        valid = (
            depth.shape == tuple(target_shape) and np.isfinite(depth).all() and np.all(depth > 0)
        )
        return depth if valid else None

    def _write_cache(self, path, image_path, cache_id, depth):
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "signature": self.cache_signature,
            "rgb_sha256": _file_sha256(image_path),
            "depth_shape": list(depth.shape),
        }
        identity = str(cache_id).split("/", 1)
        if len(identity) == 2:
            metadata.update(scene_id=identity[0], frame_id=identity[1])
        else:
            metadata["cache_id"] = identity[0]
        with path.open("wb") as handle:
            np.savez(
                handle,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                canonical_depth=depth,
                has_confidence=np.asarray(False),
                raw_confidence=np.empty((0,), dtype=np.float32),
            )

    def predict_raw(self, image_path, target_shape=None, *, cache_id=None):
        """Return raw canonical-focal DA3 depth, without focal correction."""
        image_path = Path(image_path)
        if target_shape is None:
            target_shape = (self.process_res, self.process_res)
        target_shape = tuple(map(int, target_shape))
        if len(target_shape) != 2 or min(target_shape) < 1:
            raise ValueError("target_shape must contain two positive dimensions")
        path = self.cache_path(cache_id)
        depth = self._read_cache(path, target_shape)
        if depth is not None:
            return depth

        result = self._load().inference(
            [str(image_path)],
            process_res=self.process_res,
            process_res_method=DA3_PROCESS_METHOD,
            export_dir=None,
        )
        depth = np.asarray(result.depth[0], dtype=np.float32)
        depth = _resize_depth(depth, target_shape)
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError(f"DA3 produced invalid depth: {image_path}")

        self._write_cache(path, image_path, cache_id, depth)
        return depth

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
