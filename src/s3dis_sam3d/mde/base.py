# src/s3dis_sam3d/mde/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_SCHEMA_VERSION = 1
CACHE_ARRAY_STORAGE = "lossless_float32_uncompressed_npz"


def file_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def resize_depth(depth, shape):
    depth = np.asarray(depth, dtype=np.float32)

    if depth.shape == tuple(shape):
        return depth

    import cv2

    height, width = shape

    return cv2.resize(
        depth,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


class MetricMDEPredictor(ABC):
    """
    Common interface for cached metric monocular-depth predictors.

    Subclasses only need to define:
        1. model-specific cache signature
        2. default output shape
        3. model loading
        4. raw inference
    """

    cache_depth_key = "raw_depth"

    def __init__(
        self,
        *,
        cache_root=None,
    ):
        self.cache_root = None if cache_root is None else Path(cache_root).expanduser()

        self.model = None

    # ============================================================
    # Cache identity
    # ============================================================

    @property
    def cache_signature(self):
        """
        Everything affecting the numerical prediction should appear
        in this signature.
        """
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            **self._model_cache_signature(),
            "array_storage": CACHE_ARRAY_STORAGE,
        }

    @abstractmethod
    def _model_cache_signature(self):
        """
        Return model-specific fields affecting prediction.

        Example:
            model name
            checkpoint revision
            inference resolution
            preprocessing method
        """
        raise NotImplementedError

    @property
    def cache_namespace(self):
        if self.cache_root is None:
            return None

        payload = json.dumps(
            self.cache_signature,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

        namespace = hashlib.sha256(payload).hexdigest()[:20]

        return self.cache_root / namespace

    def cache_path(self, cache_id):
        """
        Example:

            cache_id = "scene/frame"

        produces:

            <cache_root>/<namespace>/scene/<sha256(cache_id)>.npz
        """

        if self.cache_root is None or cache_id is None:
            return None

        cache_id = str(cache_id).strip("/")

        if "/" in cache_id:
            group = cache_id.split("/", 1)[0]
        else:
            group = "images"

        filename = hashlib.sha256(cache_id.encode()).hexdigest() + ".npz"

        return self.cache_namespace / group / filename

    # ============================================================
    # Cache I/O
    # ============================================================

    def _read_cache(
        self,
        path,
        target_shape,
    ):
        if path is None or not path.is_file():
            return None

        try:
            with np.load(
                path,
                allow_pickle=False,
            ) as item:

                if self.cache_depth_key not in item:
                    return None

                depth = np.asarray(
                    item[self.cache_depth_key],
                    dtype=np.float32,
                )

        except (
            OSError,
            ValueError,
            KeyError,
        ):
            return None

        valid = (
            depth.shape == tuple(target_shape)
            and np.isfinite(depth).all()
            and np.all(depth > 0)
        )

        if not valid:
            return None

        return depth

    def _cache_metadata(
        self,
        *,
        image_path,
        cache_id,
        depth,
    ):
        metadata = {
            "signature": self.cache_signature,
            "rgb_sha256": file_sha256(image_path),
            "depth_shape": list(depth.shape),
        }

        identity = str(cache_id).split("/", 1)

        if len(identity) == 2:
            metadata.update(
                scene_id=identity[0],
                frame_id=identity[1],
            )
        else:
            metadata["cache_id"] = identity[0]

        return metadata

    def _write_cache(
        self,
        path,
        image_path,
        cache_id,
        depth,
    ):
        if path is None:
            return

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata = self._cache_metadata(
            image_path=image_path,
            cache_id=cache_id,
            depth=depth,
        )

        with path.open("wb") as handle:
            np.savez(
                handle,
                metadata=np.asarray(
                    json.dumps(
                        metadata,
                        sort_keys=True,
                    )
                ),
                **{
                    self.cache_depth_key: depth.astype(
                        np.float32,
                        copy=False,
                    )
                },
                has_confidence=np.asarray(False),
                raw_confidence=np.empty(
                    (0,),
                    dtype=np.float32,
                ),
            )

    # ============================================================
    # Prediction template
    # ============================================================

    @abstractmethod
    def _default_target_shape(
        self,
        image_path,
    ):
        raise NotImplementedError

    @abstractmethod
    def _load(self):
        raise NotImplementedError

    @abstractmethod
    def _infer_raw(
        self,
        image_path,
        target_shape,
    ):
        """
        Run model-specific inference.

        Return:
            np.ndarray [H, W]

        It does not need to already match target_shape.
        """
        raise NotImplementedError

    @staticmethod
    def _validate_target_shape(
        target_shape,
    ):
        target_shape = tuple(map(int, target_shape))

        if len(target_shape) != 2 or min(target_shape) < 1:
            raise ValueError("target_shape must contain " "two positive dimensions")

        return target_shape

    def predict_raw(
        self,
        image_path,
        target_shape=None,
        *,
        cache_id=None,
    ):
        """
        Generic cached raw prediction.
        """

        image_path = Path(image_path)

        if target_shape is None:
            target_shape = self._default_target_shape(image_path)

        target_shape = self._validate_target_shape(target_shape)

        path = self.cache_path(cache_id)

        depth = self._read_cache(
            path,
            target_shape,
        )

        if depth is not None:
            return depth

        depth = self._infer_raw(
            image_path,
            target_shape,
        )

        depth = np.asarray(
            depth,
            dtype=np.float32,
        )

        depth = resize_depth(
            depth,
            target_shape,
        )

        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError(f"MDE model produced invalid depth: " f"{image_path}")

        self._write_cache(
            path,
            image_path,
            cache_id,
            depth,
        )

        return depth
