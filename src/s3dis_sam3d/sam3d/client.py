from __future__ import annotations

import base64
import json
import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests


@dataclass(frozen=True)
class SAM3DResult:
    request_id: str
    glb_path: Path
    pose_path: Path | None
    mask_path: Path | None
    optimized_pose_path: Path | None
    cached: bool = False


class SAM3DClient:
    """HTTP client for the existing multipart SAM3D inference service."""

    def __init__(self, url: str, *, timeout: float = 600.0, session: requests.Session | None = None):
        self.url = url
        self.timeout = timeout
        self.session = session or requests.Session()

    @staticmethod
    def _paths(folder: Path, request_id: str) -> dict[str, Path]:
        return {
            "glb": folder / f"{request_id}.glb",
            "pose": folder / f"{request_id}.json",
            "mask": folder / f"{request_id}_mask.png",
            "optimized": folder / f"{request_id}_optimized.json",
        }

    def _cache_hit(
        self,
        cache_dir: Path,
        output_dir: Path,
        request_id: str,
        *,
        require_mask=False,
        require_optimized=False,
    ) -> SAM3DResult | None:
        source = self._paths(cache_dir, request_id)
        if not source["glb"].is_file() or not source["pose"].is_file():
            return None
        if require_mask and not source["mask"].is_file():
            return None
        if require_optimized and not source["optimized"].is_file():
            return None
        target = self._paths(output_dir, request_id)
        for key, source_path in source.items():
            if source_path.is_file() and source_path.resolve() != target[key].resolve():
                shutil.copy2(source_path, target[key])
        return SAM3DResult(
            request_id,
            target["glb"],
            target["pose"],
            target["mask"] if target["mask"].exists() else None,
            target["optimized"] if target["optimized"].exists() else None,
            cached=True,
        )

    def infer(
        self,
        image_path: str | Path,
        *,
        request_id: str,
        output_dir: str | Path,
        mask_path: str | Path | None = None,
        bbox: list[float] | np.ndarray | None = None,
        depth_path: str | Path | None = None,
        use_depth: bool = False,
        intrinsics: np.ndarray | None = None,
        camera_transform: np.ndarray | None = None,
        depth_scale: float = 512.0,
        invalid_depth_value: float = 65535.0,
        seed: int = 42,
        return_mask: bool = False,
        optimize_pose: bool = False,
        optimize_iterations: int = 300,
        cache_dir: str | Path | None = None,
        cache_only: bool = False,
    ) -> SAM3DResult | None:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not image_path.is_file():
            raise FileNotFoundError(f"image not found: {image_path}")
        if mask_path is None and bbox is None:
            raise ValueError("either mask_path or bbox is required")
        cache_path = Path(cache_dir) if cache_dir is not None else None
        if cache_path is not None:
            cache_path.mkdir(parents=True, exist_ok=True)
            hit = self._cache_hit(
                cache_path,
                output_dir,
                request_id,
                require_mask=return_mask,
                require_optimized=optimize_pose,
            )
            if hit is not None:
                return hit
        if cache_only:
            return None

        files = {}
        for field, value in (
            ("image", image_path),
            ("mask", mask_path),
            ("depth_image", depth_path),
        ):
            if value is not None:
                path = Path(value)
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                files[field] = (path.name, path.read_bytes(), content_type)

        data = {
            "request_id": request_id,
            "depth_scale": depth_scale,
            "invalid_depth_value": invalid_depth_value,
            "seed": seed,
            "use_depth": str(use_depth).lower(),
        }
        if bbox is not None:
            data["bbox"] = json.dumps(np.asarray(bbox).tolist())
        if intrinsics is not None:
            data["K"] = json.dumps(np.asarray(intrinsics).tolist())
        if camera_transform is not None:
            data["Rt"] = json.dumps(np.asarray(camera_transform).tolist())
        if return_mask:
            data["return_mask"] = "true"
        if optimize_pose:
            data["optimize_pose"] = "true"
            data["optimize_num_iterations"] = optimize_iterations

        response = self.session.post(self.url, files=files, data=data, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        actual_id = str(payload.get("request_id") or request_id)
        paths = self._paths(output_dir, actual_id)
        paths["glb"].write_bytes(base64.b64decode(payload["glb_b64"]))
        written: dict[str, Path | None] = {"pose": None, "mask": None, "optimized": None}
        for payload_key, path_key in (("pose", "pose"), ("pose_optimized", "optimized")):
            if payload.get(payload_key) is not None:
                paths[path_key].write_text(json.dumps(payload[payload_key], indent=2), "utf-8")
                written[path_key] = paths[path_key]
        if payload.get("mask_png_b64"):
            paths["mask"].write_bytes(base64.b64decode(payload["mask_png_b64"]))
            written["mask"] = paths["mask"]
        if cache_path is not None:
            cache_paths = self._paths(cache_path, actual_id)
            for key in ("glb", "pose", "mask", "optimized"):
                if (
                    paths[key].is_file()
                    and paths[key].resolve() != cache_paths[key].resolve()
                ):
                    shutil.copy2(paths[key], cache_paths[key])
        return SAM3DResult(actual_id, paths["glb"], written["pose"], written["mask"], written["optimized"])
