"""Matterport3D frame, panorama, scene, and dataset helpers.

This module follows the same data model as :mod:`s3dis_sam3d.s23dis`: frames
own their RGB/depth data, point-cloud operations return the shared
``PointCloud`` model, and scenes reuse the package's geometry operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import open3d as o3d

from .frames import RGBDFrame
from .config import MATTERPORT_ROOT
from .models import PointCloud
from .pointcloud import visualize_point_clouds, voxel_downsample
from .utils import backproject_regular, project_pinhole_points

MATTERPORT_DEPTH_SCALE = 4000.0
# The undistorted .conf pose maps Matterport/OpenGL camera coordinates
# (x right, y up, looking along -z) to world coordinates. Pinhole depth
# backprojection uses CV coordinates (x right, y down, looking along +z).
MATTERPORT_CV_TO_OPENGL = np.diag([1, -1, -1, 1]).astype(np.float32)

def _mask_for_frame(mask, frame: MatterportFrame):
    if mask is None:
        return None
    if isinstance(mask, Mapping):
        return mask.get(frame.frame_id)
    if callable(mask):
        return mask(frame)
    return mask


@dataclass(frozen=True)
class MatterportFrame(RGBDFrame):
    """One Matterport perspective RGB-D frame and its calibrated pose."""

    depth_scale = MATTERPORT_DEPTH_SCALE

    scene_id: str
    panorama_id: str
    camera_index: int
    yaw_index: int
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    distortion: np.ndarray | None = None
    undistorted: bool = True
    _undistorted_source: MatterportFrame | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def frame_id(self) -> str:
        return f"{self.panorama_id}_i{self.camera_index}_{self.yaw_index}"

    @cached_property
    def _original_frame(self) -> MatterportFrame:
        root = self.rgb_path.parent.parent
        rgb_path = root / "matterport_color_images" / self.rgb_path.name
        depth_path = root / "matterport_depth_images" / self.depth_path.name
        intrinsics_path = (
            root
            / "matterport_camera_intrinsics"
            / f"{self.panorama_id}_intrinsics_{self.camera_index}.txt"
        )
        pose_path = (
            root
            / "matterport_camera_poses"
            / f"{self.panorama_id}_pose_{self.camera_index}_{self.yaw_index}.txt"
        )
        missing = [
            path
            for path in (rgb_path, depth_path, intrinsics_path, pose_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Original Matterport data for frame {self.frame_id!r} is incomplete: "
                + ", ".join(str(path) for path in missing)
            )

        calibration = np.loadtxt(intrinsics_path, dtype=np.float64).reshape(-1)
        if calibration.size < 11:
            raise ValueError(
                f"Expected 11 calibration values in {intrinsics_path}, "
                f"got {calibration.size}"
            )
        _, _, fx, fy, cx, cy = calibration[:6]
        pose = np.loadtxt(pose_path, dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 camera pose in {pose_path}, got {pose.shape}")

        return replace(
            self,
            rgb_path=rgb_path,
            depth_path=depth_path,
            intrinsics=np.asarray(
                ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)),
                dtype=np.float32,
            ),
            camera_to_world=pose,
            distortion=calibration[6:11].astype(np.float32),
            undistorted=False,
            _undistorted_source=self,
        )

    def with_undistort(self, undistort: bool = True) -> MatterportFrame:
        """Return this frame's undistorted or original calibrated variant."""

        if not isinstance(undistort, bool):
            raise TypeError("undistort must be a bool")
        if undistort == self.undistorted:
            return self
        if undistort:
            if self._undistorted_source is None:
                raise RuntimeError("the undistorted source frame is unavailable")
            return self._undistorted_source
        return self._original_frame

    def project_camera_points(self, points: np.ndarray):
        """Project camera-space points to ``(uv, depth)``."""

        return project_pinhole_points(points, self.intrinsics)

    def _backproject(
        self,
        *,
        stride: int = 1,
        depth_min: float | None = 0.1,
        depth_max: float | None = 10.0,
        mask=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        depth = self.depth
        points, ys, xs = backproject_regular(
            depth,
            self.intrinsics,
            stride=stride,
            depth_min=depth_min,
            depth_max=depth_max,
            mask=_mask_for_frame(mask, self),
        )
        return points, ys, xs, depth

    def _point_cloud_metadata(self, coordinate_frame):
        return {
            "frame_id": self.frame_id,
            "panorama_id": self.panorama_id,
            "coordinate_frame": coordinate_frame,
        }


@dataclass(frozen=True)
class MatterportPanorama:
    """The perspective views captured at one Matterport panorama position."""

    scene_id: str
    panorama_id: str
    frames: tuple[MatterportFrame, ...]

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, index):
        if isinstance(index, str):
            for frame in self.frames:
                if frame.frame_id == index:
                    return frame
            raise KeyError(index)
        return self.frames[index]

    @property
    def camera_position(self) -> np.ndarray:
        positions = np.stack([frame.camera_position for frame in self.frames])
        return positions.mean(axis=0)


class MatterportScene:
    """A lazily parsed Matterport3D scene directory."""

    def __init__(self, scene_id: str, scene_root: str | Path) -> None:
        self.scene_id = scene_id
        self.scene_root = Path(scene_root)
        if not self.scene_root.is_dir():
            raise FileNotFoundError(f"Matterport scene directory not found: {self.scene_root}")
        self.scene_dir = self.scene_root
        self.raw_mesh_dir = self.scene_root / "matterport_mesh"
        self.poisson_mesh_dir = self.scene_root / "poisson_meshes"
        self._frames: tuple[MatterportFrame, ...] | None = None
        self._panoramas: dict[str, MatterportPanorama] | None = None
        self._raw_mesh: o3d.geometry.TriangleMesh | None = None

    @property
    def camera_config_path(self) -> Path:
        candidates = (
            self.scene_root / "undistorted_camera_parameters" / f"{self.scene_id}.conf",
            self.scene_root / f"{self.scene_id}.conf",
        )
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"Camera parameter file for scene {self.scene_id!r} was not found under "
            f"{self.scene_root}"
        )

    @property
    def camera_file(self) -> Path:
        """Compatibility alias for :attr:`camera_config_path`."""

        return self.camera_config_path

    @property
    def rgb_dir(self) -> Path:
        return self._find_data_dir("undistorted_color_images", "matterport_color_images")

    @property
    def depth_dir(self) -> Path:
        return self._find_data_dir("undistorted_depth_images", "matterport_depth_images")

    def _find_data_dir(self, *names: str) -> Path:
        for name in names:
            path = self.scene_root / name
            if path.is_dir():
                return path
        raise FileNotFoundError(
            f"None of the expected directories {names!r} exists under {self.scene_root}"
        )

    @property
    def raw_mesh_path(self) -> Path:
        candidates = (
            self.scene_root / "matterport_mesh" / f"{self.scene_id}.obj",
            self.scene_root / "house_segmentations" / f"{self.scene_id}.obj",
            self.scene_root / f"{self.scene_id}.obj",
        )
        for path in candidates:
            if path.is_file():
                return path
        matches = []
        for mesh_dir in (
            self.scene_root / "matterport_mesh",
            self.scene_root / "house_segmentations",
        ):
            if mesh_dir.is_dir():
                # The official matterport_mesh archive stores the mesh below a
                # hash directory: matterport_mesh/<mesh_hash>/<mesh_hash>.obj.
                matches.extend(mesh_dir.rglob("*.obj"))
        matches = sorted(set(matches))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            preview = ", ".join(str(path) for path in matches[:3])
            raise ValueError(
                f"Multiple raw OBJ meshes were found for scene {self.scene_id!r}: {preview}"
            )
        raise FileNotFoundError(f"Raw OBJ mesh for scene {self.scene_id!r} was not found")

    @property
    def has_raw_mesh(self) -> bool:
        try:
            _ = self.raw_mesh_path
        except FileNotFoundError:
            return False
        return True

    def load_raw_mesh(self, *, force_reload: bool = False) -> o3d.geometry.TriangleMesh:
        """Load and cache the original textured Matterport OBJ mesh."""

        if self._raw_mesh is not None and not force_reload:
            return self._raw_mesh
        mesh = o3d.io.read_triangle_mesh(
            str(self.raw_mesh_path),
            enable_post_processing=False,
        )
        if mesh.is_empty():
            raise ValueError(f"Open3D could not read a non-empty mesh from {self.raw_mesh_path}")
        self._raw_mesh = mesh
        return mesh

    def unload_raw_mesh(self) -> None:
        self._raw_mesh = None

    @property
    def frames(self) -> tuple[MatterportFrame, ...]:
        self._ensure_cameras()
        assert self._frames is not None
        return self._frames

    @property
    def panoramas(self) -> dict[str, MatterportPanorama]:
        self._ensure_cameras()
        assert self._panoramas is not None
        return self._panoramas

    def _ensure_cameras(self) -> None:
        if self._frames is None:
            self._frames, self._panoramas = self._parse_camera_config()

    def _parse_camera_config(
        self,
    ) -> tuple[tuple[MatterportFrame, ...], dict[str, MatterportPanorama]]:
        rgb_dir = self.rgb_dir
        depth_dir = self.depth_dir
        intrinsics = None
        panorama_frames: dict[str, list[MatterportFrame]] = {}
        panorama_order: list[str] = []
        frames: list[MatterportFrame] = []

        with self.camera_config_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                tokens = raw_line.strip().split()
                if not tokens or tokens[0].startswith("#"):
                    continue
                if tokens[0] == "intrinsics_matrix":
                    intrinsics = self._parse_matrix(tokens[1:], 3, line_number).astype(np.float32)
                    continue
                if tokens[0] != "scan":
                    continue
                if intrinsics is None:
                    raise ValueError(
                        f"scan appears before intrinsics_matrix at "
                        f"{self.camera_config_path}:{line_number}"
                    )
                if len(tokens) < 19:
                    raise ValueError(
                        f"Invalid scan row at {self.camera_config_path}:{line_number}"
                    )

                depth_name, rgb_name = tokens[1], tokens[2]
                pose = self._parse_matrix(tokens[3:19], 4, line_number).astype(np.float32)
                panorama_id, camera_index, yaw_index = self._parse_rgb_filename(rgb_name)
                frame = MatterportFrame(
                    scene_id=self.scene_id,
                    panorama_id=panorama_id,
                    camera_index=camera_index,
                    yaw_index=yaw_index,
                    rgb_path=rgb_dir / rgb_name,
                    depth_path=depth_dir / depth_name,
                    intrinsics=intrinsics.copy(),
                    camera_to_world=pose @ MATTERPORT_CV_TO_OPENGL,
                )
                frames.append(frame)
                if panorama_id not in panorama_frames:
                    panorama_frames[panorama_id] = []
                    panorama_order.append(panorama_id)
                panorama_frames[panorama_id].append(frame)

        if not frames:
            raise ValueError(f"No scan entries found in {self.camera_config_path}")
        missing = [
            path
            for frame in frames
            for path in (frame.rgb_path, frame.depth_path)
            if not path.is_file()
        ]
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            suffix = " ..." if len(missing) > 3 else ""
            raise FileNotFoundError(f"{len(missing)} frame files are missing: {preview}{suffix}")

        panoramas = {
            panorama_id: MatterportPanorama(
                    self.scene_id,
                    panorama_id,
                    tuple(panorama_frames[panorama_id]),
            )
            for panorama_id in panorama_order
        }
        return tuple(frames), panoramas

    @staticmethod
    def _parse_rgb_filename(filename: str) -> tuple[str, int, int]:
        stem = Path(filename).stem
        try:
            prefix, yaw = stem.rsplit("_", 1)
            panorama_id, camera = prefix.rsplit("_i", 1)
            return panorama_id, int(camera), int(yaw)
        except (ValueError, IndexError) as error:
            raise ValueError(f"Invalid Matterport RGB filename: {filename}") from error

    def _parse_matrix(self, tokens: list[str], size: int, line_number: int) -> np.ndarray:
        expected = size * size
        if len(tokens) < expected:
            raise ValueError(
                f"Expected {expected} matrix values at "
                f"{self.camera_config_path}:{line_number}, got {len(tokens)}"
            )
        try:
            return np.asarray(tokens[:expected], dtype=np.float64).reshape(size, size)
        except ValueError as error:
            raise ValueError(
                f"Invalid matrix at {self.camera_config_path}:{line_number}"
            ) from error

    def get_frame(self, frame_id: str, *, undistort: bool = True) -> MatterportFrame:
        """Return a calibrated frame from the undistorted or original images.

        ``undistort=True`` preserves the dataset's default undistorted RGB-D
        representation. ``False`` selects the original Matterport images and
        loads their matching intrinsics, distortion coefficients, and pose.
        """

        for frame in self.frames:
            if frame.frame_id == frame_id:
                return frame.with_undistort(undistort)
        raise KeyError(frame_id)

    def get_panorama(self, panorama_id: str) -> MatterportPanorama:
        try:
            return self.panoramas[panorama_id]
        except KeyError as error:
            raise KeyError(panorama_id) from error

    def reconstruct(
        self,
        *,
        stride: int = 4,
        depth_min: float | None = 0.1,
        depth_max: float | None = 8.0,
        voxel_size: float | None = 0.03,
        world_coordinates: bool = True,
        mask=None,
        max_frames: int | None = None,
        progress: bool = False,
    ) -> PointCloud:
        """Reconstruct the scene from calibrated RGB-D frames.

        When voxel fusion is enabled, each frame is downsampled before the
        global pass. This keeps a full Matterport scene from retaining every
        sampled pixel in memory while still using the package's common voxel
        implementation. Set ``voxel_size=None`` for an unmodified concatenation.
        """

        frames = list(self.frames)
        if isinstance(mask, Mapping):
            frames = [frame for frame in frames if frame.frame_id in mask]
        if max_frames is not None:
            if max_frames < 1:
                raise ValueError("max_frames must be positive")
            frames = frames[:max_frames]

        iterator = frames
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(frames, desc=f"Reconstructing {self.scene_id}")
        clouds = []
        for frame in iterator:
            cloud = frame.point_cloud(
                stride=stride,
                depth_min=depth_min,
                depth_max=depth_max,
                world_coordinates=world_coordinates,
                mask=mask,
            )
            if cloud.xyz.shape[0] == 0:
                continue
            clouds.append(voxel_downsample(cloud, voxel_size))

        metadata = {
            "scene_id": self.scene_id,
            "frame_count": len(clouds),
            "selected_frame_count": len(frames),
            "coordinate_frame": "world" if world_coordinates else "camera",
            "stride": stride,
            "depth_min": depth_min,
            "depth_max": depth_max,
            "voxel_size": voxel_size,
        }
        if not clouds:
            raise ValueError(
                f"No valid depth points remained for scene {self.scene_id}"
            )

        cloud = PointCloud(
            xyz=np.concatenate([item.xyz for item in clouds], axis=0),
            rgb=np.concatenate([item.rgb for item in clouds], axis=0),
            metadata=metadata,
        )
        return voxel_downsample(cloud, voxel_size)

    def visualize(
        self,
        *,
        show_raw_mesh: bool = False,
        point_size: float = 2.0,
        **reconstruct_options,
    ) -> PointCloud:
        """Reconstruct and display the scene, optionally with its raw mesh."""

        cloud = self.reconstruct(**reconstruct_options)
        geometries = [cloud]
        if show_raw_mesh:
            geometries.append(self.load_raw_mesh())
        visualize_point_clouds(geometries, point_size=point_size)
        return cloud

    def save_ply(self, output_path: str | Path, **reconstruct_options) -> Path:
        """Reconstruct the scene and save it through the shared PLY writer."""

        from .io import write_ply

        return write_ply(output_path, self.reconstruct(**reconstruct_options))

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.get_frame(index)
        return self.frames[index]

    def __repr__(self) -> str:
        camera_state = "loaded" if self._frames is not None else "not loaded"
        mesh_state = "loaded" if self._raw_mesh is not None else "not loaded"
        return (
            f"MatterportScene(scene_id={self.scene_id!r}, "
            f"cameras={camera_state}, raw_mesh={mesh_state})"
        )


class Matterport3DDataset:
    """Index Matterport scenes in either flat or official nested layouts."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(MATTERPORT_ROOT if root is None else root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Matterport root directory not found: {self.root}")
        self._scene_dirs = self._discover_scene_dirs()
        if not self._scene_dirs:
            raise FileNotFoundError(f"No Matterport scenes found below {self.root}")
        self._scene_cache: dict[str, MatterportScene] = {}

    def _discover_scene_dirs(self) -> dict[str, Path]:
        roots = [self.root]
        for relative in (Path("v1/scans"), Path("scans")):
            candidate = self.root / relative
            if candidate.is_dir():
                roots.append(candidate)

        found: dict[str, Path] = {}
        for base in roots:
            for path in sorted(item for item in base.iterdir() if item.is_dir()):
                nested = path / path.name
                candidates = [path]
                if nested.is_dir():
                    candidates.append(nested)
                scene_dir = max(
                    candidates,
                    key=lambda candidate: self._scene_score(candidate, path.name),
                )
                score = self._scene_score(scene_dir, path.name)
                current = found.get(path.name)
                if score > 0 and (
                    current is None or score > self._scene_score(current, current.name)
                ):
                    found[path.name] = scene_dir
        return found

    @staticmethod
    def _scene_score(path: Path, scene_id: str) -> int:
        score = 0
        if (path / "undistorted_camera_parameters" / f"{scene_id}.conf").is_file():
            score += 4
        if (path / "matterport_color_images").is_dir() or (
            path / "undistorted_color_images"
        ).is_dir():
            score += 2
        if (path / "matterport_depth_images").is_dir() or (
            path / "undistorted_depth_images"
        ).is_dir():
            score += 2
        if (path / "house_segmentations" / f"{scene_id}.obj").is_file():
            score += 1
        if (path / "matterport_mesh" / f"{scene_id}.obj").is_file():
            score += 1
        matterport_mesh_dir = path / "matterport_mesh"
        if matterport_mesh_dir.is_dir() and any(matterport_mesh_dir.rglob("*.obj")):
            score += 1
        return score

    @property
    def scene_ids(self) -> list[str]:
        return sorted(self._scene_dirs)

    def list_scenes(self) -> list[str]:
        return self.scene_ids

    def get_scene(self, scene_id: str) -> MatterportScene:
        if scene_id not in self._scene_cache:
            try:
                path = self._scene_dirs[scene_id]
            except KeyError as error:
                raise KeyError(f"Unknown Matterport scene: {scene_id}") from error
            self._scene_cache[scene_id] = MatterportScene(scene_id, path)
        return self._scene_cache[scene_id]

    def iter_scenes(self):
        return iter(self)

    def __len__(self) -> int:
        return len(self._scene_dirs)

    def __iter__(self):
        for scene_id in self.scene_ids:
            yield self.get_scene(scene_id)

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.get_scene(index)
        return self.get_scene(self.scene_ids[index])

    def __repr__(self) -> str:
        return f"Matterport3DDataset(root={str(self.root)!r}, scenes={len(self)})"
