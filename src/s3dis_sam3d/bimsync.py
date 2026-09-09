from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from .config import CONFIG, bimsync_calibration_dir
from .pointcloud import visualize_point_clouds, voxel_downsample
from .registration import (
    downsample_points,
    register_upright,
    sample_labeled_mesh,
    stable_seed,
)
from .rendering import FrameRender, MeshRaycaster, render_geometries
from .s23dis import S23DIS_DEPTH_SCALE, S23DIS_INVALID_DEPTH
from .utils_ifc import (
    STRUCTURAL_CLASSES,
    STRUCTURAL_IFC_TYPES,
    load_ifc_mesh,
    load_labeled_ifc_geometry,
    raycaster_mesh_options,
)

STRUCTURAL_S3DIS_CLASSES = STRUCTURAL_CLASSES


def _canonical_target_labels(room, cloud):
    if cloud.semantic_labels is None:
        return None
    names = getattr(getattr(room, "dataset", None), "semantic_classes", None)
    if names is None:
        names = cloud.metadata.get("label_names")
    if names is None:
        return None
    names = [str(name).casefold() for name in names]
    labels = np.full(len(cloud.xyz), -1, dtype=np.int32)
    for output_id, name in enumerate(STRUCTURAL_S3DIS_CLASSES):
        if name in names:
            labels[cloud.semantic_labels == names.index(name)] = output_id
    return labels


@dataclass(frozen=True)
class BIMSyncScene:
    dataset: BIMSyncDataset = field(repr=False, compare=False)
    area: str
    name: str
    path: Path
    _raycaster_cache: dict = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def key(self):
        return f"{self.area}/{self.name}"

    @property
    def calibration(self):
        matrix = self.dataset._calibrations.get(self.name.casefold())
        return None if matrix is None else matrix.copy()

    @property
    def is_calibrated(self):
        return self.name.casefold() in self.dataset._calibrations

    def set_calibration(self, transform):
        self.dataset._calibrations[self.name.casefold()] = np.asarray(
            transform,
            dtype=np.float64,
        ).copy()
        self._raycaster_cache.clear()
        return self

    def _raw_mesh(self, include_types=None):
        return load_ifc_mesh(self.path, include_types)

    def _registration_geometry(self, include_types):
        return load_labeled_ifc_geometry(self.path, include_types)

    def mesh(
        self,
        include_types=None,
        *,
        calibrated=True,
        transform=None,
    ):
        if include_types is None:
            include_types = STRUCTURAL_IFC_TYPES
        mesh = self._raw_mesh(include_types)
        transform = (
            transform if transform is not None else (self.calibration if calibrated else None)
        )
        if transform is not None:
            mesh.transform(np.asarray(transform))
        return mesh

    def export(
        self,
        output_path,
        *,
        include_types=None,
        calibrated=True,
    ):
        mesh = self.mesh(
            include_types,
            calibrated=calibrated,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_triangle_mesh(str(output_path), mesh):
            raise RuntimeError(f"failed to export IFC mesh: {output_path}")
        return output_path

    def register(
        self,
        s3dis_room,
        *,
        s3dis_classes=STRUCTURAL_S3DIS_CLASSES,
        ifc_types=STRUCTURAL_IFC_TYPES,
        ifc_samples=8_000,
        voxel_size=None,
        thresholds=(1.5, 0.75, 0.35, 0.18),
        yaw_candidates=None,
        yaw_starts=36,
        refine_candidates=4,
        coarse_points=1_500,
        max_iterations=25,
        seed=20260810,
        with_scaling=False,
    ):
        """Register IFC to S3DIS with unit scale, Z-up and yaw-only rotation.

        The upright prior is intentional: both IFC and Stanford geometry use
        metres and a vertical Z axis.  Symmetric trimmed correspondences avoid
        letting the denser point set dominate, while class-aware reranking
        resolves otherwise indistinguishable 180-degree room layouts.
        """
        if with_scaling:
            raise ValueError("upright IFC registration fixes scale to one")
        if voxel_size is not None and voxel_size <= 0:
            raise ValueError("voxel_size must be positive or None")

        s3_cloud = s3dis_room.point_cloud(include_classes=s3dis_classes)
        if voxel_size is not None:
            s3_cloud = voxel_downsample(s3_cloud, voxel_size)
        target_labels = _canonical_target_labels(s3dis_room, s3_cloud)
        target = np.asarray(s3_cloud.xyz, dtype=np.float64)
        valid = np.isfinite(target).all(axis=1)
        target = target[valid]
        target_labels = None if target_labels is None else target_labels[valid]
        target, target_labels = downsample_points(
            target,
            target_labels,
            ifc_samples,
            stable_seed(seed, self.name, "s3dis"),
        )

        vertices, triangles, face_labels = self._registration_geometry(ifc_types)
        source, source_labels = sample_labeled_mesh(
            vertices,
            triangles,
            face_labels,
            ifc_samples,
            stable_seed(seed, self.name, "ifc"),
        )
        explicit_yaws = (
            None
            if yaw_candidates is None
            else [np.deg2rad(float(value)) for value in yaw_candidates]
        )
        result = register_upright(
            source,
            target,
            source_labels=source_labels if target_labels is not None else None,
            target_labels=target_labels,
            label_names=STRUCTURAL_S3DIS_CLASSES,
            yaw_starts=yaw_starts,
            yaw_angles=explicit_yaws,
            refine_candidates=refine_candidates,
            coarse_points=min(coarse_points, len(source), len(target)),
            distances=tuple(float(value) for value in thresholds),
            iterations=max_iterations,
        )
        ifc_to_s3dis = result.transform
        s3dis_to_ifc = np.linalg.inv(ifc_to_s3dis)
        yaw_deg = float(np.rad2deg(np.arctan2(ifc_to_s3dis[1, 0], ifc_to_s3dis[0, 0])))
        return BIMSyncRegistration(
            area=self.area,
            scene=self.name,
            ifc_path=self.path,
            s3dis_room=s3dis_room.key,
            ifc_to_s3dis=ifc_to_s3dis,
            s3dis_to_ifc=s3dis_to_ifc,
            fitness=result.metrics["fitness"],
            rmse=result.metrics["rmse"],
            yaw_deg=yaw_deg,
            stages=result.candidates,
            accepted=result.accepted,
            quality_checks=result.quality_checks,
            semantic_audit=result.semantic_audit,
        )

    def save_registration(self, registration, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{self.name}_ifc_to_s3dis_transform.json"
        npy_path = output_dir / f"{self.name}_ifc_to_s3dis_transform.npy"
        json_path.write_text(json.dumps(registration.as_dict(), indent=2), "utf-8")
        np.save(npy_path, registration.ifc_to_s3dis)
        np.save(
            output_dir / f"{self.name}_s3dis_to_ifc_transform.npy",
            registration.s3dis_to_ifc,
        )
        self.set_calibration(registration.ifc_to_s3dis)
        return json_path, npy_path

    def visualize_registration(
        self,
        registration,
        s3dis_room,
        output_path=None,
        *,
        show=False,
        width=1600,
        height=1000,
        ifc_samples=100_000,
        seed=42,
    ):
        s3dis_cloud = s3dis_room.point_cloud()
        mesh = self.mesh(calibrated=False, transform=registration.ifc_to_s3dis)
        o3d.utility.random.seed(seed)
        ifc_cloud = mesh.sample_points_uniformly(ifc_samples)
        ifc_cloud.paint_uniform_color([1.0, 0.15, 0.05])
        return visualize_point_clouds(
            [s3dis_cloud, ifc_cloud],
            window_name=f"BIMSync IFC → S3DIS | {self.name}",
            width=width,
            height=height,
            background_color=(0.05, 0.05, 0.05),
            save_path=output_path,
            show=show,
        )

    def render_depth(self, frame, *, include_types=None, size=None):
        """Raycast metric depth for a regular S3DIS frame using a cached scene."""
        if include_types is None:
            include_types = STRUCTURAL_IFC_TYPES
        if not self.is_calibrated:
            raise ValueError(f"no calibration found for IFC scene: {self.name}")
        if frame.projection_type != "regular":
            raise ValueError("IFC pinhole rendering requires a regular 2D-3D-S frame")

        include_types, cache_key = raycaster_mesh_options(include_types)
        raycaster = self._raycaster_cache.get(cache_key)
        if raycaster is None:
            raycaster = MeshRaycaster(self.mesh(include_types))
            self._raycaster_cache[cache_key] = raycaster

        if size is not None:
            height, width = size
            intrinsics = frame.intrinsics_for_size(size)
        else:
            height, width = frame.image_shape
            intrinsics = frame.intrinsics

        return raycaster.depth(
            intrinsics,
            frame.world_to_camera,
            width,
            height,
        )

    def render_frame(
        self,
        frame,
        *,
        mesh_color=(0.75, 0.75, 0.75),
        background_color=(0.05, 0.05, 0.05),
        render_depth=True,
        show=False,
    ):
        if not self.is_calibrated:
            raise ValueError(f"no calibration found for IFC scene: {self.name}")
        if frame.projection_type != "regular":
            raise ValueError("IFC pinhole rendering requires a regular 2D-3D-S frame")

        source_image = frame.rgb
        height, width = source_image.shape[:2]

        mesh = self.mesh()
        mesh.paint_uniform_color(mesh_color)
        rendered_image, rendered_depth = render_geometries(
            [mesh],
            window_name=f"BIMSync IFC | {frame.room} | frame {frame.frame_id}",
            width=width,
            height=height,
            background_color=background_color,
            intrinsics=frame.intrinsics,
            world_to_camera=frame.world_to_camera,
            render_depth=render_depth,
            show=show,
            mesh_show_back_face=True,
        )

        source_depth = frame.depth if frame.has_depth else None
        return BIMSyncFrameRender(
            scene=self.name,
            room=frame.room,
            frame_id=frame.frame_id,
            uuid=frame.uuid,
            rendered_image_path=None,
            rendered_depth_path=None,
            source_image_path=frame.rgb_path,
            source_depth_path=frame.depth_path,
            source_image=source_image,
            source_depth=source_depth,
            rendered_image=rendered_image,
            rendered_depth=rendered_depth,
        )


@dataclass(frozen=True)
class BIMSyncRegistration:
    area: str
    scene: str
    ifc_path: Path
    s3dis_room: str
    ifc_to_s3dis: np.ndarray
    s3dis_to_ifc: np.ndarray
    fitness: float
    rmse: float
    yaw_deg: float
    stages: list[dict]
    accepted: bool = True
    quality_checks: dict[str, bool] = field(default_factory=dict)
    semantic_audit: dict = field(default_factory=dict)

    @property
    def scale(self):
        return float(abs(np.linalg.det(self.ifc_to_s3dis[:3, :3])) ** (1 / 3))

    def as_dict(self):
        return {
            "source": "IFC",
            "target": "S3DIS",
            "transform_direction": "ifc_meters_to_s3dis_meters",
            "area": self.area,
            "scene": self.scene,
            "ifc_path": str(self.ifc_path),
            "s3dis_room": self.s3dis_room,
            "ifc_to_s3dis": self.ifc_to_s3dis.tolist(),
            "s3dis_to_ifc": self.s3dis_to_ifc.tolist(),
            "ifc_to_s3dis_scale": self.scale,
            "fitness": self.fitness,
            "rmse": self.rmse,
            "yaw_deg": self.yaw_deg,
            "accepted": self.accepted,
            "quality_checks": self.quality_checks,
            "semantic_reranking": self.semantic_audit,
            "candidates": self.stages,
        }


@dataclass(frozen=True)
class BIMSyncFrameRender(FrameRender):
    depth_scale = S23DIS_DEPTH_SCALE
    depth_invalid_value = S23DIS_INVALID_DEPTH

    scene: str
    room: str
    frame_id: int
    uuid: str


class BIMSyncDataset:
    """Discover BIMSync scenes and coordinate Area-level batch operations."""

    def __init__(self, root=None, area="Area_1", calibration_dir=None):
        using_default_root = root is None

        if root is None:
            root = CONFIG.require("bimsync_root")

        if using_default_root and calibration_dir is None:
            calibration_dir = bimsync_calibration_dir(area)

        self.root = Path(root).expanduser().resolve()
        self.area = area
        self.calibration_dir = None
        self._calibrations = {}

        self.ifc_dir = self._resolve_ifc_dir()

        self.scenes = [
            BIMSyncScene(self, area, path.stem, path) for path in sorted(self.ifc_dir.glob("*.ifc"))
        ]

        if not self.scenes:
            raise ValueError(f"no IFC files found under {self.ifc_dir}")

        if calibration_dir is not None:
            self.load_calibrations(calibration_dir)

    def _resolve_ifc_dir(self):
        if self.root.is_file():
            return self.root.parent
        area_names = tuple(dict.fromkeys((str(self.area), str(self.area).casefold())))
        roots = (
            self.root,
            self.root / "ifc",
            self.root / "BIM_model" / "ifc",
        )
        candidates = tuple(root / area_name for root in roots for area_name in area_names) + roots
        return next((path for path in candidates if list(path.glob("*.ifc"))), self.root)

    def scene(self, scene):
        if isinstance(scene, int):
            return self.scenes[scene]
        name = Path(str(scene).replace("\\", "/")).stem.casefold()
        return next(item for item in self.scenes if item.name.casefold() == name)

    def matching_scenes(self, s3dis):
        room_names = {
            room.name.casefold()
            for room in s3dis.rooms
            if room.area.casefold() == self.area.casefold()
        }
        return [scene for scene in self.scenes if scene.name.casefold() in room_names]

    def load_calibrations(self, calibration_dir):
        self.calibration_dir = Path(calibration_dir).expanduser().resolve()

        if not self.calibration_dir.is_dir():
            raise FileNotFoundError(
                f"BIMSync calibration directory not found: {self.calibration_dir}"
            )

        calibration_paths = tuple(
            sorted(self.calibration_dir.rglob("*_ifc_to_s3dis_transform.npy"))
        )

        if not calibration_paths:
            raise ValueError(
                f"no *_ifc_to_s3dis_transform.npy files found under {self.calibration_dir}"
            )

        self._calibrations = {
            path.name.removesuffix("_ifc_to_s3dis_transform.npy").casefold(): np.load(path)
            for path in calibration_paths
        }
        for scene in self.scenes:
            scene._raycaster_cache.clear()
        return self._calibrations

    def export_meshes(self, output_dir, extension=".ply", **mesh_options):
        output_dir = Path(output_dir)
        extension = extension if extension.startswith(".") else f".{extension}"
        return [
            scene.export(
                output_dir / f"{scene.name}{extension}",
                **mesh_options,
            )
            for scene in self.scenes
        ]

    def calibrate_scenes(
        self,
        s3dis,
        output_dir,
        scenes=None,
        *,
        visualize=False,
        visualization_options=None,
        progress=True,
        **registration_options,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dir = output_dir.resolve()
        scenes = (
            self.matching_scenes(s3dis)
            if scenes is None
            else [self.scene(scene) for scene in scenes]
        )
        iterator = tqdm(scenes, desc=f"{self.area} IFC → S3DIS") if progress else scenes
        summary = {"area": self.area, "success": {}, "errors": {}}
        summary_path = output_dir / "calibration_summary.json"

        for scene in iterator:
            try:
                room = s3dis.room(f"{self.area}/{scene.name}")
                registration = scene.register(room, **registration_options)
                json_path, npy_path = scene.save_registration(
                    registration,
                    output_dir / scene.name,
                )
                result = {
                    "fitness": registration.fitness,
                    "rmse": registration.rmse,
                    "scale": registration.scale,
                    "accepted": registration.accepted,
                    "quality_checks": registration.quality_checks,
                    "json": str(json_path),
                    "npy": str(npy_path),
                }
                if visualize:
                    image_path = output_dir / scene.name / "registration.png"
                    scene.visualize_registration(
                        registration,
                        room,
                        image_path,
                        **(visualization_options or {}),
                    )
                    result["visualization"] = str(image_path)
                destination = "success" if registration.accepted else "errors"
                summary[destination][scene.name] = result
            except Exception as error:  # noqa: BLE001 - keep the Area batch running
                summary["errors"][scene.name] = f"{type(error).__name__}: {error}"

            temporary = summary_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(summary, indent=2), "utf-8")
            temporary.replace(summary_path)

        return summary
