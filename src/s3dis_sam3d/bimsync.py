from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

from .config import BIMSYNC_ROOT, bimsync_calibration_dir
from .pointcloud import to_open3d_point_cloud, visualize_point_clouds
from .s23dis import S23DIS_DEPTH_SCALE, S23DIS_INVALID_DEPTH

STRUCTURAL_S3DIS_CLASSES = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
)

STRUCTURAL_IFC_TYPES = {
    "IfcWall",
    "IfcWallStandardCase",
    "IfcSlab",
    "IfcCovering",
    "IfcColumn",
    "IfcBeam",
    "IfcDoor",
    "IfcWindow",
}


@dataclass(frozen=True)
class IFCRegion:
    area: str
    name: str
    path: Path

    @property
    def key(self):
        return f"{self.area}/{self.name}"


@dataclass(frozen=True)
class IFCRegistration:
    area: str
    region: str
    ifc_path: Path
    s3dis_room: str
    ifc_to_s3dis: np.ndarray
    s3dis_to_ifc: np.ndarray
    fitness: float
    rmse: float
    yaw_deg: float
    stages: list[dict]

    @property
    def scale(self):
        """Isotropic scale embedded in the IFC-to-S3DIS matrix."""
        return float(abs(np.linalg.det(self.ifc_to_s3dis[:3, :3])) ** (1 / 3))

    def as_dict(self):
        return {
            "source": "IFC",
            "target": "S3DIS",
            "transform_direction": "ifc_meters_to_s3dis_meters",
            "area": self.area,
            "region": self.region,
            "ifc_path": str(self.ifc_path),
            "s3dis_room": self.s3dis_room,
            "ifc_to_s3dis": self.ifc_to_s3dis.tolist(),
            "s3dis_to_ifc": self.s3dis_to_ifc.tolist(),
            "ifc_to_s3dis_scale": self.scale,
            "fitness": self.fitness,
            "rmse": self.rmse,
            "initial_yaw_deg": self.yaw_deg,
            "stages": self.stages,
        }


@dataclass(frozen=True)
class IFCFrameRender:
    region: str
    room: str
    frame_id: int
    uuid: str
    rendered_image_path: Path
    rendered_depth_path: Path | None
    source_image_path: Path
    source_depth_path: Path | None
    source_image: np.ndarray = field(repr=False)
    rendered_depth: np.ndarray | None = field(repr=False)
    source_depth: np.ndarray | None = field(repr=False)


class BIMSyncDataset:
    """Area/region IFC reader and S3DIS registration helper."""

    def __init__(self, root=None, area="Area_1", calibration_dir=None):
        using_default_root = root is None
        root = BIMSYNC_ROOT if root is None else root
        if using_default_root and calibration_dir is None:
            default_calibration_dir = bimsync_calibration_dir(area)
            if default_calibration_dir.is_dir():
                calibration_dir = default_calibration_dir
        self.root = Path(root).expanduser().resolve()
        self.area = area
        self.calibration_dir = None
        self.calibrations = {}
        self.ifc_dir = self._resolve_ifc_dir()
        self.regions = [
            IFCRegion(area, path.stem, path)
            for path in sorted(self.ifc_dir.glob("*.ifc"))
        ]
        if not self.regions:
            raise ValueError(f"no IFC files found under {self.ifc_dir}")
        if calibration_dir is not None:
            self.load_calibrations(calibration_dir)

    def _resolve_ifc_dir(self):
        if self.root.is_file():
            return self.root.parent
        candidates = (
            self.root / self.area,
            self.root / self.area.casefold(),
            self.root,
        )
        return next((path for path in candidates if list(path.glob("*.ifc"))), self.root)

    def list_regions(self):
        return [region.key for region in self.regions]

    def resolve_region(self, region):
        if isinstance(region, int):
            return self.regions[region]
        name = Path(str(region).replace("\\", "/")).stem.casefold()
        matches = [item for item in self.regions if item.name.casefold() == name]
        if not matches:
            raise ValueError(f"IFC region not found: {region}")
        return matches[0]

    def matching_regions(self, s3dis):
        rooms = {
            room.split("/", 1)[1]
            for room in s3dis.list_rooms()
            if room.casefold().startswith(f"{self.area}/".casefold())
        }
        return [region.name for region in self.regions if region.name in rooms]

    def load_calibrations(self, calibration_dir):
        """Load saved IFC-to-S3DIS matrices from a batch calibration directory."""
        self.calibration_dir = Path(calibration_dir).expanduser().resolve()
        self.calibrations = {
            path.name.removesuffix("_ifc_to_s3dis_transform.npy").casefold(): np.load(path)
            for path in sorted(
                self.calibration_dir.rglob("*_ifc_to_s3dis_transform.npy")
            )
        }
        return self.calibrations

    def set_calibration(self, region, transform):
        info = self.resolve_region(region)
        self.calibrations[info.name.casefold()] = np.asarray(
            transform, dtype=np.float64
        ).copy()

    def get_calibration(self, region):
        info = self.resolve_region(region)
        matrix = self.calibrations.get(info.name.casefold())
        return None if matrix is None else matrix.copy()

    def calibrated_regions(self):
        return [
            region.name
            for region in self.regions
            if region.name.casefold() in self.calibrations
        ]

    @staticmethod
    def _normalize_units(vertices, unit_scale, strategy):
        if strategy == "force":
            return vertices * unit_scale
        if strategy == "auto" and len(vertices):
            extent = np.ptp(vertices, axis=0)
            if float(extent.max()) > 50 and unit_scale < 1:
                return vertices * unit_scale
        return vertices

    def _load_raw_mesh(self, info, include_types=None, unit_strategy="auto"):
        import ifcopenshell
        import ifcopenshell.geom
        import ifcopenshell.util.unit

        model = ifcopenshell.open(str(info.path))
        unit_scale = float(ifcopenshell.util.unit.calculate_unit_scale(model))
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        include_types = None if include_types is None else set(include_types)

        vertices = []
        triangles = []
        offset = 0
        for product in model.by_type("IfcProduct"):
            if include_types and product.is_a() not in include_types:
                continue
            if product.Representation is None:
                continue
            try:
                shape = ifcopenshell.geom.create_shape(settings, product)
            except RuntimeError:
                continue
            product_vertices = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
            product_faces = np.asarray(shape.geometry.faces, dtype=np.int32).reshape(-1, 3)
            if not len(product_vertices) or not len(product_faces):
                continue
            vertices.append(product_vertices)
            triangles.append(product_faces + offset)
            offset += len(product_vertices)

        if not vertices:
            raise RuntimeError(f"no IFC mesh geometry found: {info.path}")
        vertices = self._normalize_units(np.vstack(vertices), unit_scale, unit_strategy)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(np.vstack(triangles)),
        )
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return mesh

    def load_mesh(
        self,
        region,
        include_types=None,
        unit_strategy="auto",
        *,
        apply_calibration=True,
    ):
        """Load IFC geometry as a mesh and apply its saved calibration by default."""
        info = self.resolve_region(region)
        mesh = self._load_raw_mesh(info, include_types, unit_strategy)
        calibration = self.get_calibration(info.name) if apply_calibration else None
        if calibration is not None:
            mesh.transform(calibration)
        return mesh

    def export_mesh(
        self,
        region,
        output_path,
        *,
        include_types=None,
        unit_strategy="auto",
        apply_calibration=True,
    ):
        mesh = self.load_mesh(
            region,
            include_types,
            unit_strategy,
            apply_calibration=apply_calibration,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_triangle_mesh(str(output_path), mesh):
            raise RuntimeError(f"failed to export IFC mesh: {output_path}")
        return output_path

    def export_meshes(self, output_dir, extension=".ply", **mesh_options):
        output_dir = Path(output_dir)
        extension = extension if extension.startswith(".") else f".{extension}"
        return [
            self.export_mesh(region.name, output_dir / f"{region.name}{extension}", **mesh_options)
            for region in self.regions
        ]

    @staticmethod
    def _preprocess(cloud, voxel_size):
        cloud = cloud.voxel_down_sample(voxel_size) if voxel_size else copy.deepcopy(cloud)
        radius = max(voxel_size * 3, 0.05) if voxel_size else 0.1
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60)
        )
        return cloud

    @staticmethod
    def _initial_s3dis_to_ifc(source, target, yaw_deg, with_scaling=False):
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        source_center = source_points.mean(axis=0)
        target_center = target_points.mean(axis=0)
        angle = math.radians(yaw_deg)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        scale = 1.0
        if with_scaling:
            source_extent = np.ptp(source_points @ rotation.T, axis=0)
            target_extent = np.ptp(target_points, axis=0)
            valid = (source_extent > 1e-6) & (target_extent > 1e-6)
            scale = float(np.median(target_extent[valid] / source_extent[valid]))

        transform = np.eye(4)
        transform[:3, :3] = scale * rotation
        transform[:3, 3] = target_center - transform[:3, :3] @ source_center
        if not with_scaling:
            transform[2, 3] = target_points[:, 2].min() - source_points[:, 2].min()
        return transform

    @staticmethod
    def _run_icp(
        source,
        target,
        initial,
        thresholds,
        max_iterations,
        with_scaling=False,
    ):
        transform = initial
        stages = []
        estimator = (
            o3d.pipelines.registration.TransformationEstimationPointToPoint(True)
            if with_scaling
            else o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        for threshold in thresholds:
            result = o3d.pipelines.registration.registration_icp(
                source,
                target,
                float(threshold),
                transform,
                estimator,
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=max_iterations
                ),
            )
            transform = result.transformation
            stages.append(
                {
                    "threshold": float(threshold),
                    "fitness": float(result.fitness),
                    "rmse": float(result.inlier_rmse),
                }
            )
        return transform, stages

    def estimate_ifc_to_s3dis(
        self,
        region,
        s3dis,
        *,
        s3dis_room=None,
        s3dis_classes=STRUCTURAL_S3DIS_CLASSES,
        ifc_types=STRUCTURAL_IFC_TYPES,
        ifc_samples=250_000,
        voxel_size=0.05,
        thresholds=(0.8, 0.4, 0.2, 0.1, 0.05),
        yaw_candidates=(0, 90, 180, 270),
        max_iterations=500,
        seed=42,
        with_scaling=False,
    ):
        """Estimate the rigid or similarity matrix mapping IFC into S3DIS."""
        info = self.resolve_region(region)
        room = s3dis_room or f"{self.area}/{info.name}"
        s3_cloud = s3dis.filter_room(room, include_classes=s3dis_classes)
        s3_o3d = to_open3d_point_cloud(s3_cloud)
        ifc_mesh = self.load_mesh(
            info.name,
            ifc_types,
            apply_calibration=False,
        )
        o3d.utility.random.seed(seed)
        ifc_o3d = ifc_mesh.sample_points_uniformly(ifc_samples)

        source = self._preprocess(s3_o3d, voxel_size)
        target = self._preprocess(ifc_o3d, voxel_size)
        best = None
        for yaw in yaw_candidates:
            initial = self._initial_s3dis_to_ifc(
                source,
                target,
                yaw,
                with_scaling,
            )
            transform, stages = self._run_icp(
                source,
                target,
                initial,
                thresholds,
                max_iterations,
                with_scaling,
            )
            final = stages[-1]
            score = (final["fitness"], -final["rmse"])
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "yaw": float(yaw),
                    "transform": transform,
                    "stages": stages,
                }

        s3dis_to_ifc = best["transform"]
        ifc_to_s3dis = np.linalg.inv(s3dis_to_ifc)
        final = best["stages"][-1]
        return IFCRegistration(
            self.area,
            info.name,
            info.path,
            room,
            ifc_to_s3dis,
            s3dis_to_ifc,
            final["fitness"],
            final["rmse"],
            best["yaw"],
            best["stages"],
        )

    def save_registration(self, registration, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = registration.region
        json_path = output_dir / f"{stem}_ifc_to_s3dis_transform.json"
        npy_path = output_dir / f"{stem}_ifc_to_s3dis_transform.npy"
        json_path.write_text(json.dumps(registration.as_dict(), indent=2), "utf-8")
        np.save(npy_path, registration.ifc_to_s3dis)
        np.save(output_dir / f"{stem}_s3dis_to_ifc_transform.npy", registration.s3dis_to_ifc)
        self.set_calibration(registration.region, registration.ifc_to_s3dis)
        return json_path, npy_path

    def transformed_mesh(self, region, ifc_to_s3dis, **mesh_options):
        mesh = self.load_mesh(region, apply_calibration=False, **mesh_options)
        mesh.transform(np.asarray(ifc_to_s3dis))
        return mesh

    def calibrate_regions(
        self,
        s3dis,
        output_dir,
        regions=None,
        *,
        visualize=False,
        visualization_options=None,
        progress=True,
        **registration_options,
    ):
        """Calibrate matching IFC regions, save matrices, and keep successful results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dir = output_dir.resolve()
        regions = self.matching_regions(s3dis) if regions is None else list(regions)
        iterator = tqdm(regions, desc=f"{self.area} IFC → S3DIS") if progress else regions
        summary = {"area": self.area, "success": {}, "errors": {}}
        summary_path = output_dir / "calibration_summary.json"

        for region in iterator:
            try:
                registration = self.estimate_ifc_to_s3dis(
                    region,
                    s3dis,
                    **registration_options,
                )
                json_path, npy_path = self.save_registration(
                    registration,
                    output_dir / region,
                )
                result = {
                    "fitness": registration.fitness,
                    "rmse": registration.rmse,
                    "scale": registration.scale,
                    "json": str(json_path),
                    "npy": str(npy_path),
                }
                if visualize:
                    image_path = output_dir / region / "registration.png"
                    self.visualize_registration(
                        registration,
                        s3dis.get_region_point_cloud(region, area=self.area),
                        image_path,
                        **(visualization_options or {}),
                    )
                    result["visualization"] = str(image_path)
                summary["success"][region] = result
            except Exception as error:  # noqa: BLE001 - keep the Area batch running
                summary["errors"][region] = f"{type(error).__name__}: {error}"

            temporary = summary_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(summary, indent=2), "utf-8")
            temporary.replace(summary_path)

        return summary

    def visualize_registration(
        self,
        registration,
        s3dis_cloud,
        output_path=None,
        *,
        show=False,
        width=1600,
        height=1000,
        ifc_samples=100_000,
        seed=42,
    ):
        """Overlay RGB S3DIS points with red points sampled from the aligned IFC mesh."""
        mesh = self.transformed_mesh(
            registration.region,
            registration.ifc_to_s3dis,
        )
        o3d.utility.random.seed(seed)
        ifc_cloud = mesh.sample_points_uniformly(ifc_samples)
        ifc_cloud.paint_uniform_color([1.0, 0.15, 0.05])
        return visualize_point_clouds(
            [s3dis_cloud, ifc_cloud],
            window_name=f"BIMSync IFC → S3DIS | {registration.region}",
            width=width,
            height=height,
            background_color=(0.05, 0.05, 0.05),
            save_path=output_path,
            show=show,
        )

    def render_regular_frame(
        self,
        region,
        s23dis,
        frame_id,
        output_path,
        *,
        room=None,
        uuid=None,
        mesh_color=(0.75, 0.75, 0.75),
        background_color=(0.05, 0.05, 0.05),
        render_depth=True,
        depth_output_path=None,
        show=False,
    ):
        """Render calibrated IFC RGB/depth and return the matching source frame."""
        if s23dis.image_type != "regular":
            raise ValueError("IFC pinhole rendering requires a regular 2D-3D-S dataset")
        info = self.resolve_region(region)
        if self.get_calibration(info.name) is None:
            raise ValueError(f"no calibration found for IFC region: {info.name}")

        room = room or info.name
        frame = s23dis.get_frame(room, frame_id, uuid)
        source_image = s23dis.get_image(room, frame_id, frame.uuid)
        height, width = source_image.shape[:2]

        output_path = Path(output_path)
        if render_depth:
            depth_output_path = Path(
                depth_output_path
                or output_path.with_name(f"{output_path.stem}_depth.png")
            )
        else:
            depth_output_path = None

        mesh = self.load_mesh(info.name)
        mesh.paint_uniform_color(mesh_color)
        visualize_point_clouds(
            [mesh],
            window_name=f"BIMSync IFC | {room} | frame {frame_id}",
            width=width,
            height=height,
            background_color=background_color,
            set_parameters=(
                s23dis.get_k(room, frame_id, frame.uuid),
                s23dis.world_to_camera(room, frame_id, frame.uuid),
            ),
            save_path=output_path,
            depth_path=depth_output_path,
            depth_scale=S23DIS_DEPTH_SCALE,
            show=show,
            mesh_show_back_face=True,
        )

        source_depth = (
            None
            if frame.depth_path is None
            else s23dis.get_depth(room, frame_id, frame.uuid)
        )
        rendered_depth = None
        if depth_output_path is not None:
            with Image.open(depth_output_path) as depth_image:
                raw_depth = np.asarray(depth_image, dtype=np.uint16).copy()
            raw_depth[raw_depth == 0] = S23DIS_INVALID_DEPTH
            Image.fromarray(raw_depth).save(depth_output_path)
            rendered_depth = raw_depth.astype(np.float32) / S23DIS_DEPTH_SCALE
            rendered_depth[raw_depth == S23DIS_INVALID_DEPTH] = 0
        return IFCFrameRender(
            region=info.name,
            room=room,
            frame_id=frame.frame_id,
            uuid=frame.uuid,
            rendered_image_path=output_path,
            rendered_depth_path=depth_output_path,
            source_image_path=frame.rgb_path,
            source_depth_path=frame.depth_path,
            source_image=source_image,
            rendered_depth=rendered_depth,
            source_depth=source_depth,
        )
