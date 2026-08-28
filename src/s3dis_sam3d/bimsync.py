from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from .config import BIMSYNC_ROOT, bimsync_calibration_dir
from .pointcloud import to_open3d_point_cloud, visualize_point_clouds
from .rendering import FrameRender, render_geometries
from .s23dis import S23DIS_DEPTH_SCALE, S23DIS_INVALID_DEPTH
from .utils import (
    initial_registration_transform,
    preprocess_registration_cloud,
    run_icp,
)

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


def load_ifc_mesh(path, include_types=None):
    """Load IFC products into one Open3D mesh in IFC world coordinates."""

    import ifcopenshell
    import ifcopenshell.geom

    path = Path(path)
    model = ifcopenshell.open(str(path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    include_types = None if include_types is None else {
        str(ifc_type).casefold() for ifc_type in include_types
    }

    vertices = []
    triangles = []
    offset = 0
    for product in model.by_type("IfcProduct"):
        if include_types and product.is_a().casefold() not in include_types:
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
        raise RuntimeError(f"no IFC mesh geometry found: {path}")
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.vstack(vertices)),
        o3d.utility.Vector3iVector(np.vstack(triangles)),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


@dataclass(frozen=True)
class BIMSyncRegion:
    dataset: BIMSyncDataset = field(repr=False, compare=False)
    area: str
    name: str
    path: Path

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
        return self

    def _raw_mesh(self, include_types=None):
        return load_ifc_mesh(self.path, include_types)

    def mesh(
        self,
        include_types=None,
        *,
        calibrated=True,
        transform=None,
    ):
        mesh = self._raw_mesh(include_types)
        transform = transform if transform is not None else (
            self.calibration if calibrated else None
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
        ifc_samples=250_000,
        voxel_size=0.05,
        thresholds=(0.8, 0.4, 0.2, 0.1, 0.05),
        yaw_candidates=(0, 90, 180, 270),
        max_iterations=500,
        seed=42,
        with_scaling=False,
    ):
        s3_cloud = s3dis_room.point_cloud(include_classes=s3dis_classes)
        s3_o3d = to_open3d_point_cloud(s3_cloud)
        ifc_mesh = self.mesh(ifc_types, calibrated=False)
        o3d.utility.random.seed(seed)
        ifc_o3d = ifc_mesh.sample_points_uniformly(ifc_samples)

        source = preprocess_registration_cloud(s3_o3d, voxel_size)
        target = preprocess_registration_cloud(ifc_o3d, voxel_size)
        best = None
        for yaw in yaw_candidates:
            initial = initial_registration_transform(
                source,
                target,
                yaw,
                with_scaling,
            )
            transform, stages = run_icp(
                source,
                target,
                initial,
                thresholds,
                max_iterations,
                with_scaling,
            )
            final = stages[-1]
            score = final["fitness"], -final["rmse"]
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
        return BIMSyncRegistration(
            self.area,
            self.name,
            self.path,
            s3dis_room.key,
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
            raise ValueError(f"no calibration found for IFC region: {self.name}")
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
            region=self.name,
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
class BIMSyncFrameRender(FrameRender):
    depth_scale = S23DIS_DEPTH_SCALE
    depth_invalid_value = S23DIS_INVALID_DEPTH

    region: str
    room: str
    frame_id: int
    uuid: str


class BIMSyncDataset:
    """Discover BIMSync regions and coordinate Area-level batch operations."""

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
        self._calibrations = {}
        self.ifc_dir = self._resolve_ifc_dir()
        self.regions = [
            BIMSyncRegion(self, area, path.stem, path)
            for path in sorted(self.ifc_dir.glob("*.ifc"))
        ]
        if not self.regions:
            raise ValueError(f"no IFC files found under {self.ifc_dir}")
        if calibration_dir is not None:
            self.load_calibrations(calibration_dir)

    def _resolve_ifc_dir(self):
        if self.root.is_file():
            return self.root.parent
        candidates = self.root / self.area, self.root / self.area.casefold(), self.root
        return next((path for path in candidates if list(path.glob("*.ifc"))), self.root)

    def region(self, region):
        if isinstance(region, int):
            return self.regions[region]
        name = Path(str(region).replace("\\", "/")).stem.casefold()
        return next(item for item in self.regions if item.name.casefold() == name)

    def matching_regions(self, s3dis):
        room_names = {
            room.name.casefold()
            for room in s3dis.rooms
            if room.area.casefold() == self.area.casefold()
        }
        return [region for region in self.regions if region.name.casefold() in room_names]

    def load_calibrations(self, calibration_dir):
        self.calibration_dir = Path(calibration_dir).expanduser().resolve()
        self._calibrations = {
            path.name.removesuffix("_ifc_to_s3dis_transform.npy").casefold(): np.load(path)
            for path in sorted(
                self.calibration_dir.rglob("*_ifc_to_s3dis_transform.npy")
            )
        }
        return self._calibrations

    def export_meshes(self, output_dir, extension=".ply", **mesh_options):
        output_dir = Path(output_dir)
        extension = extension if extension.startswith(".") else f".{extension}"
        return [
            region.export(
                output_dir / f"{region.name}{extension}",
                **mesh_options,
            )
            for region in self.regions
        ]

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
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dir = output_dir.resolve()
        regions = (
            self.matching_regions(s3dis)
            if regions is None
            else [self.region(region) for region in regions]
        )
        iterator = (
            tqdm(regions, desc=f"{self.area} IFC → S3DIS") if progress else regions
        )
        summary = {"area": self.area, "success": {}, "errors": {}}
        summary_path = output_dir / "calibration_summary.json"

        for region in iterator:
            try:
                room = s3dis.room(f"{self.area}/{region.name}")
                registration = region.register(room, **registration_options)
                json_path, npy_path = region.save_registration(
                    registration,
                    output_dir / region.name,
                )
                result = {
                    "fitness": registration.fitness,
                    "rmse": registration.rmse,
                    "scale": registration.scale,
                    "json": str(json_path),
                    "npy": str(npy_path),
                }
                if visualize:
                    image_path = output_dir / region.name / "registration.png"
                    region.visualize_registration(
                        registration,
                        room,
                        image_path,
                        **(visualization_options or {}),
                    )
                    result["visualization"] = str(image_path)
                summary["success"][region.name] = result
            except Exception as error:  # noqa: BLE001 - keep the Area batch running
                summary["errors"][region.name] = f"{type(error).__name__}: {error}"

            temporary = summary_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(summary, indent=2), "utf-8")
            temporary.replace(summary_path)

        return summary
