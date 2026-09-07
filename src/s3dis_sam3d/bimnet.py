"""BIMNet scene parser for IFC, component OBJ, rooms, and labeled point clouds."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path, PurePosixPath

import numpy as np
import open3d as o3d

from .bimsync import load_ifc_mesh
from .config import CONFIG
from .matterport import MATTERPORT_DEPTH_SCALE, MatterportFrame
from .models import PointCloud
from .pointcloud import transform_points, visualize_point_clouds, voxel_downsample
from .rendering import FrameRender, MeshRaycaster, render_geometries

BIMNET_LABELS = (
    "wall",
    "slab",
    "beam",
    "column",
    "door",
    "window",
    "stair",
    "railing",
    "lighting",
    "furniture",
    "sanitary",
    "equipment",
    "object",
    "other",
)

# BIMNet's component OBJ export uses Y-up coordinates, while IfcOpenShell
# returns the IFC model in its native Z-up coordinates:
#     (x_obj, y_obj, z_obj) = (x_ifc, z_ifc, -y_ifc)
BIMNET_IFC_TO_OBJ = np.asarray(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)

BIMNET_MATTERPORT_SCANS = {
    "1px": "1pXnuDYAj8r",
    "759": "759xd9YjKW5",
    "7y3": "7y3sRwLe3Va",
    "7y3_1": "7y3sRwLe3Va",
    "ac2": "ac26ZMwG7aT",
    "b6b": "B6ByNegPMKs",
    "d7n": "D7N2EKCX4Sj",
    "e9z": "e9zR4mvMWw7",
    "e9z_1": "e9zR4mvMWw7",
    "hxp": "HxpKQynjfin",
    "i5n": "i5noydFURQK",
    "i5n_1": "i5noydFURQK",
    "px4": "PX4nDJXEHrG",
    "px4_1": "PX4nDJXEHrG",
    "px4_2": "PX4nDJXEHrG",
    "q9v": "q9vSo1VnCiC",
    "s9h": "S9hNv5qa7GM",
    "skl": "sKLMLpTHeUy",
    "sn8": "SN83YJsR3w2",
    "st4": "sT4fr6TAbpF",
    "ur6": "ur6pFq6Qu1A",
    "vt2": "Vt2qJdWjCF2",
    "vt2_1": "Vt2qJdWjCF2",
    "vvo": "Vvot9Ly1tCj",
    "zsn": "zsNo4HB9uLZ",
}

WALL_FILLED_ALIASES = {"d7n": "d7n2"}

ELEMENT_PATTERN = re.compile(
    r"^(?P<ifc_type>IFC[A-Z]+)-IFC#(?P<ifc_id>\d+)-RVT#(?P<rvt_id>[^-]+)-"
    r"Curved(?P<curved>True|False)-(?P<element_key>.+)\.obj$",
    re.IGNORECASE,
)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _record_transform(record):
    transform = record.get("transform") or {}
    values = [transform.get(f"M{row}{column}") for row in range(1, 5) for column in range(1, 5)]
    if any(value is None for value in values):
        return np.eye(4, dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def _normalize_ifc_types(include_types):
    if include_types is None:
        return None
    if isinstance(include_types, str):
        include_types = (include_types,)
    return {str(value).casefold() for value in include_types}


def _raycaster_mesh_options(source, wall_filled, include_types):
    """Return reusable mesh arguments and their canonical cache key."""

    source = str(source).casefold()
    if include_types is None:
        normalized_types = None
        type_key = None
    else:
        if isinstance(include_types, str):
            include_types = (include_types,)
        normalized_types = tuple(include_types)
        type_key = frozenset(str(value).casefold() for value in normalized_types)
    return source, normalized_types, (source, bool(wall_filled), type_key)


@dataclass(frozen=True)
class BIMNetElement:
    """One IFC component exported as an individual BIMNet OBJ file."""

    scene: BIMNetScene = field(repr=False, compare=False)
    path: Path
    ifc_id: int
    rvt_id: str
    guid: str | None
    ifc_type: str
    name: str | None = None
    family_type: str | None = None
    curved: bool = False
    angle: float = 0.0
    instance_transform: np.ndarray = field(
        default_factory=lambda: np.eye(4),
        repr=False,
        compare=False,
    )
    metadata: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def key(self):
        return f"{self.scene.key}/IFC#{self.ifc_id}"

    @property
    def material_path(self):
        return self.path.with_suffix(".mtl")

    @property
    def has_material(self):
        return self.material_path.is_file()

    @cached_property
    def material_color(self):
        """Read the first diffuse ``Kd`` color from the matching BIMNet MTL."""

        if not self.has_material:
            return None
        with self.material_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                tokens = line.strip().split()
                if len(tokens) == 4 and tokens[0].casefold() == "kd":
                    try:
                        color = np.asarray(tokens[1:], dtype=np.float64)
                    except ValueError:
                        continue
                    if np.isfinite(color).all():
                        return np.clip(color, 0, 1)
        return None

    def mesh(self, transform=None):
        """Load this component OBJ, optionally applying an extra transform."""

        mesh = o3d.io.read_triangle_mesh(str(self.path), enable_post_processing=False)
        if mesh.is_empty():
            raise ValueError(f"empty BIMNet OBJ mesh: {self.path}")
        if not mesh.has_vertex_colors() and self.material_color is not None:
            # BIMNet OBJ files commonly reference a shortened MTL filename even
            # though the downloaded MTL uses the full OBJ stem.
            mesh.paint_uniform_color(self.material_color)
        if transform is not None:
            mesh.transform(np.asarray(transform, dtype=np.float64))
        return mesh

    def export(self, output_path, transform=None):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_triangle_mesh(str(output_path), self.mesh(transform)):
            raise RuntimeError(f"failed to export BIMNet element: {output_path}")
        return output_path


@dataclass(frozen=True)
class BIMNetRoom:
    """An IFCSPACE record and its optional component GUID associations."""

    scene: BIMNetScene = field(repr=False, compare=False)
    ifc_id: int
    guid: str
    name: str
    element_guids: tuple[str, ...] = ()
    bounding_guids: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def key(self):
        return f"{self.scene.key}/IFCSPACE#{self.ifc_id}"

    @property
    def elements(self):
        by_guid = {element.guid: element for element in self.scene.instances if element.guid}
        return tuple(by_guid[guid] for guid in self.element_guids if guid in by_guid)

    @property
    def bounding_elements(self):
        by_guid = {element.guid: element for element in self.scene.instances if element.guid}
        return tuple(by_guid[guid] for guid in self.bounding_guids if guid in by_guid)


@dataclass(frozen=True)
class BIMNetFrameRender(FrameRender):
    """A BIMNet mesh rendering paired with its source Matterport frame."""

    depth_scale = MATTERPORT_DEPTH_SCALE
    depth_invalid_value = 0

    bimnet_scene_id: str
    matterport_scene_id: str
    frame_id: str
    mesh_source: str


class BIMNetScene:
    """One single-floor BIMNet scene with all available asset representations."""

    def __init__(self, dataset: BIMNetDataset, scene_id: str, split: str):
        self.dataset = dataset
        self.scene_id = scene_id
        self.split = split
        self.root = dataset.root
        self._element_cache = {}
        self._raycaster_cache = {}

    @property
    def key(self):
        return f"{self.split}/{self.scene_id}"

    @property
    def floor_index(self):
        parts = self.scene_id.rsplit("_", 1)
        return int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else None

    @property
    def matterport_scan_id(self):
        try:
            return BIMNET_MATTERPORT_SCANS[self.scene_id.casefold()]
        except KeyError as error:
            raise KeyError(f"Matterport scan mapping is unavailable for {self.scene_id}") from error

    @property
    def ifc_path(self):
        return self.root / "ifc" / self.split / f"{self.scene_id}.ifc"

    @property
    def matrix_path(self):
        return self.root / "mat_pc2obj" / self.split / f"{self.scene_id}.txt"

    @property
    def obj_dir(self):
        return self.root / "obj" / self.split / self.scene_id

    @property
    def wall_filled_obj_dir(self):
        base = self.root / "obj_wall_filled" / self.split
        direct = base / self.scene_id
        if direct.is_dir():
            return direct
        return base / WALL_FILLED_ALIASES.get(self.scene_id.casefold(), self.scene_id)

    @property
    def has_wall_filled_mesh(self):
        return self.wall_filled_obj_dir.is_dir() and any(self.wall_filled_obj_dir.glob("*.obj"))

    @property
    def rvt_path(self):
        directory = self.root / "rvt" / self.split
        direct = directory / f"{self.scene_id}.rvt"
        if direct.is_file():
            return direct
        matches = sorted(directory.glob(f"{self.scene_id}-*.rvt")) if directory.is_dir() else []
        return matches[0] if len(matches) == 1 else direct

    @property
    def has_rvt(self):
        return self.rvt_path.is_file()

    @property
    def point_cloud_path(self):
        directory = self.root / "point_cloud" / self.split
        suffix = f"_{self.floor_index}" if self.floor_index is not None else ""
        candidates = (
            directory / f"{self.matterport_scan_id}{suffix}.txt",
            directory / f"{self.scene_id}.txt",
            directory / f"{self.matterport_scan_id}.txt",
        )
        return next((path for path in candidates if path.is_file()), candidates[0])

    @property
    def has_point_cloud(self):
        return self.point_cloud_path.is_file()

    @cached_property
    def point_cloud_to_obj(self):
        if not self.matrix_path.is_file():
            raise FileNotFoundError(f"BIMNet pc-to-OBJ matrix is unavailable: {self.matrix_path}")
        matrix = np.loadtxt(self.matrix_path, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid 4x4 pc-to-OBJ matrix: {self.matrix_path}")
        return matrix

    @cached_property
    def obj_to_point_cloud(self):
        return np.linalg.inv(self.point_cloud_to_obj)

    @property
    def ifc_to_obj(self):
        """Return BIMNet's fixed native-IFC to native-OBJ axis conversion."""

        return BIMNET_IFC_TO_OBJ.copy()

    @cached_property
    def ifc_to_point_cloud(self):
        """Transform native IFC coordinates into the original point-cloud frame."""

        return self.obj_to_point_cloud @ BIMNET_IFC_TO_OBJ

    def mesh_to_point_cloud_transform(self, source="obj"):
        """Return the source-specific transform into point-cloud coordinates."""

        source = str(source).casefold()
        if source == "obj":
            return self.obj_to_point_cloud.copy()
        if source == "ifc":
            return self.ifc_to_point_cloud.copy()
        raise ValueError("source must be 'obj' or 'ifc'")

    @property
    def instances(self):
        return self._load_elements(wall_filled=False)

    @cached_property
    def rooms(self):
        path = self.obj_dir / "ifcrooms.json"
        if not path.is_file():
            return ()
        return tuple(
            BIMNetRoom(
                scene=self,
                ifc_id=int(record["id"]),
                guid=str(record["guid"]),
                name=str(record.get("name") or ""),
                element_guids=tuple(record.get("instances") or ()),
                bounding_guids=tuple(record.get("boundings") or ()),
                metadata=dict(record),
            )
            for record in _read_json(path)
        )

    def _load_elements(self, wall_filled=False):
        if wall_filled in self._element_cache:
            return self._element_cache[wall_filled]
        directory = self.wall_filled_obj_dir if wall_filled else self.obj_dir
        if not directory.is_dir():
            self._element_cache[wall_filled] = ()
            return ()

        metadata_path = directory / "ifcinstances.json"
        if metadata_path.is_file():
            records = _read_json(metadata_path)
            elements = tuple(
                self._element_from_record(directory, record)
                for record in records
                if record.get("path") and (directory / record["path"]).is_file()
            )
        else:
            elements = tuple(
                self._element_from_path(path) for path in sorted(directory.glob("*.obj"))
            )
        self._element_cache[wall_filled] = elements
        return elements

    def _element_from_record(self, directory, record):
        path = directory / record["path"]
        match = ELEMENT_PATTERN.match(path.name)
        ifc_type = str(record.get("ifctype") or (match.group("ifc_type") if match else ""))
        ifc_id = int(record.get("id") or (match.group("ifc_id") if match else 0))
        rvt_id = str(record.get("rvtid") or (match.group("rvt_id") if match else ""))
        return BIMNetElement(
            scene=self,
            path=path,
            ifc_id=ifc_id,
            rvt_id=rvt_id,
            guid=record.get("guid"),
            ifc_type=ifc_type.upper(),
            name=record.get("name"),
            family_type=record.get("type"),
            curved=bool(record.get("curved", False)),
            angle=float(record.get("angle", 0.0)),
            instance_transform=_record_transform(record),
            metadata=dict(record),
        )

    def _element_from_path(self, path):
        match = ELEMENT_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"invalid BIMNet component filename: {path.name}")
        return BIMNetElement(
            scene=self,
            path=path,
            ifc_id=int(match.group("ifc_id")),
            rvt_id=match.group("rvt_id"),
            guid=None,
            ifc_type=match.group("ifc_type").upper(),
            curved=match.group("curved").casefold() == "true",
        )

    def elements(self, include_types=None, *, wall_filled=False):
        elements = self._load_elements(wall_filled)
        include_types = _normalize_ifc_types(include_types)
        if include_types is not None:
            elements = tuple(
                element for element in elements if element.ifc_type.casefold() in include_types
            )
        return elements

    def element(self, identifier, *, wall_filled=False):
        value = str(identifier).casefold()
        for element in self._load_elements(wall_filled):
            candidates = {
                str(element.ifc_id).casefold(),
                f"ifc#{element.ifc_id}".casefold(),
                element.rvt_id.casefold(),
                element.path.stem.casefold(),
            }
            if element.guid:
                candidates.add(element.guid.casefold())
            if value in candidates:
                return element
        raise KeyError(identifier)

    def room(self, identifier):
        value = str(identifier).casefold()
        for room in self.rooms:
            if value in {str(room.ifc_id).casefold(), room.guid.casefold(), room.key.casefold()}:
                return room
        raise KeyError(identifier)

    def _validate_matterport_frame(self, frame):
        if not isinstance(frame, MatterportFrame):
            raise TypeError("frame must be a MatterportFrame")
        if frame.scene_id.casefold() != self.matterport_scan_id.casefold():
            raise ValueError(
                f"Matterport frame {frame.frame_id!r} belongs to {frame.scene_id!r}, "
                f"not BIMNet scene {self.scene_id!r} ({self.matterport_scan_id!r})"
            )

    def show(self, frame: MatterportFrame = None):
        """Interactively show the registered mesh, optionally from a frame view.

        The mesh is displayed in BIMNet's original point-cloud coordinates. If
        ``frame`` is supplied, its Matterport intrinsics and world-to-camera
        transform initialize the Open3D view; the window remains interactive.
        """

        if frame is not None:
            self._validate_matterport_frame(frame)

        mesh = self.mesh(coordinates="point_cloud")
        options = {
            "window_name": f"BIMNet | {self.key}",
            "mesh_show_back_face": True,
        }
        if frame is not None:
            height, width = frame.image_shape
            options.update(
                window_name=f"BIMNet | {self.key} | {frame.frame_id}",
                width=width,
                height=height,
                set_parameters=(frame.intrinsics, frame.world_to_camera),
            )
        return visualize_point_clouds([mesh], **options)

    def point_cloud(
        self,
        *,
        aligned=False,
        stride=1,
        include_labels=None,
        voxel_size=None,
    ):
        """Load the labeled BIMNet cloud in point-cloud or aligned OBJ coordinates."""

        if not self.has_point_cloud:
            raise FileNotFoundError(f"BIMNet point cloud is unavailable: {self.point_cloud_path}")
        if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
            raise ValueError("stride must be a positive integer")
        values = np.loadtxt(self.point_cloud_path, dtype=np.float32, ndmin=2)
        if values.shape[1] < 7:
            raise ValueError(f"expected x y z r g b label rows in {self.point_cloud_path}")
        values = values[::stride, :7]
        labels = values[:, 6].astype(np.int32)
        if include_labels is not None:
            include_labels = np.asarray(list(include_labels), dtype=np.int32)
            keep = np.isin(labels, include_labels)
            values, labels = values[keep], labels[keep]

        xyz = values[:, :3]
        if aligned:
            xyz = transform_points(xyz, self.point_cloud_to_obj)
        rgb = values[:, 3:6]
        if len(rgb) and np.nanmax(rgb) > 1:
            rgb = rgb / 255.0
        cloud = PointCloud(
            xyz=xyz,
            rgb=rgb,
            semantic_labels=labels,
            metadata={
                "scene_id": self.scene_id,
                "split": self.split,
                "matterport_scan_id": self.matterport_scan_id,
                "coordinate_frame": "obj" if aligned else "point_cloud",
                "label_names": BIMNET_LABELS,
            },
        )
        return voxel_downsample(cloud, voxel_size)

    def mesh(
        self,
        source="obj",
        *,
        wall_filled=False,
        include_types=None,
        coordinates="point_cloud",
        progress=False,
    ):
        """Load an IFC mesh or merge per-component OBJ meshes.

        Args:
            source: Select the component ``"obj"`` representation or the IFC
                representation.
            wall_filled: Use BIMNet's wall-filled OBJ variant. Only valid for
                ``source="obj"``.
            include_types: Optional IFC entity types to retain.
            coordinates: ``"original"`` preserves the selected source's own
                native coordinates. ``"point_cloud"`` registers either source
                to the original point-cloud coordinates.
            progress: Show component-level OBJ loading progress.
        """

        source = str(source).casefold()
        if source not in {"obj", "ifc"}:
            raise ValueError("source must be 'obj' or 'ifc'")
        coordinates = str(coordinates).casefold()
        if coordinates not in {"original", "point_cloud"}:
            raise ValueError("coordinates must be 'original' or 'point_cloud'")

        if source == "ifc":
            if wall_filled:
                raise ValueError("wall_filled is only available for OBJ meshes")
            mesh = load_ifc_mesh(self.ifc_path, include_types)
        else:
            elements = self.elements(include_types, wall_filled=wall_filled)
            if not elements:
                variant = "wall-filled OBJ" if wall_filled else "OBJ"
                raise FileNotFoundError(f"no {variant} components found for {self.key}")
            iterator = elements
            if progress:
                from tqdm.auto import tqdm

                iterator = tqdm(elements, desc=f"Loading BIMNet {self.key}")

            mesh = o3d.geometry.TriangleMesh()
            for element in iterator:
                mesh += element.mesh()
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()

        if coordinates == "point_cloud":
            mesh.transform(self.mesh_to_point_cloud_transform(source))
        return mesh

    def export_mesh(self, output_path, **mesh_options):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_triangle_mesh(str(output_path), self.mesh(**mesh_options)):
            raise RuntimeError(f"failed to export BIMNet mesh: {output_path}")
        return output_path

    def render_depth(
        self,
        frame: MatterportFrame,
        *,
        source="obj",
        wall_filled=False,
        include_types=None,
    ):
        """Raycast metric depth for a Matterport frame using a cached scene."""

        self._validate_matterport_frame(frame)
        source, include_types, cache_key = _raycaster_mesh_options(
            source,
            wall_filled,
            include_types,
        )
        raycaster = self._raycaster_cache.get(cache_key)
        if raycaster is None:
            mesh = self.mesh(
                source=source,
                wall_filled=wall_filled,
                include_types=include_types,
                coordinates="point_cloud",
            )
            raycaster = MeshRaycaster(mesh)
            self._raycaster_cache[cache_key] = raycaster

        height, width = frame.image_shape
        return raycaster.depth(
            frame.intrinsics,
            frame.world_to_camera,
            width,
            height,
        )

    def render_frame(
        self,
        frame: MatterportFrame,
        *,
        source="obj",
        wall_filled=False,
        include_types=None,
        mesh_color=(0.75, 0.75, 0.75),
        background_color=(0.05, 0.05, 0.05),
        render_depth=True,
        show=False,
    ):
        """Render a registered BIMNet mesh from a corresponding Matterport frame.

        The mesh is always transformed into BIMNet's original point-cloud
        coordinates, which are the Matterport world coordinates used by
        ``frame.camera_to_world``. The source RGB and depth remain owned by the
        supplied :class:`MatterportFrame`.
        """

        self._validate_matterport_frame(frame)

        source = str(source).casefold()
        source_image = frame.rgb
        source_depth = frame.depth
        height, width = source_image.shape[:2]

        mesh = self.mesh(
            source=source,
            wall_filled=wall_filled,
            include_types=include_types,
            coordinates="point_cloud",
        )
        if mesh_color is not None:
            mesh.paint_uniform_color(mesh_color)
        rendered_image, rendered_depth = render_geometries(
            [mesh],
            window_name=f"BIMNet {source.upper()} | {self.key} | {frame.frame_id}",
            width=width,
            height=height,
            background_color=background_color,
            intrinsics=frame.intrinsics,
            world_to_camera=frame.world_to_camera,
            render_depth=render_depth,
            show=show,
            mesh_show_back_face=True,
        )

        return BIMNetFrameRender(
            bimnet_scene_id=self.scene_id,
            matterport_scene_id=frame.scene_id,
            frame_id=frame.frame_id,
            mesh_source=source,
            rendered_image_path=None,
            rendered_depth_path=None,
            source_image_path=frame.rgb_path,
            source_depth_path=frame.depth_path,
            source_image=source_image,
            source_depth=source_depth,
            rendered_image=rendered_image,
            rendered_depth=rendered_depth,
        )

    def render(self, frame: MatterportFrame, **render_options):
        """Compatibility spelling for :meth:`render_frame`."""

        return self.render_frame(frame, **render_options)

    def visualize(
        self,
        *,
        show_point_cloud=True,
        show_mesh=True,
        aligned=True,
        mesh_options=None,
        point_cloud_options=None,
        **visualization_options,
    ):
        mesh_options = dict(mesh_options or {})
        point_cloud_options = dict(point_cloud_options or {})
        if aligned:
            # Use the original PC frame as the common visualization frame for
            # both OBJ and IFC sources.
            point_cloud_options.setdefault("aligned", False)
            if "coordinates" not in mesh_options:
                mesh_options["coordinates"] = "point_cloud"
        geometries = []
        if show_point_cloud:
            geometries.append(self.point_cloud(**point_cloud_options))
        if show_mesh:
            geometries.append(self.mesh(**mesh_options))
        if not geometries:
            raise ValueError("at least one of show_point_cloud/show_mesh must be enabled")
        return visualize_point_clouds(
            geometries,
            window_name=f"BIMNet | {self.key}",
            **visualization_options,
        )

    def matterport_scene(self, matterport_dataset):
        """Resolve the complete source house from a ``Matterport3DDataset``."""

        return matterport_dataset[self.matterport_scan_id]

    @property
    def availability(self):
        return {
            "ifc": self.ifc_path.is_file(),
            "matrix": self.matrix_path.is_file(),
            "obj": self.obj_dir.is_dir() and any(self.obj_dir.glob("*.obj")),
            "obj_wall_filled": self.has_wall_filled_mesh,
            "point_cloud": self.has_point_cloud,
            "rvt": self.has_rvt,
            "rooms": (self.obj_dir / "ifcrooms.json").is_file(),
        }

    def __len__(self):
        return len(self.instances)

    def __iter__(self):
        return iter(self.instances)

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.element(index)
        return self.instances[index]

    def __repr__(self):
        return f"BIMNetScene(key={self.key!r}, elements={len(self.instances)})"


class BIMNetScanScene:
    """A Matterport scan assembled from all matching BIMNet floor scenes."""

    def __init__(self, dataset: BIMNetDataset, scenes):
        scenes = tuple(scenes)
        if len(scenes) < 2:
            raise ValueError("a combined BIMNet scan requires at least two scenes")
        scan_ids = {scene.matterport_scan_id.casefold() for scene in scenes}
        if len(scan_ids) != 1:
            raise ValueError("all combined BIMNet scenes must map to the same Matterport scan")
        self.dataset = dataset
        self.root = dataset.root
        self.scenes = scenes
        self.matterport_scan_id = scenes[0].matterport_scan_id
        self.scene_id = "+".join(scene.scene_id for scene in scenes)
        self.split = "combined"
        self._raycaster_cache = {}

    @property
    def key(self):
        return f"scan/{self.matterport_scan_id}"

    @property
    def scene_ids(self):
        return tuple(scene.scene_id for scene in self.scenes)

    @property
    def instances(self):
        return tuple(element for scene in self.scenes for element in scene.instances)

    @property
    def rooms(self):
        return tuple(room for scene in self.scenes for room in scene.rooms)

    def elements(self, include_types=None, *, wall_filled=False):
        return tuple(
            element
            for scene in self.scenes
            for element in scene.elements(include_types, wall_filled=wall_filled)
        )

    def mesh(
        self,
        source="obj",
        *,
        wall_filled=False,
        include_types=None,
        coordinates="point_cloud",
        progress=False,
    ):
        """Merge all floor meshes in their shared Matterport coordinate frame."""

        coordinates = str(coordinates).casefold()
        if coordinates != "point_cloud":
            raise ValueError(
                "combined BIMNet scenes can only be merged in 'point_cloud' coordinates"
            )

        scenes = self.scenes
        if progress:
            from tqdm.auto import tqdm

            scenes = tqdm(scenes, desc=f"Loading BIMNet {self.key}")

        mesh = o3d.geometry.TriangleMesh()
        for scene in scenes:
            mesh += scene.mesh(
                source=source,
                wall_filled=wall_filled,
                include_types=include_types,
                coordinates="point_cloud",
                progress=False,
            )
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return mesh

    def export_mesh(self, output_path, **mesh_options):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_triangle_mesh(str(output_path), self.mesh(**mesh_options)):
            raise RuntimeError(f"failed to export combined BIMNet mesh: {output_path}")
        return output_path

    def render_depth(
        self,
        frame: MatterportFrame,
        *,
        source="obj",
        wall_filled=False,
        include_types=None,
    ):
        """Raycast metric depth from the combined mesh using a cached scene."""

        self._validate_matterport_frame(frame)
        source, include_types, cache_key = _raycaster_mesh_options(
            source,
            wall_filled,
            include_types,
        )
        raycaster = self._raycaster_cache.get(cache_key)
        if raycaster is None:
            mesh = self.mesh(
                source=source,
                wall_filled=wall_filled,
                include_types=include_types,
                coordinates="point_cloud",
            )
            raycaster = MeshRaycaster(mesh)
            self._raycaster_cache[cache_key] = raycaster

        height, width = frame.image_shape
        return raycaster.depth(
            frame.intrinsics,
            frame.world_to_camera,
            width,
            height,
        )

    def _validate_matterport_frame(self, frame):
        if not isinstance(frame, MatterportFrame):
            raise TypeError("frame must be a MatterportFrame")
        if frame.scene_id.casefold() != self.matterport_scan_id.casefold():
            raise ValueError(
                f"Matterport frame {frame.frame_id!r} belongs to {frame.scene_id!r}, "
                f"not BIMNet scan {self.matterport_scan_id!r}"
            )

    def show(self, frame: MatterportFrame = None):
        if frame is not None:
            self._validate_matterport_frame(frame)
        mesh = self.mesh()
        options = {
            "window_name": f"BIMNet | {self.key}",
            "mesh_show_back_face": True,
        }
        if frame is not None:
            height, width = frame.image_shape
            options.update(
                window_name=f"BIMNet | {self.key} | {frame.frame_id}",
                width=width,
                height=height,
                set_parameters=(frame.intrinsics, frame.world_to_camera),
            )
        return visualize_point_clouds([mesh], **options)

    def render_frame(
        self,
        frame: MatterportFrame,
        *,
        source="obj",
        wall_filled=False,
        include_types=None,
        mesh_color=(0.75, 0.75, 0.75),
        background_color=(0.05, 0.05, 0.05),
        render_depth=True,
        show=False,
    ):
        self._validate_matterport_frame(frame)
        source = str(source).casefold()
        source_image = frame.rgb
        source_depth = frame.depth
        height, width = source_image.shape[:2]
        mesh = self.mesh(
            source=source,
            wall_filled=wall_filled,
            include_types=include_types,
        )
        if mesh_color is not None:
            mesh.paint_uniform_color(mesh_color)
        rendered_image, rendered_depth = render_geometries(
            [mesh],
            window_name=f"BIMNet {source.upper()} | {self.key} | {frame.frame_id}",
            width=width,
            height=height,
            background_color=background_color,
            intrinsics=frame.intrinsics,
            world_to_camera=frame.world_to_camera,
            render_depth=render_depth,
            show=show,
            mesh_show_back_face=True,
        )
        return BIMNetFrameRender(
            bimnet_scene_id=self.scene_id,
            matterport_scene_id=frame.scene_id,
            frame_id=frame.frame_id,
            mesh_source=source,
            rendered_image_path=None,
            rendered_depth_path=None,
            source_image_path=frame.rgb_path,
            source_depth_path=frame.depth_path,
            source_image=source_image,
            source_depth=source_depth,
            rendered_image=rendered_image,
            rendered_depth=rendered_depth,
        )

    def render(self, frame: MatterportFrame, **render_options):
        return self.render_frame(frame, **render_options)

    def matterport_scene(self, matterport_dataset):
        return matterport_dataset[self.matterport_scan_id]

    def __len__(self):
        return len(self.instances)

    def __iter__(self):
        return iter(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

    def __repr__(self):
        return f"BIMNetScanScene(key={self.key!r}, scenes={self.scene_ids!r})"


class BIMNetDataset:
    """Discover BIMNet train/test scenes and their parallel asset trees."""

    def __init__(self, root=None, split=None):
        if root is None:
            root = CONFIG.require("bimnet_root")
        self.root = Path(root).expanduser().resolve()

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"BIMNet root directory not found: {self.root}"
            )
        if split not in (None, "train", "test"):
            raise ValueError("split must be None, 'train', or 'test'")
        self.selected_split = split
        self._scene_index = self._discover_scenes()
        self._scene_aliases = self._build_scene_aliases()
        self._scene_cache = {}
        self._scan_scene_cache = {}
        if not self._scene_index:
            raise ValueError(f"no BIMNet scenes found below {self.root}")

    def _discover_scenes(self):
        index = {}
        splits = (self.selected_split,) if self.selected_split else ("train", "test")
        for split in splits:
            names = set()
            ifc_dir = self.root / "ifc" / split
            if ifc_dir.is_dir():
                names.update(path.stem for path in ifc_dir.glob("*.ifc"))
            matrix_dir = self.root / "mat_pc2obj" / split
            if matrix_dir.is_dir():
                names.update(path.stem for path in matrix_dir.glob("*.txt"))
            obj_dir = self.root / "obj" / split
            if obj_dir.is_dir():
                names.update(path.name for path in obj_dir.iterdir() if path.is_dir())
            for name in sorted(names):
                index[f"{split}/{name}".casefold()] = (name, split)
        return index

    def _build_scene_aliases(self):
        aliases = {}
        for key, (name, _) in self._scene_index.items():
            values = (key, name, BIMNET_MATTERPORT_SCANS.get(name.casefold()))
            for value in values:
                if value is None:
                    continue
                alias = str(value).replace("\\", "/").casefold()
                aliases.setdefault(alias, []).append(key)
        return {alias: tuple(dict.fromkeys(keys)) for alias, keys in aliases.items()}

    @staticmethod
    def _normalize_scene_identifier(identifier):
        value = str(identifier).strip().replace("\\", "/").rstrip("/")
        if not value:
            raise KeyError("BIMNet scene identifier cannot be empty")
        return value.casefold()

    @staticmethod
    def _path_scene_aliases(value):
        """Extract ``split/name`` and ``name`` aliases from an asset path."""

        path = PurePosixPath(value)
        parts = path.parts
        candidates = []

        for index, part in enumerate(parts[:-1]):
            if part in ("train", "test") and index + 1 < len(parts):
                name = PurePosixPath(parts[index + 1]).stem
                candidates.append(f"{part}/{name}")

        leaf = path.stem if path.suffix else path.name
        if leaf:
            candidates.append(leaf)
        return tuple(dict.fromkeys(candidates))

    def _scene_from_matches(self, identifier, matches):
        matches = tuple(dict.fromkeys(matches))
        if not matches:
            raise KeyError(
                f"unknown BIMNet scene: {identifier!r}; expected a BIMNet scene ID, "
                "split/name, Matterport scan ID, scene asset path, or integer index"
            )
        if len(matches) > 1:
            choices = ", ".join(self._scene_index[key][0] for key in matches)
            raise KeyError(
                f"ambiguous BIMNet scene {identifier!r}; matches: {choices}. "
                "Use split/name or scenes_for_scan() to select explicitly"
            )

        key = matches[0]
        if key not in self._scene_cache:
            name, split = self._scene_index[key]
            self._scene_cache[key] = BIMNetScene(self, name, split)
        return self._scene_cache[key]

    @property
    def scenes(self):
        return tuple(self.scene(key) for key in self._scene_index)

    @property
    def scene_ids(self):
        return [scene.scene_id for scene in self.scenes]

    @property
    def splits(self):
        return {
            split: tuple(scene for scene in self.scenes if scene.split == split)
            for split in ("train", "test")
        }

    def scene(self, identifier):
        """Find a scene from its index, IDs, split key, object, or asset path.

        Both BIMNet IDs (for example ``"hxp"``) and corresponding Matterport
        scan IDs (``"HxpKQynjfin"``) are accepted. A Matterport scan can map to
        several BIMNet floor scenes; a Matterport scan-ID lookup automatically
        returns a :class:`BIMNetScanScene` that merges all floors in their
        shared point-cloud coordinates. Use a BIMNet ID or ``split/name`` to
        retrieve one floor explicitly.
        """

        if isinstance(identifier, BIMNetScanScene):
            return identifier
        if isinstance(identifier, BIMNetScene):
            return self._scene_from_matches(identifier, (identifier.key.casefold(),))
        if isinstance(identifier, int):
            return self.scenes[identifier]
        if isinstance(identifier, (tuple, list)):
            if len(identifier) != 2:
                raise ValueError("a BIMNet scene tuple must contain (split, scene_id)")
            identifier = f"{identifier[0]}/{identifier[1]}"

        value = self._normalize_scene_identifier(identifier)
        direct_matches = self._scene_aliases.get(value, ())
        if direct_matches:
            if len(direct_matches) > 1:
                scan_ids = {
                    BIMNET_MATTERPORT_SCANS.get(self._scene_index[key][0].casefold())
                    for key in direct_matches
                }
                if len(scan_ids) == 1 and None not in scan_ids:
                    scan_id = next(iter(scan_ids))
                    if value == scan_id.casefold():
                        cache_key = scan_id.casefold()
                        if cache_key not in self._scan_scene_cache:
                            scenes = tuple(
                                self._scene_from_matches(identifier, (key,))
                                for key in direct_matches
                            )
                            self._scan_scene_cache[cache_key] = BIMNetScanScene(self, scenes)
                        return self._scan_scene_cache[cache_key]
            return self._scene_from_matches(identifier, direct_matches)

        path_matches = []
        for alias in self._path_scene_aliases(value):
            path_matches.extend(self._scene_aliases.get(alias, ()))
        return self._scene_from_matches(identifier, path_matches)

    def split(self, name):
        if name not in ("train", "test"):
            raise ValueError("split name must be 'train' or 'test'")
        return self.splits[name]

    def scenes_for_scan(self, matterport_scan_id):
        value = str(matterport_scan_id).casefold()
        return tuple(
            scene for scene in self.scenes if scene.matterport_scan_id.casefold() == value
        )

    def __len__(self):
        return len(self._scene_index)

    def __iter__(self):
        return iter(self.scenes)

    def __getitem__(self, index):
        return self.scene(index)

    def __repr__(self):
        split = self.selected_split or "all"
        return f"BIMNetDataset(root={str(self.root)!r}, split={split!r}, scenes={len(self)})"
