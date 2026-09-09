"""Small IFC geometry helpers used by BIMSync."""

from pathlib import Path

import numpy as np
import open3d as o3d

STRUCTURAL_CLASSES = (
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
    "IfcCurtainWall",
    "IfcRoof",
}


def raycaster_mesh_options(include_types):
    """Return reusable IFC type arguments and their canonical cache key."""
    if include_types is None:
        return None, None
    if isinstance(include_types, str):
        include_types = (include_types,)
    include_types = tuple(include_types)
    cache_key = frozenset(str(ifc_type).casefold() for ifc_type in include_types)
    return include_types, cache_key


def load_ifc_mesh(path, include_types=None):
    """Load IFC products into one Open3D mesh in IFC world coordinates."""
    import ifcopenshell
    import ifcopenshell.geom

    path = Path(path)
    model = ifcopenshell.open(str(path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    include_types = (
        None if include_types is None else {str(ifc_type).casefold() for ifc_type in include_types}
    )

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


def ifc_structural_class(product):
    """Map an IFC product onto an indoor structural class."""
    predefined = str(getattr(product, "PredefinedType", "") or "").upper()
    if product.is_a("IfcWall") or product.is_a("IfcCurtainWall"):
        return "wall"
    if product.is_a("IfcSlab"):
        if predefined == "ROOF":
            return "ceiling"
        if predefined in {"", "NOTDEFINED", "USERDEFINED", "FLOOR", "BASESLAB"}:
            return "floor"
        return None
    if product.is_a("IfcCovering"):
        return {"CEILING": "ceiling", "FLOORING": "floor"}.get(predefined)
    if product.is_a("IfcRoof"):
        return "ceiling"
    for ifc_class, name in (
        ("IfcDoor", "door"),
        ("IfcWindow", "window"),
        ("IfcColumn", "column"),
        ("IfcBeam", "beam"),
    ):
        if product.is_a(ifc_class):
            return name
    return None


def load_labeled_ifc_geometry(path, include_types=STRUCTURAL_IFC_TYPES):
    """Load structural IFC triangles and retain their semantic class IDs."""
    import ifcopenshell
    import ifcopenshell.geom

    model = ifcopenshell.open(str(Path(path)))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    include_types = {str(name).casefold() for name in include_types}
    class_to_id = {name: index for index, name in enumerate(STRUCTURAL_CLASSES)}
    vertices, triangles, labels = [], [], []
    offset = 0

    for product in sorted(model.by_type("IfcProduct"), key=lambda item: int(item.id())):
        class_name = ifc_structural_class(product)
        if (
            class_name is None
            or product.is_a().casefold() not in include_types
            or product.Representation is None
        ):
            continue
        shape = ifcopenshell.geom.create_shape(settings, product)
        product_vertices = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
        product_faces = np.asarray(shape.geometry.faces, dtype=np.int32).reshape(-1, 3)
        if not len(product_vertices) or not len(product_faces):
            continue
        vertices.append(product_vertices)
        triangles.append(product_faces + offset)
        labels.append(np.full(len(product_faces), class_to_id[class_name], dtype=np.int32))
        offset += len(product_vertices)

    if not vertices:
        raise RuntimeError(f"no structural IFC geometry found: {path}")
    return np.vstack(vertices), np.vstack(triangles), np.concatenate(labels)
