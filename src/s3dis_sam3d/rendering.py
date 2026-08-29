"""Shared rendering result models and image conventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import open3d as o3d
from PIL import Image

from .pointcloud import to_open3d_geometry


class MeshRaycaster:
    """One reusable CPU raycasting scene for all cameras in a BIMNet scene."""

    def __init__(self, mesh: o3d.geometry.TriangleMesh) -> None:
        if mesh.is_empty():
            raise ValueError("BIM mesh is empty")
        self.minimum = np.asarray(mesh.get_min_bound(), dtype=np.float64)
        self.maximum = np.asarray(mesh.get_max_bound(), dtype=np.float64)
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(tensor_mesh)

    def depth(
        self,
        intrinsics: np.ndarray,
        world_to_camera: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        rays = self.scene.create_rays_pinhole(
            np.asarray(intrinsics, dtype=np.float64),
            np.asarray(world_to_camera, dtype=np.float64),
            width,
            height,
        )
        depth = self.scene.cast_rays(rays)["t_hit"].numpy().astype(np.float32)
        depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
        return depth

    def contains_camera(self, camera_position: np.ndarray, margin: float) -> bool:
        position = np.asarray(camera_position, dtype=np.float64)
        return bool(
            np.all(position >= self.minimum - margin) and np.all(position <= self.maximum + margin)
        )


def render_geometries(
    geometries,
    *,
    intrinsics,
    world_to_camera,
    width,
    height,
    window_name="Open3D",
    background_color=(0.05, 0.05, 0.05),
    render_depth=True,
    show=False,
    mesh_show_back_face=True,
):
    """Render fixed-camera RGB-D buffers without requiring output paths."""

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name=window_name,
        width=width,
        height=height,
        visible=show,
    )
    try:
        for geometry in geometries:
            visualizer.add_geometry(to_open3d_geometry(geometry))

        intrinsics = np.asarray(intrinsics)
        parameters = o3d.camera.PinholeCameraParameters()
        parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsics[0, 0],
            intrinsics[1, 1],
            intrinsics[0, 2],
            intrinsics[1, 2],
        )
        parameters.extrinsic = np.asarray(world_to_camera)
        visualizer.get_view_control().convert_from_pinhole_camera_parameters(
            parameters,
            allow_arbitrary=True,
        )

        options = visualizer.get_render_option()
        options.background_color = np.asarray(background_color)
        options.mesh_show_back_face = mesh_show_back_face
        visualizer.poll_events()
        visualizer.update_renderer()

        rendered_image = np.asarray(
            visualizer.capture_screen_float_buffer(do_render=True),
            dtype=np.float32,
        ).copy()
        rendered_depth = None
        if render_depth:
            rendered_depth = np.asarray(
                visualizer.capture_depth_float_buffer(do_render=True),
                dtype=np.float32,
            ).copy()
        if show:
            visualizer.run()
        return rendered_image, rendered_depth
    finally:
        visualizer.destroy_window()


@dataclass(frozen=True)
class FrameRender:
    """Common source and rendered RGB-D data for one camera frame."""

    depth_scale: ClassVar[float] = 1000.0
    depth_invalid_value: ClassVar[int] = 0

    rendered_image_path: Path | None
    rendered_depth_path: Path | None
    source_image_path: Path
    source_depth_path: Path | None
    source_image: np.ndarray = field(repr=False)
    source_depth: np.ndarray | None = field(repr=False)
    rendered_image: np.ndarray = field(repr=False)
    rendered_depth: np.ndarray | None = field(repr=False)

    def __post_init__(self):
        for name in (
            "rendered_image_path",
            "rendered_depth_path",
            "source_image_path",
            "source_depth_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))

        for name in ("source_image", "rendered_image"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.ndim != 3 or value.shape[2] != 3:
                raise ValueError(f"{name} must have shape (H, W, 3)")
            object.__setattr__(self, name, value)

        for name in ("source_depth", "rendered_depth"):
            value = getattr(self, name)
            if value is None:
                continue
            value = np.asarray(value, dtype=np.float32)
            if value.ndim != 2:
                raise ValueError(f"{name} must have shape (H, W)")
            object.__setattr__(self, name, value)

        image_shape = self.source_image.shape[:2]
        if self.rendered_image.shape[:2] != image_shape:
            raise ValueError("source and rendered image sizes do not match")
        for name in ("source_depth", "rendered_depth"):
            depth = getattr(self, name)
            if depth is not None and depth.shape != image_shape:
                raise ValueError(f"{name} size does not match the RGB images")

    @staticmethod
    def read_image(path):
        """Read an RGB result as ``float32`` values in ``[0, 1]``."""

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    def save(self, output_path, *, depth_output_path=None, save_depth=True):
        """Save rendered RGB-D buffers and remember their output paths."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgb = np.clip(np.rint(self.rendered_image * 255), 0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(output_path)

        saved_depth_path = None
        if save_depth and self.rendered_depth is not None:
            saved_depth_path = Path(
                depth_output_path
                or output_path.with_name(f"{output_path.stem}_depth.png")
            )
            saved_depth_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(self._encoded_depth()).save(saved_depth_path)

        object.__setattr__(self, "rendered_image_path", output_path)
        object.__setattr__(self, "rendered_depth_path", saved_depth_path)
        return self

    def _encoded_depth(self):
        depth = self.rendered_depth
        if depth is None:
            raise ValueError("rendered depth is unavailable")
        valid = np.isfinite(depth) & (depth > 0)
        raw = np.full(depth.shape, self.depth_invalid_value, dtype=np.uint16)
        max_valid = 65534 if self.depth_invalid_value == 65535 else 65535
        scaled = np.rint(depth[valid] * self.depth_scale)
        raw[valid] = np.clip(scaled, 1, max_valid).astype(np.uint16)
        return raw

    def depth_limits(self, depth_range=None, percentiles=None):
        """Return one depth range jointly computed from source and rendering."""

        if depth_range is not None:
            if len(depth_range) != 2:
                raise ValueError("depth_range must contain (min_depth, max_depth)")
            minimum, maximum = (float(value) for value in depth_range)
            if not np.isfinite((minimum, maximum)).all() or minimum >= maximum:
                raise ValueError("depth_range must be finite and strictly increasing")
            return minimum, maximum

        valid_depths = []
        for depth in (self.source_depth, self.rendered_depth):
            if depth is None:
                continue
            valid = depth[np.isfinite(depth) & (depth > 0)]
            if len(valid):
                valid_depths.append(valid)
        if not valid_depths:
            raise ValueError("valid source or rendered depth is unavailable")

        values = np.concatenate(valid_depths)
        if percentiles is None:
            minimum, maximum = float(values.min()), float(values.max())
        else:
            if len(percentiles) != 2:
                raise ValueError("percentiles must contain (lower, upper)")
            lower, upper = (float(value) for value in percentiles)
            if not 0 <= lower < upper <= 100:
                raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
            minimum, maximum = np.percentile(values, (lower, upper)).tolist()

        if minimum == maximum:
            epsilon = max(abs(minimum), 1.0) * 1e-6
            minimum, maximum = max(0.0, minimum - epsilon), maximum + epsilon
        return minimum, maximum

    def visualize(
        self,
        *,
        mode="rgb",
        depth_range=None,
        depth_percentiles=None,
        colormap="turbo",
        invalid_depth_color=(0.15, 0.15, 0.15, 1.0),
        figsize=(12, 5),
        save_path=None,
        show=True,
        dpi=150,
    ):
        """Compare source/rendered RGB or depth in one selected mode."""

        from matplotlib import pyplot as plt
        from matplotlib.colors import Normalize

        mode = str(mode).casefold()
        if mode not in {"rgb", "depth"}:
            raise ValueError("mode must be 'rgb' or 'depth'")
        figure, axes = plt.subplots(
            1,
            2,
            figsize=figsize,
            constrained_layout=True,
            squeeze=True,
        )

        if mode == "rgb":
            for axis, image, title in (
                (axes[0], self.source_image, "Source RGB"),
                (axes[1], self.rendered_image, "Rendered RGB"),
            ):
                axis.imshow(np.clip(image, 0, 1))
                axis.set_title(title)
                axis.axis("off")
        else:
            minimum, maximum = self.depth_limits(depth_range, depth_percentiles)
            normalization = Normalize(vmin=minimum, vmax=maximum, clip=True)
            depth_colormap = plt.get_cmap(colormap).copy()
            depth_colormap.set_bad(invalid_depth_color)
            heatmap = None
            for axis, depth, title in (
                (axes[0], self.source_depth, "Source depth"),
                (axes[1], self.rendered_depth, "Rendered depth"),
            ):
                axis.set_title(title)
                axis.axis("off")
                if depth is None:
                    axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
                    continue
                masked = np.ma.masked_where(~np.isfinite(depth) | (depth <= 0), depth)
                heatmap = axis.imshow(masked, cmap=depth_colormap, norm=normalization)
            if heatmap is not None:
                colorbar = figure.colorbar(
                    heatmap,
                    ax=axes.tolist(),
                    label="Depth (m)",
                    shrink=0.9,
                )
                colorbar.ax.set_title(f"{minimum:g}–{maximum:g} m", fontsize=9)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(save_path, dpi=dpi)
        if show:
            plt.show()
            return None
        return figure

    @property
    def image_shape(self):
        return self.source_image.shape[:2]

    @property
    def has_source_depth(self):
        return self.source_depth is not None

    @property
    def has_rendered_depth(self):
        return self.rendered_depth is not None
