"""Room-level BIM-like depth generated from labeled structural meshes."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from .registration import stable_seed
from .rendering import MeshRaycaster


class SyncBIMScene:
    """Fit a simple wall/floor shell once and render calibrated RGB-D frames."""

    def __init__(
        self,
        scene,
        sample_points=100_000,
        distance_threshold=0.03,
        seed=20260906,
    ):
        self.scene = scene
        self.distance_threshold = float(distance_threshold)
        rng = np.random.default_rng(stable_seed(seed, scene.key, "syncbim"))
        self.planes = self._fit_planes(scene, int(sample_points), rng)
        self.mesh, self.statistics = self._build_mesh(self.planes)
        self.raycaster = MeshRaycaster(self.mesh)

    def render_depth(self, frame, size=None, *, intrinsics=None):
        if size is None:
            height, width = frame.image_shape
        else:
            height, width = size
        if intrinsics is None:
            intrinsics = frame.intrinsics if size is None else frame.intrinsics_for_size(size)
        return self.raycaster.depth(
            intrinsics,
            frame.world_to_camera,
            width,
            height,
        )

    def _fit_plane(self, points, rng):
        fit_points = points
        if len(fit_points) > 8000:
            indices = rng.choice(len(fit_points), size=8000, replace=False)
            fit_points = fit_points[indices]

        triplets = rng.integers(0, len(fit_points), size=(160, 3))
        point_a = fit_points[triplets[:, 0]]
        point_b = fit_points[triplets[:, 1]]
        point_c = fit_points[triplets[:, 2]]
        normals = np.cross(point_b - point_a, point_c - point_a)
        lengths = np.linalg.norm(normals, axis=1)
        keep = lengths > 1e-8
        if not np.any(keep):
            return None
        normals = normals[keep] / lengths[keep, None]
        offsets = -np.sum(normals * point_a[keep], axis=1)
        distances = np.abs(fit_points @ normals.T + offsets)
        best = int(
            np.argmax(
                np.count_nonzero(
                    distances < self.distance_threshold,
                    axis=0,
                )
            )
        )
        residuals = np.abs(points @ normals[best] + offsets[best])
        inliers = residuals < self.distance_threshold
        if np.count_nonzero(inliers) < 3:
            return None

        inlier_points = points[inliers]
        center = np.mean(inlier_points, axis=0)
        covariance = (inlier_points - center).T @ (inlier_points - center)
        _, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        normal /= np.linalg.norm(normal)
        offset = -float(normal @ center)
        residuals = np.abs(points @ normal + offset)
        inliers = residuals < self.distance_threshold
        return normal, offset, inliers, float(np.median(residuals[inliers]))

    @staticmethod
    def _orient(normal, offset, camera_positions):
        if np.median(camera_positions @ normal + offset) < 0:
            return -normal, -offset
        return normal, offset

    def _fit_planes(self, scene, sample_points, rng):
        cloud = scene.structural_point_cloud(
            include_classes=("floor", "wall", "ceiling"),
            sample_points=sample_points,
        )
        point_groups = {}
        for instance_id, record in cloud.metadata["instances"].items():
            mask = cloud.instance_labels == instance_id
            if np.any(mask):
                point_groups[(record["class_name"], instance_id)] = cloud.xyz[mask]

        camera_positions = np.stack(
            [frame.camera_to_world[:3, 3] for frame in scene.frames]
        ).astype(np.float64)
        floor_points = np.concatenate(
            [
                points
                for (class_name, _), points in point_groups.items()
                if class_name == "floor"
            ]
        )
        floor_fit = self._fit_plane(floor_points, rng)
        if floor_fit is None:
            raise ValueError("could not fit floor for " + scene.key)
        floor_normal, floor_offset, floor_inliers, floor_residual = floor_fit
        floor_normal, floor_offset = self._orient(
            floor_normal,
            floor_offset,
            camera_positions,
        )
        planes = [
            {
                "class_name": "floor",
                "instance_id": -1,
                "normal": floor_normal,
                "offset": floor_offset,
                "source_points": len(floor_points),
                "inlier_ratio": float(np.mean(floor_inliers)),
                "median_residual_m": floor_residual,
                "_points": floor_points[floor_inliers],
            }
        ]

        candidates = []
        for (class_name, instance_id), points in point_groups.items():
            if class_name != "wall" or len(points) < 600:
                continue
            fit = self._fit_plane(points, rng)
            if fit is None:
                continue
            normal, _, inliers, residual = fit
            if np.mean(inliers) < 0.60:
                continue
            normal -= np.dot(normal, floor_normal) * floor_normal
            length = np.linalg.norm(normal)
            if length < 1e-6:
                continue
            normal /= length
            offset = -float(np.median(points[inliers] @ normal))
            normal, offset = self._orient(normal, offset, camera_positions)
            candidates.append(
                {
                    "class_name": "wall",
                    "instance_id": instance_id,
                    "normal": normal,
                    "offset": offset,
                    "source_points": len(points),
                    "inlier_ratio": float(np.mean(inliers)),
                    "median_residual_m": residual,
                    "_points": points[inliers],
                }
            )

        candidates.sort(key=lambda item: item["source_points"], reverse=True)
        for candidate in candidates:
            duplicate = any(
                np.dot(candidate["normal"], plane["normal"]) > np.cos(np.deg2rad(4.0))
                and abs(candidate["offset"] - plane["offset"]) < 0.10
                for plane in planes[1:]
            )
            if not duplicate:
                planes.append(candidate)

        ceiling_groups = [
            points
            for (class_name, _), points in point_groups.items()
            if class_name == "ceiling"
        ]
        if ceiling_groups:
            ceiling_points = np.concatenate(ceiling_groups)
            ceiling_fit = self._fit_plane(ceiling_points, rng)
            if ceiling_fit is not None:
                _, _, ceiling_inliers, ceiling_residual = ceiling_fit
                ceiling_offset = -float(
                    np.median(ceiling_points[ceiling_inliers] @ floor_normal)
                )
                ceiling_normal, ceiling_offset = self._orient(
                    floor_normal.copy(),
                    ceiling_offset,
                    camera_positions,
                )
                planes.append(
                    {
                        "class_name": "ceiling_clip",
                        "instance_id": -1,
                        "normal": ceiling_normal,
                        "offset": ceiling_offset,
                        "source_points": len(ceiling_points),
                        "inlier_ratio": float(np.mean(ceiling_inliers)),
                        "median_residual_m": ceiling_residual,
                        "_points": ceiling_points[ceiling_inliers],
                    }
                )

        if sum(plane["class_name"] == "wall" for plane in planes) < 2:
            raise ValueError("too few walls fitted for " + scene.key)
        return planes

    @staticmethod
    def _intersection(planes):
        matrix = np.stack([plane["normal"] for plane in planes])
        values = -np.asarray([plane["offset"] for plane in planes])
        if abs(np.linalg.det(matrix)) < 1e-8:
            return None
        return np.linalg.solve(matrix, values)

    @staticmethod
    def _project_to_plane(points, plane):
        normal = plane["normal"]
        distance = points @ normal + plane["offset"]
        return points - distance[..., None] * normal

    def _slab_triangles(self, class_name, plane, reference_normal):
        room = self.scene.dataset.semantic_mesh.room(self.scene.name)
        class_id = room.dataset.semantic_classes.index(class_name)
        triangles = room.vertices[room.triangles[room.face_labels == class_id]].astype(
            np.float64
        )
        if not len(triangles):
            return triangles

        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        double_area = np.linalg.norm(cross, axis=1)
        alignment = np.abs(cross @ reference_normal) / np.maximum(double_area, 1e-12)
        triangles = triangles[(double_area > 1e-8) & (alignment > 0.75)]
        if not len(triangles):
            return triangles

        triangles = self._project_to_plane(triangles, plane)
        projected_cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        return triangles[np.linalg.norm(projected_cross, axis=1) > 1e-8]

    @staticmethod
    def _append_triangles(vertices, triangles, points):
        offset = len(vertices)
        vertices.extend(points.reshape(-1, 3))
        triangles.extend(
            np.arange(offset, offset + 3 * len(points), dtype=np.int32).reshape(-1, 3)
        )

    def _build_mesh(self, planes):
        floor = next(plane for plane in planes if plane["class_name"] == "floor")
        ceiling = next(
            (plane for plane in planes if plane["class_name"] == "ceiling_clip"),
            None,
        )
        walls = [plane for plane in planes if plane["class_name"] == "wall"]
        floor_normal = floor["normal"]
        floor_offset = floor["offset"]
        wall_heights = np.concatenate(
            [wall["_points"] @ floor_normal + floor_offset for wall in walls]
        )
        if ceiling is None:
            ceiling_height = float(np.percentile(wall_heights, 99))
        else:
            ceiling_height = float(
                np.median(ceiling["_points"] @ floor_normal) + floor_offset
            )
        if not 1.5 < ceiling_height < 6.0:
            ceiling_height = float(np.percentile(wall_heights, 99))
        if not 1.5 < ceiling_height < 6.0:
            raise ValueError("implausible room height")

        segments = []
        for wall in walls:
            direction = np.cross(floor_normal, wall["normal"])
            length = np.linalg.norm(direction)
            if length < 1e-7:
                continue
            direction /= length
            matrix = np.stack((floor_normal, wall["normal"]))
            values = -np.asarray((floor_offset, wall["offset"]))
            base = matrix.T @ np.linalg.solve(matrix @ matrix.T, values)
            coordinates = (wall["_points"] - base) @ direction
            lower, upper = np.percentile(coordinates, (1, 99))
            segments.append(
                {
                    "plane": wall,
                    "base": base,
                    "direction": direction,
                    "lower": float(lower),
                    "upper": float(upper),
                }
            )

        vertices = []
        triangles = []
        for index, segment in enumerate(segments):
            center = 0.5 * (segment["lower"] + segment["upper"])
            intersections = []
            for other_index, other in enumerate(segments):
                if other_index == index:
                    continue
                point = self._intersection((floor, segment["plane"], other["plane"]))
                if point is not None:
                    intersections.append(
                        float((point - segment["base"]) @ segment["direction"])
                    )
            lower = [value for value in intersections if value < center]
            upper = [value for value in intersections if value > center]
            if lower:
                nearest = min(lower, key=lambda value: abs(value - segment["lower"]))
                if abs(nearest - segment["lower"]) < 1.5:
                    segment["lower"] = nearest
            else:
                segment["lower"] -= 0.25
            if upper:
                nearest = min(upper, key=lambda value: abs(value - segment["upper"]))
                if abs(nearest - segment["upper"]) < 1.5:
                    segment["upper"] = nearest
            else:
                segment["upper"] += 0.25

            bottom_a = segment["base"] + segment["lower"] * segment["direction"]
            bottom_b = segment["base"] + segment["upper"] * segment["direction"]
            top_a = bottom_a + ceiling_height * floor_normal
            top_b = bottom_b + ceiling_height * floor_normal
            offset = len(vertices)
            vertices.extend((bottom_a, bottom_b, top_b, top_a))
            triangles.extend(
                ((offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3))
            )

        floor_triangles = self._slab_triangles("floor", floor, floor_normal)
        self._append_triangles(vertices, triangles, floor_triangles)

        if ceiling is None:
            ceiling_triangles = floor_triangles + ceiling_height * floor_normal
        else:
            ceiling_triangles = self._slab_triangles(
                "ceiling", ceiling, floor_normal
            )
            if not len(ceiling_triangles):
                ceiling_triangles = floor_triangles + ceiling_height * floor_normal
        self._append_triangles(vertices, triangles, ceiling_triangles)

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(vertices)),
            o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
        )
        return mesh, {
            "planes": len(planes),
            "walls": len(segments),
            "floor_triangles": len(floor_triangles),
            "ceiling_triangles": len(ceiling_triangles),
            "ceiling_height_m": ceiling_height,
            "vertices": len(vertices),
            "triangles": len(triangles),
        }
