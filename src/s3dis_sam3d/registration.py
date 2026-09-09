"""Robust upright registration for IFC and indoor structural geometry."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class YawRegistration:
    transform: np.ndarray
    metrics: dict
    candidates: list[dict]
    semantic_audit: dict
    quality_checks: dict[str, bool]

    @property
    def accepted(self):
        return all(self.quality_checks.values())


def stable_seed(seed, name, stream):
    value = f"{seed}\0{name}\0{stream}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def sample_labeled_mesh(vertices, triangles, labels, count, seed):
    """Uniformly sample a triangle mesh while retaining one label per face."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int32)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(labels) != len(triangles):
        raise ValueError("triangles and face labels must have matching lengths")

    corners = vertices[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )
    valid = np.isfinite(areas) & (areas > 0)
    if not valid.any():
        raise ValueError("mesh has no finite, non-degenerate triangles")
    corners, labels, areas = corners[valid], labels[valid], areas[valid]

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(corners), size=count, replace=True, p=areas / areas.sum())
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, :1])
    weights = np.concatenate((1 - root, root * (1 - uv[:, 1:]), root * uv[:, 1:]), axis=1)
    points = np.sum(corners[selected] * weights[:, :, None], axis=1)
    return points, labels[selected]


def downsample_points(points, labels, count, seed):
    points = np.asarray(points, dtype=np.float64)
    labels = None if labels is None else np.asarray(labels, dtype=np.int32)
    if len(points) <= count:
        return points, labels
    indices = np.random.default_rng(seed).choice(len(points), count, replace=False)
    return points[indices], None if labels is None else labels[indices]


def _yaw_transform(yaw, translation):
    c, s = math.cos(yaw), math.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = ((c, -s, 0), (s, c, 0), (0, 0, 1))
    transform[:3, 3] = translation
    return transform


def _apply(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def _initial_transform(source, target, yaw):
    rotated = source @ _yaw_transform(yaw, np.zeros(3))[:3, :3].T
    source_bounds = np.quantile(rotated[:, :2], (0.02, 0.98), axis=0)
    target_bounds = np.quantile(target[:, :2], (0.02, 0.98), axis=0)
    translation = np.r_[
        0.5 * (target_bounds[0] + target_bounds[1] - source_bounds[0] - source_bounds[1]),
        np.quantile(target[:, 2], 0.01) - np.quantile(rotated[:, 2], 0.01),
    ]
    return _yaw_transform(yaw, translation)


def _clipped_symmetric_rmse(source, target, transform, clip):
    transformed = _apply(source, transform)
    forward = cKDTree(target).query(transformed, workers=1)[0]
    reverse = cKDTree(transformed).query(target, workers=1)[0]
    distances = np.minimum(np.r_[forward, reverse], clip)
    return float(np.sqrt(np.mean(distances**2)))


def _trim(distances, maximum, fraction):
    valid = np.isfinite(distances) & (distances <= maximum)
    if valid.sum() < 3 or fraction >= 1:
        return valid
    limit = np.quantile(distances[valid], fraction)
    return valid & (distances <= limit)


def _fit_yaw_translation(source, target, weights):
    weights = weights / weights.sum()
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    source_xy = source[:, :2] - source_center[:2]
    target_xy = target[:, :2] - target_center[:2]
    covariance = source_xy.T @ (weights[:, None] * target_xy)
    yaw = math.atan2(covariance[0, 1] - covariance[1, 0], covariance[0, 0] + covariance[1, 1])
    rotation = _yaw_transform(yaw, np.zeros(3))[:3, :3]
    return _yaw_transform(yaw, target_center - rotation @ source_center)


def _correspondences(source, target, transform, maximum, trim_fraction, huber_delta):
    transformed = _apply(source, transform)
    target_tree = cKDTree(target)
    forward_distance, forward_index = target_tree.query(transformed, workers=1)
    source_tree = cKDTree(transformed)
    reverse_distance, reverse_index = source_tree.query(target, workers=1)
    forward = _trim(forward_distance, maximum, trim_fraction)
    reverse = _trim(reverse_distance, maximum, trim_fraction)
    if forward.sum() < 3 or reverse.sum() < 3:
        raise RuntimeError("too few symmetric correspondences")

    source_pairs = np.r_[source[forward], source[reverse_index[reverse]]]
    target_pairs = np.r_[target[forward_index[forward]], target[reverse]]
    distances = np.r_[forward_distance[forward], reverse_distance[reverse]]
    weights = np.minimum(1.0, huber_delta / np.maximum(distances, np.finfo(float).eps))
    split = int(forward.sum())
    weights[:split] *= 0.5 / weights[:split].sum()
    weights[split:] *= 0.5 / weights[split:].sum()
    return source_pairs, target_pairs, weights


def _refine(source, target, initial, distances, iterations, trim_fraction, huber_delta):
    transform = initial.copy()
    completed = 0
    for maximum in distances:
        for _ in range(iterations):
            pairs = _correspondences(source, target, transform, maximum, trim_fraction, huber_delta)
            updated = _fit_yaw_translation(*pairs)
            yaw_old = math.atan2(transform[1, 0], transform[0, 0])
            yaw_new = math.atan2(updated[1, 0], updated[0, 0])
            yaw_change = abs((yaw_new - yaw_old + math.pi) % (2 * math.pi) - math.pi)
            shift = np.linalg.norm(updated[:3, 3] - transform[:3, 3])
            transform = updated
            completed += 1
            if yaw_change <= 1e-6 and shift <= 1e-5:
                break
    return transform, completed


def _metrics(source, target, transform, threshold, trim_fraction):
    transformed = _apply(source, transform)
    forward = cKDTree(target).query(transformed, workers=1)[0]
    reverse = cKDTree(transformed).query(target, workers=1)[0]
    both = np.r_[forward, reverse]
    inliers = np.r_[forward[forward <= threshold], reverse[reverse <= threshold]]
    keep = max(1, math.ceil(len(both) * trim_fraction))
    trimmed = np.partition(both, keep - 1)[:keep]
    fitness = 0.5 * (np.mean(forward <= threshold) + np.mean(reverse <= threshold))
    return {
        "fitness": float(fitness),
        "rmse": float(np.sqrt(np.mean(inliers**2))),
        "symmetric_trimmed_rmse": float(np.sqrt(np.mean(trimmed**2))),
        "symmetric_median": float(np.median(both)),
        "symmetric_p90": float(np.quantile(both, 0.90)),
    }


def _semantic_score(source, target, source_labels, target_labels, transform, label_names, options):
    transformed = _apply(source, transform)
    scores, discriminative = [], []
    details = {}
    for label, name in enumerate(label_names):
        source_part = transformed[source_labels == label]
        target_part = target[target_labels == label]
        if min(len(source_part), len(target_part)) < options["min_points"]:
            continue
        distances = np.r_[
            cKDTree(target_part).query(source_part, workers=1)[0],
            cKDTree(source_part).query(target_part, workers=1)[0],
        ]
        clipped = np.minimum(distances, options["clip"])
        keep = max(1, math.ceil(len(clipped) * options["trim_fraction"]))
        score = float(np.sqrt(np.mean(np.partition(clipped, keep - 1)[:keep] ** 2)))
        is_discriminative = name in options["discriminative"]
        weight = options["weight"] if is_discriminative else 1.0
        scores.append((score, weight))
        if is_discriminative:
            discriminative.append(score)
        details[name] = {
            "score": score,
            "source_points": len(source_part),
            "target_points": len(target_part),
        }
    if not scores or not discriminative:
        return None
    return {
        "score": sum(score * weight for score, weight in scores)
        / sum(weight for _, weight in scores),
        "discriminative_score": float(np.mean(discriminative)),
        "per_class": details,
    }


def register_upright(
    source,
    target,
    *,
    source_labels=None,
    target_labels=None,
    label_names=(),
    yaw_starts=36,
    yaw_angles=None,
    refine_candidates=4,
    coarse_points=1500,
    distances=(1.5, 0.75, 0.35, 0.18),
    iterations=25,
    trim_fraction=0.85,
    huber_delta=0.12,
    metric_threshold=0.20,
    min_fitness=0.55,
    max_rmse=0.15,
    semantic_min_points=24,
    semantic_tolerance=0.03,
    semantic_min_improvement=0.02,
    semantic_discriminative=("door", "window", "beam", "column"),
    semantic_weight=3.0,
):
    """Estimate a unit-scale, Z-up transform with yaw and XYZ translation only."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1:] != (3,) or target.shape[1:] != (3,):
        raise ValueError("source and target must be Nx3 point arrays")
    if min(len(source), len(target)) < max(100, coarse_points):
        raise ValueError("not enough points for upright registration")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("registration points must be finite")

    coarse_source = source[np.linspace(0, len(source) - 1, coarse_points, dtype=int)]
    coarse_target = target[np.linspace(0, len(target) - 1, coarse_points, dtype=int)]
    angles = (
        [-math.pi + 2 * math.pi * index / yaw_starts for index in range(yaw_starts)]
        if yaw_angles is None
        else [float(value) for value in yaw_angles]
    )
    if len(angles) < refine_candidates:
        raise ValueError("yaw candidates must be at least refine_candidates")
    candidates = []
    for index, yaw in enumerate(angles):
        initial = _initial_transform(coarse_source, coarse_target, yaw)
        candidates.append(
            {
                "index": index,
                "initial_yaw": yaw,
                "coarse_score": _clipped_symmetric_rmse(
                    coarse_source, coarse_target, initial, distances[0]
                ),
                "initial": initial,
            }
        )

    refined = []
    for candidate in sorted(candidates, key=lambda item: item["coarse_score"])[:refine_candidates]:
        try:
            transform, completed = _refine(
                source,
                target,
                candidate["initial"],
                distances,
                iterations,
                trim_fraction,
                huber_delta,
            )
            metrics = _metrics(source, target, transform, metric_threshold, trim_fraction)
            objective = metrics["symmetric_trimmed_rmse"] + metric_threshold * (
                1 - metrics["fitness"]
            )
            candidate.update(
                transform=transform,
                iterations=completed,
                metrics=metrics,
                objective=float(objective),
            )
            refined.append(candidate)
        except (RuntimeError, ValueError) as error:
            candidate["error"] = f"{type(error).__name__}: {error}"

    if not refined:
        raise RuntimeError("all upright registration candidates failed")
    refined.sort(key=lambda item: item["objective"])
    selected = refined[0]
    semantic_audit = {"enabled": False, "changed_geometry_best": False}
    labels_available = source_labels is not None and target_labels is not None and label_names
    if labels_available:
        options = {
            "min_points": semantic_min_points,
            "clip": 0.75,
            "trim_fraction": 0.90,
            "discriminative": set(semantic_discriminative),
            "weight": semantic_weight,
        }
        for candidate in refined:
            candidate["semantic"] = _semantic_score(
                source,
                target,
                np.asarray(source_labels),
                np.asarray(target_labels),
                candidate["transform"],
                label_names,
                options,
            )
        geometry_yaw = math.atan2(
            refined[0]["transform"][1, 0],
            refined[0]["transform"][0, 0],
        )
        comparable = [
            item
            for item in refined
            if item["semantic"] is not None
            and item["objective"] <= refined[0]["objective"] + semantic_tolerance
            and item["metrics"]["fitness"] >= min_fitness
            and item["metrics"]["rmse"] <= max_rmse
            and (
                item is refined[0]
                or abs(
                    (
                        math.atan2(item["transform"][1, 0], item["transform"][0, 0])
                        - geometry_yaw
                        + math.pi
                    )
                    % (2 * math.pi)
                    - math.pi
                )
                >= math.pi / 6
            )
        ]
        geometry_is_usable = refined[0] in comparable
        semantic_audit["enabled"] = geometry_is_usable
        if geometry_is_usable:
            semantic_best = min(comparable, key=lambda item: item["semantic"]["score"])
            geometry_semantic = selected["semantic"]
            improvement = geometry_semantic["score"] - semantic_best["semantic"]["score"]
            discriminative_improvement = (
                geometry_semantic["discriminative_score"]
                - semantic_best["semantic"]["discriminative_score"]
            )
            if (
                improvement >= semantic_min_improvement
                and discriminative_improvement >= semantic_min_improvement
            ):
                selected = semantic_best
            semantic_audit.update(
                geometry_best_index=refined[0]["index"],
                semantic_best_index=semantic_best["index"],
                selected_index=selected["index"],
                improvement=float(improvement),
                discriminative_improvement=float(discriminative_improvement),
                changed_geometry_best=selected is not refined[0],
            )

    metrics = selected["metrics"]
    checks = {
        "minimum_fitness": metrics["fitness"] >= min_fitness,
        "maximum_rmse": metrics["rmse"] <= max_rmse,
        "unit_scale": True,
        "z_axis_up": True,
    }
    audit = []
    for item in candidates:
        row = {key: value for key, value in item.items() if key not in {"initial", "transform"}}
        if "transform" in item:
            row["transform"] = item["transform"].tolist()
        row["selected"] = item is selected
        audit.append(row)
    return YawRegistration(selected["transform"], metrics, audit, semantic_audit, checks)
