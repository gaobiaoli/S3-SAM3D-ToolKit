"""Common monocular metric-depth evaluation."""

from __future__ import annotations

import math

import numpy as np

METRIC_NAMES = (
    "abs_rel",
    "sq_rel",
    "rmse",
    "mae",
    "rmse_log",
    "mean_abs_log_error",
    "log10",
    "silog_x100",
    "delta1",
    "delta2",
    "delta3",
)

SUM_NAMES = (
    "abs_rel",
    "sq_rel",
    "absolute_error",
    "squared_error",
    "log_error",
    "abs_log_error",
    "squared_log_error",
    "log10_error",
    "delta1",
    "delta2",
    "delta3",
)


def _empty_sums():
    return {name: 0.0 for name in SUM_NAMES}


def _metric_sums(prediction, target):
    error = prediction - target
    absolute_error = np.abs(error)
    log_error = np.log(prediction) - np.log(target)
    ratio = np.maximum(prediction / target, target / prediction)
    return {
        "abs_rel": float(np.sum(absolute_error / target)),
        "sq_rel": float(np.sum(error**2 / target)),
        "absolute_error": float(np.sum(absolute_error)),
        "squared_error": float(np.sum(error**2)),
        "log_error": float(np.sum(log_error)),
        "abs_log_error": float(np.sum(np.abs(log_error))),
        "squared_log_error": float(np.sum(log_error**2)),
        "log10_error": float(np.sum(np.abs(np.log10(prediction) - np.log10(target)))),
        "delta1": float(np.sum(ratio < 1.25)),
        "delta2": float(np.sum(ratio < 1.25**2)),
        "delta3": float(np.sum(ratio < 1.25**3)),
    }


def _metrics_from_sums(sums, count):
    if count == 0:
        result = {name: float("nan") for name in METRIC_NAMES}
        result["count"] = 0
        return result

    mean_log_error = sums["log_error"] / count
    log_variance = max(
        0.0,
        sums["squared_log_error"] / count - mean_log_error**2,
    )
    return {
        "abs_rel": sums["abs_rel"] / count,
        "sq_rel": sums["sq_rel"] / count,
        "rmse": math.sqrt(sums["squared_error"] / count),
        "mae": sums["absolute_error"] / count,
        "rmse_log": math.sqrt(sums["squared_log_error"] / count),
        "mean_abs_log_error": sums["abs_log_error"] / count,
        "log10": sums["log10_error"] / count,
        "silog_x100": 100.0 * math.sqrt(log_variance),
        "delta1": sums["delta1"] / count,
        "delta2": sums["delta2"] / count,
        "delta3": sums["delta3"] / count,
        "count": count,
    }


class DepthMetricAccumulator:
    """Accumulate pixel-micro and frame-macro depth metrics together.

    Each ``update`` is treated as one frame. Call it once per sample when
    evaluating a batch.
    """

    def __init__(self):
        self.frames = 0
        self.valid_pixels = 0
        self.pixel_sums = _empty_sums()
        self.frame_sums = {name: 0.0 for name in METRIC_NAMES}

    def update(self, prediction, target, valid=None):
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")

        if valid is None:
            support = np.isfinite(target) & (target > 0)
        else:
            support = np.asarray(valid, dtype=bool)
            if support.shape != target.shape:
                raise ValueError("valid mask and target shapes must match")
            if np.any(support & (~np.isfinite(target) | (target <= 0))):
                raise ValueError("target is invalid on the fixed support")

        pred = np.asarray(prediction[support], dtype=np.float64)
        gt = np.asarray(target[support], dtype=np.float64)
        if pred.size == 0:
            return
        if not np.isfinite(pred).all() or np.any(pred <= 0):
            raise ValueError("prediction must be finite and positive on the fixed support")

        count = int(pred.size)
        sums = _metric_sums(pred, gt)
        frame_metrics = _metrics_from_sums(sums, count)

        self.frames += 1
        self.valid_pixels += count
        for name in SUM_NAMES:
            self.pixel_sums[name] += sums[name]
        for name in METRIC_NAMES:
            self.frame_sums[name] += frame_metrics[name]

    def merge(self, other):
        """Add another accumulator, for example a completed scene."""
        if not isinstance(other, DepthMetricAccumulator):
            raise TypeError("other must be a DepthMetricAccumulator")
        self.frames += other.frames
        self.valid_pixels += other.valid_pixels
        for name in SUM_NAMES:
            self.pixel_sums[name] += other.pixel_sums[name]
        for name in METRIC_NAMES:
            self.frame_sums[name] += other.frame_sums[name]
        return self

    def compute(self):
        """Return pixel-micro and frame-macro rows with their respective counts."""
        pixel_micro = _metrics_from_sums(self.pixel_sums, self.valid_pixels)
        if self.frames == 0:
            frame_macro = _metrics_from_sums(_empty_sums(), 0)
        else:
            frame_macro = {name: self.frame_sums[name] / self.frames for name in METRIC_NAMES}
            frame_macro["count"] = self.frames
        return {
            "pixel_micro": pixel_micro,
            "frame_macro": frame_macro,
        }


def depth_metrics(prediction, target, valid=None):
    """Compute standard scores for one depth frame."""
    metrics = DepthMetricAccumulator()
    metrics.update(prediction, target, valid)
    return metrics.compute()["pixel_micro"]
