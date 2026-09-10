import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from s3dis_sam3d.mde import (
    DA3Predictor,
    DepthMetricAccumulator,
    da3_processed_geometry,
    depth_metrics,
)


def test_depth_metrics_are_exact_for_simple_arrays():
    target = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    result = depth_metrics(target * 2.0, target)

    assert result["abs_rel"] == 1.0
    assert result["sq_rel"] == 3.75
    assert result["mae"] == 3.75
    assert result["rmse"] == pytest.approx(np.sqrt(21.25))
    assert result["rmse_log"] == pytest.approx(np.log(2.0))
    assert result["silog_x100"] == pytest.approx(0.0, abs=1e-6)
    assert result["delta1"] == 0.0
    assert result["delta2"] == 0.0
    assert result["delta3"] == 0.0
    assert result["count"] == 4


def test_metric_accumulators_can_be_merged():
    target = np.array([1.0, 2.0, 4.0, 8.0])
    first = DepthMetricAccumulator()
    second = DepthMetricAccumulator()
    first.update(target[:2] * 1.1, target[:2])
    second.update(target[2:] * 1.1, target[2:])

    first.merge(second)
    expected = depth_metrics(target * 1.1, target)
    for name, value in expected.items():
        assert first.compute()["pixel_micro"][name] == pytest.approx(value)


def test_frame_macro_weights_each_frame_equally():
    metrics = DepthMetricAccumulator()
    metrics.update(np.array([2.0]), np.array([1.0]))
    metrics.update(np.ones(3), np.ones(3))

    result = metrics.compute()
    assert result["pixel_micro"]["abs_rel"] == pytest.approx(0.25)
    assert result["pixel_micro"]["count"] == 4
    assert result["frame_macro"]["abs_rel"] == pytest.approx(0.5)
    assert result["frame_macro"]["count"] == 2


def test_metrics_reject_invalid_values_on_fixed_support():
    target = np.ones((2, 2), dtype=np.float32)
    prediction = target.copy()
    prediction[0, 0] = 0
    with pytest.raises(ValueError, match="prediction must be finite and positive"):
        depth_metrics(prediction, target, np.ones_like(target, dtype=bool))


def test_da3_processed_geometry_matches_upper_bound_resize():
    intrinsics = np.array(
        [[1072.25, 0.0, 638.382], [0.0, 1072.15, 521.447], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    height, width, processed, focal_scale = da3_processed_geometry(
        (1024, 1280),
        intrinsics,
    )

    assert (height, width) == (406, 504)
    assert processed[0, 0] == pytest.approx(422.1984375)
    assert processed[1, 1] == pytest.approx(425.09072265625)
    assert focal_scale == pytest.approx(1.4121486002604168)


def test_da3_predictor_reuses_compatible_cache(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 3)).save(image_path)

    class Model:
        def __init__(self):
            self.calls = 0

        def inference(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(depth=np.full((1, 2, 3), 2.0, dtype=np.float32))

    model = Model()
    predictor = DA3Predictor(cache_root=tmp_path / "cache")
    predictor.model = model
    first = predictor.predict_raw(image_path, (2, 3), cache_id="scene/frame")
    second = predictor.predict_raw(image_path, (2, 3), cache_id="scene/frame")

    np.testing.assert_array_equal(first, second)
    assert model.calls == 1
    assert predictor.cache_namespace.name == "e7f29e2a962c7cd3ddf8"
    with np.load(predictor.cache_path("scene/frame"), allow_pickle=False) as item:
        metadata = json.loads(str(item["metadata"].item()))
    assert metadata["scene_id"] == "scene"
    assert metadata["frame_id"] == "frame"
    assert metadata["signature"] == predictor.cache_signature
