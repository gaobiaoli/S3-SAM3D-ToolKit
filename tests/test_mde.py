import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from s3dis_sam3d.mde import (
    MOGE2_MODEL,
    MOGE3_MODEL,
    MOGE_OUTPUT_SIZE,
    UNIDEPTHV2_MODEL,
    UNIDEPTHV2_OUTPUT_SIZE,
    DA3Predictor,
    DepthMetricAccumulator,
    MoGe2Predictor,
    MoGe3Predictor,
    UniDepthV2Predictor,
    da3_processed_geometry,
    depth_metrics,
    moge_fov_x,
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


def test_unidepthv2_predictor_uses_intrinsics_and_separate_caches(tmp_path):
    torch = pytest.importorskip("torch")
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(image_path)

    class Model:
        def __init__(self):
            self.calls = []

        def infer(self, rgb, camera):
            self.calls.append((rgb.clone(), None if camera is None else camera.clone()))
            value = 2.0 if camera is None else camera[0, 0].item() / 100.0
            return {"depth": torch.full((1, 1, 3, 4), value)}

    predictor = UniDepthV2Predictor(cache_root=tmp_path / "cache", output_size=56)
    predictor.model = Model()
    assert predictor.model_name == UNIDEPTHV2_MODEL
    intrinsics = np.array([[200, 0, 2], [0, 200, 1.5], [0, 0, 1]])
    frame = SimpleNamespace(
        rgb_path=image_path,
        image_shape=(3, 4),
        intrinsics=intrinsics,
        scene_id="scene",
        frame_id="frame",
    )

    first = predictor.predict_frame(frame)
    second = predictor.predict_frame(frame, focal_correct=False)
    np.testing.assert_array_equal(first, np.full((42, 56), 2, dtype=np.float32))
    np.testing.assert_array_equal(second, first)
    assert len(predictor.model.calls) == 1
    rgb, camera = predictor.model.calls[0]
    assert tuple(rgb.shape) == (3, 3, 4)
    assert rgb[:, 0, 0].tolist() == [10, 20, 30]
    np.testing.assert_array_equal(camera.numpy(), intrinsics.astype(np.float32))

    frame.intrinsics = intrinsics * np.array([[1.5, 1, 1], [1, 1, 1], [1, 1, 1]])
    changed_camera = predictor.predict_frame(frame)
    np.testing.assert_array_equal(changed_camera, np.full((42, 56), 3, dtype=np.float32))
    assert len(predictor.model.calls) == 2

    inferred_camera = predictor.predict_raw(image_path, cache_id="scene/frame")
    np.testing.assert_array_equal(inferred_camera, np.full((56, 56), 2, dtype=np.float32))
    assert predictor.model.calls[-1][1] is None
    assert len(predictor.model.calls) == 3
    assert len(list((tmp_path / "cache").rglob("*.npz"))) == 3


def test_unidepthv2_rejects_invalid_intrinsics_before_inference(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 3)).save(image_path)
    predictor = UniDepthV2Predictor()

    with pytest.raises(ValueError, match="shape"):
        predictor.predict_raw(image_path, intrinsics=np.eye(4))
    with pytest.raises(ValueError, match="positive focal"):
        predictor.predict_raw(image_path, intrinsics=np.diag([0, 1, 1]))
    with pytest.raises(ValueError, match="resolution_level"):
        UniDepthV2Predictor(resolution_level=10)
    with pytest.raises(ValueError, match="output_size"):
        UniDepthV2Predictor(output_size=0)


def test_unidepthv2_default_output_size_matches_da3_geometry(monkeypatch):
    predictor = UniDepthV2Predictor()
    assert UNIDEPTHV2_OUTPUT_SIZE == 504
    assert predictor._default_target_shape("unused") == (504, 504)

    frame = SimpleNamespace(
        rgb_path="unused.png",
        image_shape=(1024, 1280),
        intrinsics=np.array([[1000, 0, 640], [0, 1000, 512], [0, 0, 1]]),
        scene_id="scene",
        frame_id="frame",
    )
    monkeypatch.setattr(predictor, "predict_raw", lambda _path, shape, **_kwargs: shape)
    assert predictor.predict_frame(frame) == (406, 504)


def test_unidepthv2_loads_pinned_vitl_model(monkeypatch):
    calls = {}

    class Model:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls["name"] = name
            calls["kwargs"] = kwargs
            return cls()

        def to(self, device):
            calls["device"] = str(device)
            return self

        def eval(self):
            calls["eval"] = True
            return self

    unidepth = ModuleType("unidepth")
    models = ModuleType("unidepth.models")
    models.UniDepthV2 = Model
    unidepth.models = models
    monkeypatch.setitem(sys.modules, "unidepth", unidepth)
    monkeypatch.setitem(sys.modules, "unidepth.models", models)

    predictor = UniDepthV2Predictor(device="cpu", local_files_only=True, resolution_level=4)
    assert predictor._load() is predictor._load()
    assert calls["name"] == UNIDEPTHV2_MODEL
    assert calls["kwargs"] == {
        "revision": predictor.revision,
        "local_files_only": True,
    }
    assert calls["device"] == "cpu"
    assert calls["eval"] is True
    assert predictor.model.resolution_level == 4


def test_moge_fov_x_uses_pixel_space_focal_length():
    intrinsics = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    assert moge_fov_x((480, 640), intrinsics) == pytest.approx(
        np.degrees(2 * np.arctan(640 / 1000))
    )


@pytest.mark.parametrize(
    ("predictor_class", "model_name", "extra_kwargs"),
    [
        (MoGe3Predictor, MOGE3_MODEL, {"refine_steps": 2}),
        (MoGe2Predictor, MOGE2_MODEL, {}),
    ],
)
def test_moge_predictors_share_metric_interface(
    tmp_path, predictor_class, model_name, extra_kwargs
):
    torch = pytest.importorskip("torch")
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 3), color=(64, 128, 255)).save(image_path)

    class Model:
        def __init__(self):
            self.calls = []

        def infer(self, image, **kwargs):
            self.calls.append((image.clone(), kwargs))
            return {"depth": torch.full((3, 4), 2.0)}

    predictor = predictor_class(
        cache_root=tmp_path / "cache",
        output_size=56,
        resolution_level=7,
        num_tokens=1234,
        use_fp16=True,
        **extra_kwargs,
    )
    predictor.model = Model()
    assert predictor.model_name == model_name
    assert MOGE_OUTPUT_SIZE == 504
    intrinsics = np.array([[200, 0, 2], [0, 200, 1.5], [0, 0, 1]])
    frame = SimpleNamespace(
        rgb_path=image_path,
        image_shape=(3, 4),
        intrinsics=intrinsics,
        scene_id="scene",
        frame_id="frame",
    )

    first = predictor.predict_frame(frame)
    second = predictor.predict_frame(frame, focal_correct=False)
    np.testing.assert_array_equal(first, np.full((42, 56), 2, dtype=np.float32))
    np.testing.assert_array_equal(second, first)
    assert len(predictor.model.calls) == 1
    image, kwargs = predictor.model.calls[0]
    assert tuple(image.shape) == (3, 3, 4)
    torch.testing.assert_close(image[:, 0, 0], torch.tensor([64, 128, 255]) / 255)
    assert kwargs["resolution_level"] == 7
    assert kwargs["num_tokens"] == 1234
    assert kwargs["use_fp16"] is True
    assert kwargs["apply_mask"] is False
    assert kwargs["fov_x"] == pytest.approx(moge_fov_x((3, 4), intrinsics))
    if predictor_class is MoGe3Predictor:
        assert kwargs["refine_steps"] == 2
    else:
        assert "refine_steps" not in kwargs

    predictor.predict_raw(image_path, cache_id="scene/frame")
    assert predictor.model.calls[-1][1].get("fov_x") is None
    assert len(predictor.model.calls) == 2
    assert len(list((tmp_path / "cache").rglob("*.npz"))) == 2


def test_moge_predictor_validates_options():
    with pytest.raises(ValueError, match="output_size"):
        MoGe3Predictor(output_size=0)
    with pytest.raises(ValueError, match="resolution_level"):
        MoGe2Predictor(resolution_level=10)
    with pytest.raises(ValueError, match="num_tokens"):
        MoGe2Predictor(num_tokens=0)
    with pytest.raises(ValueError, match="refine_steps"):
        MoGe3Predictor(refine_steps=-1)


@pytest.mark.parametrize(
    ("predictor_class", "module_name", "model_name"),
    [
        (MoGe3Predictor, "moge.model.v3", MOGE3_MODEL),
        (MoGe2Predictor, "moge.model.v2", MOGE2_MODEL),
    ],
)
def test_moge_predictors_load_their_matching_versions(
    monkeypatch, predictor_class, module_name, model_name
):
    calls = {}

    class Model:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls["name"] = name
            calls["kwargs"] = kwargs
            return cls()

        def to(self, device):
            calls["device"] = str(device)
            return self

        def eval(self):
            return self

    moge = ModuleType("moge")
    model_package = ModuleType("moge.model")
    version_module = ModuleType(module_name)
    version_module.MoGeModel = Model
    moge.model = model_package
    monkeypatch.setitem(sys.modules, "moge", moge)
    monkeypatch.setitem(sys.modules, "moge.model", model_package)
    monkeypatch.setitem(sys.modules, module_name, version_module)

    predictor = predictor_class(device="cpu", local_files_only=True)
    assert predictor._load() is predictor._load()
    assert calls["name"] == model_name
    assert calls["kwargs"] == {
        "revision": predictor.revision,
        "local_files_only": True,
    }
    assert calls["device"] == "cpu"
