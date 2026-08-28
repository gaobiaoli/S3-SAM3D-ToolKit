from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from s3dis_sam3d import FrameRender

plt.switch_backend("Agg")


def _render():
    source_depth = np.array([[0, 1], [2, 4]], dtype=np.float32)
    rendered_depth = np.array([[0, 2], [4, 8]], dtype=np.float32)
    return FrameRender(
        rendered_image_path=None,
        rendered_depth_path=None,
        source_image_path=Path("source.png"),
        source_depth_path=Path("source_depth.png"),
        source_image=np.zeros((2, 2, 3), dtype=np.float32),
        source_depth=source_depth,
        rendered_image=np.ones((2, 2, 3), dtype=np.float32),
        rendered_depth=rendered_depth,
    )


def test_depth_limits_are_jointly_computed_from_both_depth_maps():
    render = _render()

    assert render.depth_limits() == (1.0, 8.0)
    assert render.depth_limits((0.5, 10)) == (0.5, 10.0)
    assert render.depth_limits(percentiles=(0, 100)) == (1.0, 8.0)


def test_depth_mode_uses_one_normalization_for_both_heatmaps(tmp_path):
    render = _render()
    output = tmp_path / "comparison.png"

    figure = render.visualize(mode="depth", show=False, save_path=output)
    source_heatmap = figure.axes[0].images[0]
    rendered_heatmap = figure.axes[1].images[0]

    assert output.is_file()
    assert source_heatmap.norm is rendered_heatmap.norm
    assert (source_heatmap.norm.vmin, source_heatmap.norm.vmax) == (1.0, 8.0)
    assert len(figure.axes) == 3  # Two panels plus one shared colorbar.
    plt.close(figure)


def test_rgb_mode_only_visualizes_two_rgb_images():
    render = _render()

    figure = render.visualize(mode="rgb", show=False)

    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == ["Source RGB", "Rendered RGB"]
    assert all(len(axis.images) == 1 for axis in figure.axes)
    plt.close(figure)


def test_visualize_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        _render().visualize(mode="normal", show=False)
