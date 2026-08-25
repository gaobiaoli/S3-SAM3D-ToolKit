import numpy as np

from s3dis_sam3d.io import read_xyzrgb_txt


def test_read_xyzrgb_txt_fast_path(tmp_path):
    path = tmp_path / "points.txt"
    path.write_text("0 1 2 10 20 30\n3 4 5 40 50 60\n", encoding="utf-8")

    xyz, rgb = read_xyzrgb_txt(path)

    np.testing.assert_array_equal(xyz, [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(rgb, [[10, 20, 30], [40, 50, 60]])
    assert xyz.dtype == np.float32


def test_read_xyzrgb_txt_falls_back_for_malformed_rows(tmp_path):
    path = tmp_path / "dirty_points.txt"
    path.write_text(
        "0 1 2 10 20 30\n"
        "invalid row\n"
        "3 4 5 40 50 60\x00\n",
        encoding="utf-8",
    )

    xyz, rgb = read_xyzrgb_txt(path)

    np.testing.assert_array_equal(xyz, [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(rgb, [[10, 20, 30], [40, 50, 60]])
