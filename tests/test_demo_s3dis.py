import importlib.util
from pathlib import Path

from s3dis_sam3d import S3DISDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = PROJECT_ROOT / "demo" / "s3dis_visualization.py"
SPEC = importlib.util.spec_from_file_location("s3dis_visualization_demo", DEMO_PATH)
assert SPEC is not None and SPEC.loader is not None
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


def test_s3dis_visualization_demo_prepares_open3d_input():
    cloud = S3DISDataset(DEMO.DATASET_ROOT).room(
        f"{DEMO.DEFAULT_AREA}/{DEMO.DEFAULT_ROOM}"
    ).point_cloud(
        color_mode="semantic",
        exclude_classes=["clutter", "ceiling"],
        exclude_instances=["wall_3", "wall_4"],
        ignore_missing_instances=True,
        max_points=1000,
    )
    assert len(cloud.xyz) == 1000
    assert cloud.rgb.shape == (1000, 3)
