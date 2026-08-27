"""Project-wide dataset paths. Edit this file when data locations change."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
BUNDLED_DATASET_ROOT = PROJECT_ROOT / "dataset"

# Full local datasets.
S23DIS_ROOT = Path(r"C:\Users\bgao491\pythonProject")
S3DIS_ROOT = Path(
    r"C:\Users\bgao491\Downloads\Stanford3dDataset_v1.2\Stanford3dDataset_v1.2"
)
BIMSYNC_ROOT = Path(r"C:\Users\bgao491\pythonProject\glb_scene_to_ifc\ifc\ifc")
BIMNET_ROOT = Path(r"C:\Users\bgao491\DepthEstimation\BIMNet_release")
BIMSYNC_CALIBRATION_ROOT = OUTPUT_ROOT / "ifc_to_s3dis"

# Small datasets bundled with the repository for demos and tests.
MINIMAL_S23DIS_ROOT = BUNDLED_DATASET_ROOT / "2d3ds"
MINIMAL_S3DIS_ROOT = BUNDLED_DATASET_ROOT / "s3dis"
MINIMAL_BIMSYNC_ROOT = BUNDLED_DATASET_ROOT / "bimsync"


def s23dis_area(area="Area_1", root=None):
    """Return one 2D-3D-S Area directory."""
    return Path(root or S23DIS_ROOT) / str(area).lower()


def bimsync_calibration_dir(area="Area_1"):
    """Return the directory containing one Area's saved BIMSync matrices."""
    return BIMSYNC_CALIBRATION_ROOT / str(area)
