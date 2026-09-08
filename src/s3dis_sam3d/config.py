from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------
# Project-local paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
BUNDLED_DATASET_ROOT = PROJECT_ROOT / "dataset"

MINIMAL_S23DIS_ROOT = BUNDLED_DATASET_ROOT / "2d3ds"
MINIMAL_S3DIS_ROOT = BUNDLED_DATASET_ROOT / "s3dis"
MINIMAL_BIMSYNC_ROOT = BUNDLED_DATASET_ROOT / "bimsync"

S23DIS_SEMANTIC_LABELS_PATH = BUNDLED_DATASET_ROOT / "semantic_labels.json"


# ---------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------

CONFIG_PATH = Path.home() / ".s3dis_sam3d.json"


@dataclass
class Config:
    s23dis_root: Optional[Path] = None
    s3dis_root: Optional[Path] = None
    bimsync_root: Optional[Path] = None
    bimnet_root: Optional[Path] = None
    matterport_root: Optional[Path] = None

    def require(self, name):
        """
        Return a configured path.

        Raises a clear error when a required dataset path has not been configured.
        """
        value = getattr(self, name)

        if value is None:
            raise RuntimeError(
                f"{name} is not configured.\n"
                f"Run s3dis_sam3d.configure({name}='...') first."
            )

        return value

    def save(self, path=CONFIG_PATH):
        """Save this config to disk."""
        path = Path(path).expanduser()

        data = {
            "s23dis_root": self.s23dis_root,
            "s3dis_root": self.s3dis_root,
            "bimsync_root": self.bimsync_root,
            "bimnet_root": self.bimnet_root,
            "matterport_root": self.matterport_root,
        }

        data = {
            key: str(value) if value is not None else None
            for key, value in data.items()
        }

        path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path=CONFIG_PATH):
        """Load config from disk. Return an empty config if none exists."""
        path = Path(path).expanduser()

        if not path.is_file():
            return cls()

        data = json.loads(path.read_text(encoding="utf-8"))

        for key, value in data.items():
            if value is not None:
                data[key] = Path(value).expanduser()

        return cls(**data)

    def __repr__(self):
        lines = ["Config("]

        for name in (
            "s23dis_root",
            "s3dis_root",
            "bimsync_root",
            "bimnet_root",
            "matterport_root",
        ):
            lines.append(f"    {name}={getattr(self, name)!r},")

        lines.append(")")
        return "\n".join(lines)


# Loaded once when the package is imported.
CONFIG = Config.load()


def configure(
    *,
    s23dis_root=None,
    s3dis_root=None,
    bimsync_root=None,
    bimnet_root=None,
    matterport_root=None,
):
    """
    Configure dataset paths and save them permanently for this user.

    Example
    -------
    configure(
        s23dis_root="/data/2d3ds",
        s3dis_root="/data/Stanford3dDataset_v1.2",
        bimnet_root="/data/BIMNet_release",
    )
    """
    values = {
        "s23dis_root": s23dis_root,
        "s3dis_root": s3dis_root,
        "bimsync_root": bimsync_root,
        "bimnet_root": bimnet_root,
        "matterport_root": matterport_root,
    }

    for name, value in values.items():
        if value is not None:
            setattr(
                CONFIG,
                name,
                Path(value).expanduser().resolve(),
            )

    CONFIG.save()

    return CONFIG


def s23dis_area(area="Area_1", root=None):
    """Return one 2D-3D-S Area directory."""
    if root is None:
        root = CONFIG.require("s23dis_root")

    root = Path(root).expanduser()
    area_name = str(area)
    area_names = tuple(dict.fromkeys((area_name, area_name.casefold())))

    # Accept either an Area directory or the common extracted dataset layouts:
    # ``root/area_1`` and ``root/no_xyz/area_1``.
    if (root / "data").is_dir():
        return root

    candidates = tuple(root / name for name in area_names) + tuple(
        root / "no_xyz" / name for name in area_names
    )
    return next(
        (candidate for candidate in candidates if candidate.is_dir()),
        root / area_name.casefold(),
    )


def bimsync_calibration_dir(area="Area_1"):
    """Return the saved BIMSync calibration directory for one Area."""
    return OUTPUT_ROOT / "ifc_to_s3dis" / str(area)