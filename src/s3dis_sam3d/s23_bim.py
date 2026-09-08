from __future__ import annotations

import numpy as np

from .bimsync import BIMSyncDataset
from .s23dis import S23Dataset, s23dis_area


class S23_BIMDataset:
    """Paired 2D-3D-S + BIMSync RGB/depth/mask data."""

    def __init__(
        self,
        area="Area_1",
        s23_root=None,
        bimsync_root=None,
        calibration_dir=None,
        min_gt_valid_fraction=0.1,
        min_bim_hit_fraction=0.2,
    ):
        self.area = str(area)

        if s23_root is None:
            self.s23_dataset = S23Dataset(
                area=self.area,
                projection_type="regular",
            )
        else:
            self.s23_dataset = S23Dataset(
                area_path=s23dis_area(
                    self.area,
                    root=s23_root,
                ),
                projection_type="regular",
            )

        self.bimsync_dataset = BIMSyncDataset(
            root=bimsync_root,
            area=self.area,
            calibration_dir=calibration_dir,
        )

        self.min_gt_valid_fraction = float(
            min_gt_valid_fraction
        )
        self.min_bim_hit_fraction = float(
            min_bim_hit_fraction
        )

        # A paired training scene must exist in both datasets and have a
        # loaded IFC-to-S3DIS calibration.
        matched_scenes = self.bimsync_dataset.matching_scenes(
            self.s23_dataset
        )

        if not matched_scenes:
            raise ValueError(
                f"no matching S23/BIMSync scenes found for {self.area!r}"
            )

        self._scenes = {
            scene.name.casefold(): scene
            for scene in matched_scenes
            if scene.is_calibrated
        }

        if not self._scenes:
            raise ValueError(
                f"no calibrated S23/BIMSync scenes found for {self.area!r}. "
                "Pass a calibration_dir containing "
                "*_ifc_to_s3dis_transform.npy files."
            )

        # Indexed only when requested.
        self._scene_samples = {}

    # ------------------------------------------------------------------
    # Scene access
    # ------------------------------------------------------------------

    @property
    def scene_ids(self):
        """Return all calibrated, paired S23/BIMSync scene IDs."""
        return tuple(
            scene.name
            for scene in self._scenes.values()
        )

    @property
    def indexed_scene_ids(self):
        """Return scenes that have already been indexed."""
        return tuple(self._scene_samples)

    def _resolve_scene(self, scene_id):
        """Resolve one matched BIMSync scene."""

        name = str(scene_id).replace("\\", "/").strip("/")
        key = name.rsplit("/", 1)[-1].removesuffix(".ifc").casefold()

        if key not in self._scenes:
            raise KeyError(
                f"unknown S23/BIM scene: {scene_id!r}"
            )

        return self._scenes[key]

    def _index_scene(self, scene_id):
        """Find valid frames for one BIMSync scene."""

        bim_scene = self._resolve_scene(scene_id)
        key = bim_scene.name

        if key in self._scene_samples:
            return self._scene_samples[key]

        s23_room = self.s23_dataset.room(
            f"{self.area}/{bim_scene.name}"
        )

        samples = []

        for frame in s23_room.frames:

            # ----------------------------------------------------------
            # 1. GT depth must exist
            # ----------------------------------------------------------

            if not frame.has_depth:
                continue

            gt_depth = np.asarray(
                frame.depth,
                dtype=np.float32,
            )

            gt_valid = (
                np.isfinite(gt_depth)
                & (gt_depth > 0)
            )

            gt_valid_fraction = float(
                gt_valid.mean()
            )

            if (
                gt_valid_fraction
                < self.min_gt_valid_fraction
            ):
                continue

            # ----------------------------------------------------------
            # 2. Render BIM depth
            # ----------------------------------------------------------

            bim_depth = np.asarray(
                bim_scene.render_depth(frame),
                dtype=np.float32,
            )

            if bim_depth.shape != gt_depth.shape:
                raise ValueError(
                    f"BIM depth shape {bim_depth.shape} does not match "
                    f"GT depth shape {gt_depth.shape} for {frame.stem}"
                )

            bim_mask = (
                np.isfinite(bim_depth)
                & (bim_depth > 0)
            )

            bim_hit_fraction = float(
                bim_mask.mean()
            )

            if (
                bim_hit_fraction
                < self.min_bim_hit_fraction
            ):
                continue

            # ----------------------------------------------------------
            # Valid sample
            # ----------------------------------------------------------

            samples.append(
                {
                    "frame": frame,
                    "bim_scene_key": bim_scene.key,
                    "bim_scene_id": bim_scene.name,
                    "s23_scene_id": s23_room.name,
                    "scene_id": bim_scene.name,
                    "room": s23_room.name,
                    "frame_id": frame.frame_id,
                    "uuid": frame.uuid,
                    "stem": frame.stem,
                    "gt_valid_fraction":
                        gt_valid_fraction,
                    "bim_hit_fraction":
                        bim_hit_fraction,
                }
            )

        self._scene_samples[key] = samples

        return samples

    def scene_samples(self, scene_id):
        """Return valid sample metadata for one BIMSync scene."""

        return self._index_scene(scene_id)

    def scene_size(self, scene_id):
        return len(self._index_scene(scene_id))

    def get(self, scene_id, index):
        """Load RGB, GT depth, BIM depth and BIM mask for one valid sample."""

        samples = self._index_scene(scene_id)
        sample = samples[index]

        frame = sample["frame"]
        bim_scene = self.bimsync_dataset.scene(
            sample["bim_scene_key"]
        )

        rgb = np.asarray(
            frame.rgb,
            dtype=np.float32,
        )

        gt_depth = np.asarray(
            frame.depth,
            dtype=np.float32,
        )

        bim_depth = np.asarray(
            bim_scene.render_depth(frame),
            dtype=np.float32,
        )

        if bim_depth.shape != gt_depth.shape:
            raise ValueError(
                f"BIM depth shape {bim_depth.shape} "
                f"does not match GT depth shape "
                f"{gt_depth.shape}"
            )

        if rgb.shape[:2] != gt_depth.shape:
            raise ValueError(
                f"RGB shape {rgb.shape[:2]} "
                f"does not match GT depth shape "
                f"{gt_depth.shape}"
            )

        bim_mask = (
            np.isfinite(bim_depth)
            & (bim_depth > 0)
        )

        return {
            "rgb": rgb,
            "bim_depth": bim_depth,
            "gt_depth": gt_depth,
            "bim_mask": bim_mask,

            "area": self.area,
            "bim_scene_id": sample["bim_scene_id"],
            "source_scene_id": sample["s23_scene_id"],
            "s23_scene_id": sample["s23_scene_id"],
            "scene_id": sample["scene_id"],
            "room": sample["room"],
            "frame_id": sample["frame_id"],
            "uuid": sample["uuid"],
            "stem": sample["stem"],

            "gt_valid_fraction":
                sample["gt_valid_fraction"],

            "bim_hit_fraction":
                sample["bim_hit_fraction"],
        }

    def iter_scene(self, scene_id):
        """Iterate valid samples from one scene."""

        samples = self._index_scene(scene_id)

        for index in range(len(samples)):
            yield self.get(
                scene_id,
                index,
            )

    def build_index(self, scene_ids=None):
        """
        Explicitly index several or all BIMSync scenes.

        Nothing is indexed automatically during __init__.
        """

        if scene_ids is None:
            scene_ids = self.scene_ids

        for scene_id in scene_ids:
            self._index_scene(scene_id)

        return self

    def __repr__(self):
        sample_count = sum(
            len(samples)
            for samples in self._scene_samples.values()
        )

        return (
            f"S23_BIMDataset("
            f"area={self.area!r}, "
            f"scenes={len(self.scene_ids)}, "
            f"indexed_scenes={len(self._scene_samples)}, "
            f"indexed_samples={sample_count})"
        )
