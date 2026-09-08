from __future__ import annotations

import numpy as np

from .bimnet import BIMNetDataset
from .matterport import Matterport3DDataset


class MP3D_BIMDataset:
    """Paired Matterport3D + BIMNet RGB/depth/mask data."""

    def __init__(
        self,
        mp3d_root=None,
        bimnet_root=None,
        split=None,
        min_gt_valid_fraction=0.1,
        min_bim_hit_fraction=0.2,
        containment_margin=0.0,
    ):
        self.mp3d_dataset = Matterport3DDataset(mp3d_root)
        self.bimnet_dataset = BIMNetDataset(
            bimnet_root,
            split=split,
        )

        self.min_gt_valid_fraction = float(
            min_gt_valid_fraction
        )
        self.min_bim_hit_fraction = float(
            min_bim_hit_fraction
        )
        self.containment_margin = float(
            containment_margin
        )

        # Only indexed when a scene is actually requested.
        self._scene_samples = {}

    @property
    def scene_ids(self):
        """Return all available BIMNet scene IDs."""
        return tuple(
            scene.key
            for scene in self.bimnet_dataset.scenes
        )

    def _index_scene(self, scene_id):
        """Find valid frames for one BIMNet scene."""

        bim_scene = self.bimnet_dataset.scene(scene_id)

        if bim_scene.key in self._scene_samples:
            return self._scene_samples[bim_scene.key]

        mp3d_scene = self.mp3d_dataset[
            bim_scene.matterport_scan_id
        ]

        samples = []

        for frame in mp3d_scene.frames:

            # Cheapest filter first.
            if not bim_scene.is_contained_in(
                frame,
                margin=self.containment_margin,
            ):
                continue

            gt_depth = np.asarray(
                frame.depth,
                dtype=np.float32,
            )

            gt_valid_fraction = float(
                (
                    np.isfinite(gt_depth)
                    & (gt_depth > 0)
                ).mean()
            )

            if (
                gt_valid_fraction
                < self.min_gt_valid_fraction
            ):
                continue

            bim_depth = np.asarray(
                bim_scene.render_depth(frame),
                dtype=np.float32,
            )

            if bim_depth.shape != gt_depth.shape:
                raise ValueError(
                    f"BIM depth shape {bim_depth.shape} does not match "
                    f"GT depth shape {gt_depth.shape} for frame {frame.frame_id}"
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

            samples.append(
                {
                    "frame": frame,
                    "bim_scene_key": bim_scene.key,
                    "bim_scene_id": bim_scene.scene_id,
                    "mp3d_scene_id": mp3d_scene.scene_id,
                    "gt_valid_fraction": gt_valid_fraction,
                    "bim_hit_fraction": bim_hit_fraction,
                }
            )

        self._scene_samples[bim_scene.key] = samples

        return samples

    def scene_samples(self, scene_id):
        """Return valid sample metadata for one BIMNet scene."""

        return self._index_scene(scene_id)

    def scene_size(self, scene_id):
        return len(self._index_scene(scene_id))

    def get(self, scene_id, index):
        """Load RGB, GT depth, BIM depth and BIM mask for one valid sample."""

        samples = self._index_scene(scene_id)
        sample = samples[index]

        frame = sample["frame"]
        bim_scene = self.bimnet_dataset.scene(
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
                f"BIM depth shape {bim_depth.shape} does not match "
                f"GT depth shape {gt_depth.shape}"
            )

        if rgb.shape[:2] != gt_depth.shape:
            raise ValueError(
                f"RGB shape {rgb.shape[:2]} does not match "
                f"GT depth shape {gt_depth.shape}"
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

            "frame_id": frame.frame_id,
            "bim_scene_id": sample["bim_scene_id"],
            "source_scene_id": sample["mp3d_scene_id"],
            "mp3d_scene_id": sample["mp3d_scene_id"],

            "gt_valid_fraction":
                sample["gt_valid_fraction"],

            "bim_hit_fraction":
                sample["bim_hit_fraction"],
        }

    def iter_scene(self, scene_id):
        """Iterate all valid samples from one BIMNet scene."""

        samples = self._index_scene(scene_id)

        for index in range(len(samples)):
            yield self.get(scene_id, index)

    def build_index(self, scene_ids=None):
        """
        Explicitly index several or all BIMNet scenes.

        Nothing is indexed automatically during __init__.
        """

        if scene_ids is None:
            scene_ids = [
                scene.key
                for scene in self.bimnet_dataset.scenes
            ]

        for scene_id in scene_ids:
            self._index_scene(scene_id)

        return self

    @property
    def indexed_scene_ids(self):
        return tuple(self._scene_samples)

    def __repr__(self):
        sample_count = sum(
            len(samples)
            for samples in self._scene_samples.values()
        )

        return (
            f"MP3D_BIMDataset("
            f"indexed_scenes={len(self._scene_samples)}, "
            f"indexed_samples={sample_count})"
        )
