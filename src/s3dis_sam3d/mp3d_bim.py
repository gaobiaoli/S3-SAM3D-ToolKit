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
        default_mesh_source="obj",
        min_gt_valid_fraction=0.1,
        min_bim_hit_fraction=0.2,
        containment_margin=0.0,
    ):
        self.mp3d_dataset = Matterport3DDataset(mp3d_root)
        self.bimnet_dataset = BIMNetDataset(
            bimnet_root,
            split=split,
            default_mesh_source=default_mesh_source,
        )

        self.min_gt_valid_fraction = float(min_gt_valid_fraction)
        self.min_bim_hit_fraction = float(min_bim_hit_fraction)
        self.containment_margin = float(containment_margin)

        # Only indexed when a scene is actually requested.
        self._scene_samples = {}

    @property
    def scene_ids(self):
        """Return all available BIMNet scene IDs."""
        return tuple(scene.key for scene in self.bimnet_dataset.scenes)

    @property
    def default_mesh_source(self):
        return self.bimnet_dataset.default_mesh_source

    def _scenes(self, scene_id):
        bim_scene = self.bimnet_dataset.scene(scene_id)
        mp3d_scene = self.mp3d_dataset[bim_scene.matterport_scan_id]
        return bim_scene, mp3d_scene

    def frames(self, scene_id):
        """Return the source Matterport frames for one BIMNet scene."""
        return self._scenes(scene_id)[1].frames

    def _load_sample(self, bim_scene, mp3d_scene, frame, *, size=None, load_rgb=True):
        if not bim_scene.is_contained_in(frame, margin=self.containment_margin):
            return None

        gt_depth = np.asarray(frame.depth, dtype=np.float32)
        gt_valid_fraction = float((np.isfinite(gt_depth) & (gt_depth > 0)).mean())
        if gt_valid_fraction <= self.min_gt_valid_fraction:
            return None

        bim_depth = np.asarray(bim_scene.render_depth(frame, size=size), dtype=np.float32)
        expected_shape = gt_depth.shape if size is None else tuple(size)
        if bim_depth.shape != expected_shape:
            raise ValueError(
                f"BIM depth shape {bim_depth.shape} does not match "
                f"requested shape {expected_shape} for frame {frame.frame_id}"
            )

        bim_mask = np.isfinite(bim_depth) & (bim_depth > 0)
        bim_hit_fraction = float(bim_mask.mean())
        if bim_hit_fraction <= self.min_bim_hit_fraction:
            return None

        sample = {
            "frame": frame,
            "rgb_path": frame.rgb_path,
            "bim_depth": bim_depth,
            "gt_depth": gt_depth,
            "bim_mask": bim_mask,
            "intrinsics": frame.intrinsics,
            "bim_intrinsics": (
                frame.intrinsics if size is None else frame.intrinsics_for_size(size)
            ),
            "frame_id": frame.frame_id,
            "bim_scene_id": bim_scene.scene_id,
            "source_scene_id": mp3d_scene.scene_id,
            "mp3d_scene_id": mp3d_scene.scene_id,
            "gt_valid_fraction": gt_valid_fraction,
            "bim_hit_fraction": bim_hit_fraction,
        }
        if load_rgb:
            rgb = np.asarray(frame.rgb, dtype=np.float32)
            if rgb.shape[:2] != gt_depth.shape:
                raise ValueError(
                    f"RGB shape {rgb.shape[:2]} does not match GT depth shape {gt_depth.shape}"
                )
            sample["rgb"] = rgb
        return sample

    def _index_scene(self, scene_id):
        """Find valid frames for one BIMNet scene."""

        bim_scene, mp3d_scene = self._scenes(scene_id)

        if bim_scene.key in self._scene_samples:
            return self._scene_samples[bim_scene.key]

        samples = []
        for frame in mp3d_scene.frames:
            sample = self._load_sample(
                bim_scene,
                mp3d_scene,
                frame,
                load_rgb=False,
            )
            if sample is None:
                continue

            samples.append(
                {
                    "frame": frame,
                    "bim_scene_key": bim_scene.key,
                    "bim_scene_id": bim_scene.scene_id,
                    "mp3d_scene_id": mp3d_scene.scene_id,
                    "gt_valid_fraction": sample["gt_valid_fraction"],
                    "bim_hit_fraction": sample["bim_hit_fraction"],
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
        bim_scene = self.bimnet_dataset.scene(sample["bim_scene_key"])
        mp3d_scene = self.mp3d_dataset[sample["mp3d_scene_id"]]
        loaded = self._load_sample(bim_scene, mp3d_scene, frame)
        if loaded is None:
            raise RuntimeError(f"indexed frame is no longer valid: {frame.frame_id}")
        return loaded

    def iter_scene(self, scene_id, *, size=None, progress=False):
        """Stream valid samples, optionally rendering BIM depth at ``size``."""

        bim_scene, mp3d_scene = self._scenes(scene_id)
        frames = mp3d_scene.frames
        if progress:
            from tqdm.auto import tqdm

            frames = tqdm(frames, desc=f"Pairing {bim_scene.scene_id}")

        for frame in frames:
            sample = self._load_sample(bim_scene, mp3d_scene, frame, size=size)
            if sample is not None:
                yield sample

    def build_index(self, scene_ids=None):
        """
        Explicitly index several or all BIMNet scenes.

        Nothing is indexed automatically during __init__.
        """

        if scene_ids is None:
            scene_ids = [scene.key for scene in self.bimnet_dataset.scenes]

        for scene_id in scene_ids:
            self._index_scene(scene_id)

        return self

    @property
    def indexed_scene_ids(self):
        return tuple(self._scene_samples)

    def __repr__(self):
        sample_count = sum(len(samples) for samples in self._scene_samples.values())

        return (
            f"MP3D_BIMDataset("
            f"default_mesh_source={self.default_mesh_source!r}, "
            f"indexed_scenes={len(self._scene_samples)}, "
            f"indexed_samples={sample_count})"
        )
