from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from .pointcloud import random_downsample
from .s3dis import S3DISDataset
from .s23dis import S23Frame, S23Dataset


class RoomImageAnnotationsParser:
    """Read, query, filter, and visualize per-frame annotation JSON files."""

    def __init__(self, annotations_dir, image_root=None):
        self.annotations_dir = Path(annotations_dir).expanduser().resolve()
        if not self.annotations_dir.is_dir():
            raise FileNotFoundError(f"annotations directory not found: {self.annotations_dir}")
        self.image_root = (
            None if image_root is None else Path(image_root).expanduser().resolve()
        )
        self.annotation_files = sorted(self.annotations_dir.glob("*_ann.json"))
        self.records = [json.loads(path.read_text("utf-8")) for path in self.annotation_files]
        self._build_indexes()

    @staticmethod
    def _add_index(index, key, value):
        if key is not None:
            matches = index.setdefault(key, [])
            if value not in matches:
                matches.append(value)

    @staticmethod
    def _path_key(path):
        path = str(path).replace("\\", os.sep).replace("/", os.sep)
        return os.path.normcase(os.path.abspath(path))

    def _build_indexes(self):
        self._by_image = {}
        self._by_basename = {}
        self._by_uuid_frame = {}
        self._by_room_frame = {}
        for index, record in enumerate(self.records):
            image_path = record.get("image_path")
            if image_path:
                self._add_index(self._by_image, self._path_key(image_path), index)
                resolved = self.resolve_image_path(record)
                self._add_index(self._by_image, self._path_key(resolved), index)
                basename = Path(str(image_path).replace("\\", "/")).name.casefold()
                self._add_index(self._by_basename, basename, index)

            uuid = record.get("uuid")
            frame_id = record.get("frame_id")
            if uuid is not None and frame_id is not None:
                self._add_index(
                    self._by_uuid_frame,
                    (str(uuid), int(frame_id)),
                    index,
                )
            if frame_id is not None:
                rooms = {record.get("room"), record.get("s3dis_room")}
                for room in rooms:
                    if room is not None:
                        self._add_index(
                            self._by_room_frame,
                            (str(room).replace("\\", "/"), int(frame_id)),
                            index,
                        )

    @staticmethod
    def _unique(index, key, description):
        matches = index.get(key, [])
        if not matches:
            raise ValueError(f"record not found by {description}")
        if len(matches) > 1:
            raise ValueError(f"multiple records found by {description}; use a more specific key")
        return matches[0]

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def list_records(self):
        return [
            (
                index,
                record.get("room"),
                int(record.get("frame_id", -1)),
                record.get("uuid"),
                len(record.get("annotations") or []),
            )
            for index, record in enumerate(self.records)
        ]

    def get_record(self, index):
        return self.records[index]

    def get_index_by_image_path(self, image_path, allow_basename=True):
        key = self._path_key(image_path)
        if key in self._by_image:
            return self._unique(self._by_image, key, f"image path: {image_path}")
        if allow_basename:
            basename = Path(str(image_path).replace("\\", "/")).name.casefold()
            return self._unique(
                self._by_basename,
                basename,
                f"image basename: {basename}",
            )
        raise ValueError(f"record not found by image path: {image_path}")

    def get_record_by_image_path(self, image_path, allow_basename=True):
        return self.get_record(self.get_index_by_image_path(image_path, allow_basename))

    def get_index_by_uuid_frame(self, uuid, frame_id, room=None):
        key = (str(uuid), int(frame_id))
        matches = self._by_uuid_frame.get(key, [])
        if room is not None:
            query = str(room).replace("\\", "/")
            matches = [
                index
                for index in matches
                if query
                in {
                    self.records[index].get("room"),
                    self.records[index].get("s3dis_room"),
                }
            ]
        return self._unique({key: matches}, key, f"uuid/frame: {key}")

    def get_record_by_uuid_frame(self, uuid, frame_id, room=None):
        return self.get_record(self.get_index_by_uuid_frame(uuid, frame_id, room))

    def get_index_by_room_frame(self, room, frame_id):
        key = (str(room).replace("\\", "/"), int(frame_id))
        return self._unique(self._by_room_frame, key, f"room/frame: {key}")

    def get_record_by_room_frame(self, room, frame_id):
        return self.get_record(self.get_index_by_room_frame(room, frame_id))

    def get_record_flexible(
        self,
        *,
        index=None,
        idx=None,
        image_path=None,
        uuid=None,
        frame_id=None,
        room=None,
    ):
        index = index if index is not None else idx
        if index is not None:
            return self.get_record(int(index))
        if image_path is not None:
            return self.get_record_by_image_path(image_path)
        if uuid is not None and frame_id is not None:
            return self.get_record_by_uuid_frame(uuid, frame_id, room)
        if room is not None and frame_id is not None:
            return self.get_record_by_room_frame(room, frame_id)
        raise ValueError("provide index, image_path, uuid+frame_id, or room+frame_id")

    def resolve_image_path(self, record_or_index):
        record = (
            self.get_record(record_or_index)
            if isinstance(record_or_index, int)
            else record_or_index
        )
        value = record.get("image_path")
        if not value:
            raise ValueError("record has no image_path")
        normalized = Path(str(value).replace("\\", os.sep).replace("/", os.sep))
        if normalized.is_absolute():
            return normalized
        candidates = []
        if self.image_root is not None:
            candidates.append(self.image_root / normalized)
        candidates.extend((Path.cwd() / normalized, self.annotations_dir / normalized))
        return next((path.resolve() for path in candidates if path.is_file()), candidates[0])

    def read_image(self, index, as_rgb=True):
        with Image.open(self.resolve_image_path(index)) as source:
            image = np.asarray(source.convert("RGB"))
        return image if as_rgb else image[:, :, ::-1].copy()

    @staticmethod
    def _filter_values(values):
        if not values:
            return None
        if isinstance(values, str):
            values = [values]
        return {str(value).strip().casefold() for value in values}

    @staticmethod
    def filter_annotations(
        record,
        *,
        class_filter=None,
        instance_filter=None,
        min_bbox_iou=None,
        max_occlusion_ratio=None,
        min_pixels=None,
    ):
        classes = RoomImageAnnotationsParser._filter_values(class_filter)
        instances = RoomImageAnnotationsParser._filter_values(instance_filter)
        result = []
        for annotation in record.get("annotations") or []:
            class_name = str(annotation.get("class_name", "")).strip().casefold()
            instance_name = str(annotation.get("instance_name", "")).strip().casefold()
            if classes is not None and class_name not in classes:
                continue
            if instances is not None and instance_name not in instances:
                continue
            bbox = annotation.get("bbox_xyxy")
            if bbox is None or len(bbox) != 4:
                continue
            bbox_array = np.asarray(bbox, dtype=np.float64)
            if not np.isfinite(bbox_array).all() or np.any(bbox_array[2:] < bbox_array[:2]):
                continue
            if min_bbox_iou is not None:
                iou = annotation.get("bbox_iou_with_uncropped")
                try:
                    iou = float(iou)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(iou) or iou < min_bbox_iou:
                    continue
            if max_occlusion_ratio is not None:
                occlusion = annotation.get("occlusion_ratio")
                try:
                    occlusion = float(occlusion)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(occlusion) or occlusion > max_occlusion_ratio:
                    continue
            if min_pixels is not None and int(annotation.get("pixel_count", 0)) < min_pixels:
                continue
            result.append(annotation)
        return result

    def iter_annotations(self, **filters):
        for index, record in enumerate(self.records):
            for annotation in self.filter_annotations(record, **filters):
                yield index, record, annotation

    def visualize_bboxes_on_image(
        self,
        index,
        *,
        class_filter=None,
        instance_filter=None,
        draw_label=True,
        color=(255, 0, 0),
        thickness=2,
        show=False,
        save_path=None,
    ):
        record = self.get_record(index)
        with Image.open(self.resolve_image_path(record)) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        annotations = self.filter_annotations(
            record,
            class_filter=class_filter,
            instance_filter=instance_filter,
        )
        for annotation in annotations:
            bbox = tuple(int(value) for value in annotation["bbox_xyxy"])
            draw.rectangle(bbox, outline=color, width=thickness)
            if draw_label:
                label = annotation.get("instance_name") or annotation.get("class_name", "object")
                draw.text((bbox[0], max(0, bbox[1] - 12)), str(label), fill=color)
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(save_path)
        if show:
            image.show()
        return {
            "index": index,
            "room": record.get("room"),
            "frame_id": record.get("frame_id"),
            "uuid": record.get("uuid"),
            "drawn_boxes": len(annotations),
            "save_path": None if save_path is None else str(save_path),
            "image_rgb": np.asarray(image),
        }


@dataclass(frozen=True)
class FrameProjection:
    frame: S23Frame
    observed_depth: np.ndarray | None


def bbox_iou(box_a, box_b):
    """IoU for inclusive ``xyxy`` pixel boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1) + 1)
    height = max(0.0, min(ay2, by2) - max(ay1, by1) + 1)
    intersection = width * height
    area_a = max(0.0, ax2 - ax1 + 1) * max(0.0, ay2 - ay1 + 1)
    area_b = max(0.0, bx2 - bx1 + 1) * max(0.0, by2 - by1 + 1)
    union = area_a + area_b - intersection
    return 0.0 if union == 0 else float(intersection / union)


def prepare_frame_projection(frame, occlusion_check=True):
    """Load frame geometry once for all instances projected into that frame."""
    return FrameProjection(
        frame,
        frame.depth if occlusion_check and frame.has_depth else None,
    )


def project_instance(
    context,
    points_world,
    *,
    min_pixels=20,
    depth_tolerance=0.05,
):
    """Project one S3DIS instance and return its bbox/visibility annotation."""
    frame = context.frame
    pixels, point_depth = frame.project_world_points(points_world)
    if not len(pixels):
        return None

    uncropped = (
        float(pixels[:, 0].min()),
        float(pixels[:, 1].min()),
        float(pixels[:, 0].max()),
        float(pixels[:, 1].max()),
    )
    height, width = frame.image_shape
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    if not inside.any():
        return None

    pixels = pixels[inside]
    point_depth = point_depth[inside]
    x = np.clip(np.rint(pixels[:, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint(pixels[:, 1]).astype(np.int64), 0, height - 1)
    flat = y * width + x

    order = np.argsort(point_depth)
    unique_flat, nearest = np.unique(flat[order], return_index=True)
    nearest_depth = point_depth[order][nearest]
    if len(unique_flat) < min_pixels:
        return None

    unique_x = unique_flat % width
    unique_y = unique_flat // width
    bbox = (
        int(unique_x.min()),
        int(unique_y.min()),
        int(unique_x.max()),
        int(unique_y.max()),
    )

    occlusion_ratio = None
    visible_ratio = None
    stats = {
        "visible_pixels": 0,
        "occluded_pixels": 0,
        "unknown_depth_pixels": 0,
        "compared_pixels": 0,
        "depth_tolerance": float(depth_tolerance),
    }
    if context.observed_depth is not None:
        observed = context.observed_depth.reshape(-1)[unique_flat]
        valid = np.isfinite(observed) & (observed > 1e-6)
        visible = valid & (nearest_depth <= observed + depth_tolerance)
        occluded = valid & ~visible
        visible_count = int(visible.sum())
        occluded_count = int(occluded.sum())
        compared = visible_count + occluded_count
        stats.update(
            visible_pixels=visible_count,
            occluded_pixels=occluded_count,
            unknown_depth_pixels=int((~valid).sum()),
            compared_pixels=compared,
        )
        if compared:
            occlusion_ratio = occluded_count / compared
            visible_ratio = visible_count / compared

    wraps = frame.projection_type == "pano" and float(np.ptp(pixels[:, 0])) > width / 2
    return {
        "bbox_xyxy": list(bbox),
        "bbox_uncropped_xyxy": list(uncropped),
        "bbox_iou_with_uncropped": bbox_iou(bbox, uncropped),
        "pixel_count": len(unique_flat),
        "occlusion_ratio": occlusion_ratio,
        "visible_ratio": visible_ratio,
        "occlusion_stats": stats,
        "wraps_horizontal": bool(wraps),
    }


class RoomImageBatchAnnotator:
    """Project S3DIS room instances into all matching 2D-3D-S frames."""

    def __init__(self, s3dis_root, s23_area_path, projection_type="regular", seed=42):
        self.s3dis = S3DISDataset(s3dis_root)
        self.s23 = S23Dataset(s23_area_path, projection_type=projection_type)
        self.seed = seed

    def _instances(self, room, selected_classes, max_instance_points):
        selected = None if not selected_classes else {
            str(value).strip().casefold() for value in selected_classes
        }
        clouds = [
            instance.point_cloud
            for instance in room.instances
            if selected is None
            or instance.class_name in selected
            or str(instance.class_id) in selected
        ]
        if max_instance_points is not None:
            clouds = [
                random_downsample(cloud, max_instance_points, seed=self.seed)
                for cloud in clouds
            ]
        return clouds

    def annotate_room_images(
        self,
        room_name,
        selected_classes=None,
        output_dir="outputs/room_image_annotations",
        min_pixels=20,
        max_instance_points=200_000,
        save_empty=True,
        occlusion_check=True,
        depth_tolerance=0.05,
        max_frames=None,
        progress=True,
    ):
        room = self.s3dis.room(room_name)
        frames = list(self.s23.room(room.name).frames)
        if max_frames is not None:
            frames = frames[:max_frames]
        instances = self._instances(room, selected_classes, max_instance_points)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = []
        annotation_count = 0
        for frame in tqdm(frames, desc=f"Annotating {room.name}", disable=not progress):
            context = prepare_frame_projection(frame, occlusion_check)
            annotations = []
            for instance in instances:
                annotation = project_instance(
                    context,
                    instance.xyz,
                    min_pixels=min_pixels,
                    depth_tolerance=depth_tolerance,
                )
                if annotation is None:
                    continue
                annotation = {
                    "instance_name": instance.metadata["name"],
                    "class_name": instance.metadata["class_name"],
                    **annotation,
                }
                annotations.append(annotation)

            if not annotations and not save_empty:
                continue
            record = {
                "schema_version": 1,
                "bbox_format": "xyxy_inclusive",
                "bbox_semantics": "amodal_3d_projection",
                "image_path": str(frame.rgb_path),
                "room": room.name,
                "s3dis_room": room.key,
                "frame_id": int(frame.frame_id),
                "uuid": frame.uuid,
                "annotations": annotations,
            }
            output_path = output_dir / f"{frame.stem}_ann.json"
            temporary_path = output_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            temporary_path.replace(output_path)
            files.append(str(output_path))
            annotation_count += len(annotations)

        return {
            "room": room.name,
            "s3dis_room": room.key,
            "instances_used": len(instances),
            "images_saved": len(files),
            "total_annotations": annotation_count,
            "output_dir": str(output_dir),
            "files": files,
        }
