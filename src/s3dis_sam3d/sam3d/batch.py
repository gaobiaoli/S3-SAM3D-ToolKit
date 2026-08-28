from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm


class SAM3DBatchPredictor:
    """Run SAM3D for filtered room-image annotations with resumable outputs."""

    def __init__(self, client, dataset, annotations):
        self.client = client
        self.dataset = dataset
        self.annotations = annotations

    @staticmethod
    def _result_paths(result):
        return {
            "glb_path": str(result.glb_path),
            "pose_path": None if result.pose_path is None else str(result.pose_path),
            "mask_path": None if result.mask_path is None else str(result.mask_path),
            "optimized_pose_path": (
                None
                if result.optimized_pose_path is None
                else str(result.optimized_pose_path)
            ),
        }

    @staticmethod
    def _missing_outputs(result, return_mask, optimize_pose):
        if result is None:
            return ["result"]
        missing = []
        if not result.glb_path.is_file() or result.glb_path.stat().st_size == 0:
            missing.append("glb")
        if result.pose_path is None or not result.pose_path.is_file():
            missing.append("pose")
        if return_mask and (result.mask_path is None or not result.mask_path.is_file()):
            missing.append("mask")
        if optimize_pose and (
            result.optimized_pose_path is None
            or not result.optimized_pose_path.is_file()
        ):
            missing.append("optimized_pose")
        return missing

    @staticmethod
    def _write_summary(path, summary):
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            "utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _as_list(value):
        if value is None:
            return None
        return [value] if isinstance(value, str) else list(value)

    def predict(
        self,
        output_dir,
        *,
        include_classes=None,
        exclude_classes=("clutter",),
        instances=None,
        min_bbox_iou=0.5,
        max_occlusion_ratio=0.5,
        min_pixels=None,
        skip_wrapped_pano=True,
        use_depth=True,
        return_mask=True,
        optimize_pose=True,
        optimize_iterations=300,
        seed=42,
        cache_dir=None,
        reuse_outputs=True,
        start_record=0,
        max_records=None,
        max_predictions=None,
        continue_on_error=True,
        progress=True,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "batch_predict_summary.json"
        cache_dir = output_dir if cache_dir is None and reuse_outputs else cache_dir
        excluded = {
            str(value).strip().casefold()
            for value in self._as_list(exclude_classes) or []
        }
        records = self.annotations.records[start_record:]
        if max_records is not None:
            records = records[:max_records]

        summary = {
            "annotations_dir": str(self.annotations.annotations_dir),
            "output_dir": str(output_dir),
            "summary_path": str(summary_path),
            "records": len(records),
            "annotations_seen": 0,
            "selected": 0,
            "success": 0,
            "cached": 0,
            "incomplete": 0,
            "failed": 0,
            "settings": {
                "include_classes": self._as_list(include_classes),
                "exclude_classes": self._as_list(exclude_classes) or [],
                "instances": self._as_list(instances),
                "min_bbox_iou": min_bbox_iou,
                "max_occlusion_ratio": max_occlusion_ratio,
                "min_pixels": min_pixels,
                "use_depth": use_depth,
                "return_mask": return_mask,
                "optimize_pose": optimize_pose,
                "seed": seed,
            },
            "results": [],
        }
        stop = False
        for record in tqdm(records, desc="SAM3D batch", disable=not progress):
            raw_annotations = record.get("annotations") or []
            summary["annotations_seen"] += len(raw_annotations)
            selected = self.annotations.filter_annotations(
                record,
                class_filter=include_classes,
                instance_filter=instances,
                min_bbox_iou=min_bbox_iou,
                max_occlusion_ratio=max_occlusion_ratio,
                min_pixels=min_pixels,
            )
            selected = [
                annotation
                for annotation in selected
                if str(annotation.get("class_name", "")).strip().casefold() not in excluded
                and not (
                    skip_wrapped_pano
                    and annotation.get("wraps_horizontal", False)
                )
            ]
            if not selected:
                continue

            raw_image_path = str(record.get("image_path") or "frame.png")
            raw_image_path = raw_image_path.replace("\\", "/")
            image_path = Path(raw_image_path)
            room = record["room"]
            frame_id = int(record["frame_id"])
            uuid = record.get("uuid")
            try:
                frame = self.dataset.room(room).get_frame(frame_id, uuid)
                image_path = frame.rgb_path
                needs_geometry = use_depth or optimize_pose
                if needs_geometry and not frame.has_depth:
                    raise FileNotFoundError(f"depth is unavailable for {frame.stem}")
                depth_path = frame.depth_path if needs_geometry else None
                intrinsics = (
                    frame.intrinsics
                    if needs_geometry and frame.projection_type == "regular"
                    else None
                )
                camera_transform = (
                    frame.camera_to_world
                    if optimize_pose
                    else None
                )
            except Exception as error:
                for annotation in selected:
                    if max_predictions is not None and summary["selected"] >= max_predictions:
                        stop = True
                        break
                    request_id = f"{Path(image_path).stem}_{annotation.get('instance_name')}"
                    summary["selected"] += 1
                    summary["failed"] += 1
                    summary["results"].append(
                        {
                            "request_id": request_id,
                            "status": "failed",
                            "error": str(error),
                        }
                    )
                self._write_summary(summary_path, summary)
                if not continue_on_error:
                    raise
                if stop:
                    break
                continue

            for annotation in selected:
                if max_predictions is not None and summary["selected"] >= max_predictions:
                    stop = True
                    break
                instance_name = str(annotation["instance_name"])
                request_id = f"{Path(image_path).stem}_{instance_name}"
                summary["selected"] += 1
                try:
                    result = self.client.infer(
                        image_path,
                        request_id=request_id,
                        output_dir=output_dir,
                        bbox=annotation["bbox_xyxy"],
                        depth_path=depth_path,
                        use_depth=use_depth,
                        intrinsics=intrinsics,
                        camera_transform=camera_transform,
                        seed=seed,
                        return_mask=return_mask,
                        optimize_pose=optimize_pose,
                        optimize_iterations=optimize_iterations,
                        cache_dir=cache_dir,
                    )
                    missing = self._missing_outputs(result, return_mask, optimize_pose)
                    if missing:
                        status = "incomplete"
                        entry = {
                            "request_id": request_id,
                            "status": status,
                            "missing": missing,
                        }
                        if result is not None:
                            entry.update(self._result_paths(result))
                    else:
                        status = "cached" if result.cached else "success"
                        entry = {
                            "request_id": result.request_id,
                            "requested_id": request_id,
                            "status": status,
                            "room": room,
                            "frame_id": frame_id,
                            "uuid": uuid,
                            "instance_name": instance_name,
                            **self._result_paths(result),
                        }
                    summary[status] += 1
                    summary["results"].append(entry)
                except Exception as error:
                    summary["failed"] += 1
                    summary["results"].append(
                        {
                            "request_id": request_id,
                            "status": "failed",
                            "error": str(error),
                        }
                    )
                    if not continue_on_error:
                        self._write_summary(summary_path, summary)
                        raise
                self._write_summary(summary_path, summary)
            if stop:
                break

        self._write_summary(summary_path, summary)
        return summary
