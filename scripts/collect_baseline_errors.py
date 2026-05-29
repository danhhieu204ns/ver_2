#!/usr/bin/env python3
"""Collect qualitative baseline error cases from COCO-style predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from baseline_eval_utils import (
    SPLITS,
    bbox_xywh_to_xyxy,
    limit_records,
    load_records,
    sanitize_predictions,
    scale_bucket_for_object,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=Path("data"), type=Path, help="Dataset root containing group folders.")
    parser.add_argument("--split", default="test", choices=SPLITS, help="Dataset split to inspect.")
    parser.add_argument("--predictions", required=True, type=Path, help="COCO-style predictions JSON.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for error CSV and optional overlays.")
    parser.add_argument("--score-threshold", default=0.25, type=float, help="Prediction threshold used to define qualitative errors.")
    parser.add_argument("--iou-threshold", default=0.5, type=float, help="IoU threshold for a correct object-level match.")
    parser.add_argument("--low-iou-threshold", default=0.1, type=float, help="Lower IoU bound for mislocalization cases.")
    parser.add_argument("--max-images", default=None, type=int, help="Optional smoke-test image limit.")
    parser.add_argument("--draw-limit", default=100, type=int, help="Maximum number of overlay images to draw. Use 0 to disable.")
    return parser.parse_args()


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = box
    return x, y, x + width, y + height


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def object_boxes(record: dict[str, Any]) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for index, obj in enumerate(record["objects"]):
        if not isinstance(obj, dict):
            continue
        box = bbox_xywh_to_xyxy(obj.get("bbox"), record["width"], record["height"])
        if box is None:
            continue
        boxes.append({"index": index, "box": box, "scale": scale_bucket_for_object(obj, record)})
    return boxes


def draw_overlay(
    record: dict[str, Any],
    detections: list[dict[str, Any]],
    output_path: Path,
    threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(record["image_path"]) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        for gt in object_boxes(record):
            draw.rectangle(gt["box"], outline=(0, 180, 0), width=4)
            draw.text((gt["box"][0], max(0.0, gt["box"][1] - 12)), f"GT {gt['scale']}", fill=(0, 180, 0))
        for detection in detections:
            if float(detection["score"]) < threshold:
                continue
            box = xywh_to_xyxy(detection["bbox"])
            draw.rectangle(box, outline=(220, 0, 0), width=3)
            draw.text((box[0], max(0.0, box[1] - 12)), f"{float(detection['score']):.2f}", fill=(220, 0, 0))
        image.save(output_path)


def collect_errors(
    records: list[dict[str, Any]],
    coco_ids: list[int],
    predictions: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_image[int(prediction["image_id"])].append(prediction)
    for image_predictions in predictions_by_image.values():
        image_predictions.sort(key=lambda item: float(item["score"]), reverse=True)

    rows: list[dict[str, Any]] = []
    overlay_cache: dict[int, str] = {}

    def overlay_for(record: dict[str, Any], coco_id: int, detections: list[dict[str, Any]]) -> str:
        if args.draw_limit <= 0:
            return ""
        if coco_id in overlay_cache:
            return overlay_cache[coco_id]
        if len(overlay_cache) >= args.draw_limit:
            return ""
        overlay_path = args.output_dir / "overlays" / f"{coco_id:06d}_{record['group']}__{Path(record['file_name']).stem}.jpg"
        draw_overlay(record, detections, overlay_path, args.score_threshold)
        overlay_cache[coco_id] = str(overlay_path)
        return str(overlay_path)

    for record, coco_id in zip(records, coco_ids):
        gt_boxes = object_boxes(record)
        detections = [item for item in predictions_by_image.get(coco_id, []) if float(item["score"]) >= args.score_threshold]

        if not gt_boxes and detections:
            rows.append(
                {
                    "error_type": "in_domain_false_positive" if record["group"] == "negative_in_domain" else "out_domain_false_positive",
                    "group": record["group"],
                    "image_id": coco_id,
                    "source_image_id": record["image_id"],
                    "file_name": record["file_name"],
                    "image_path": str(record["image_path"]),
                    "overlay_path": overlay_for(record, coco_id, detections),
                    "score": max(float(item["score"]) for item in detections),
                    "max_iou": 0.0,
                    "object_scale": "",
                    "gt_count": 0,
                    "detection_count": len(detections),
                    "gt_boxes": "[]",
                    "detections": json.dumps(detections, ensure_ascii=False),
                }
            )
            continue

        if not gt_boxes:
            continue

        detection_boxes = [xywh_to_xyxy(item["bbox"]) for item in detections]
        for gt in gt_boxes:
            overlaps = [iou(gt["box"], detection_box) for detection_box in detection_boxes]
            max_iou = max(overlaps) if overlaps else 0.0
            if max_iou >= args.iou_threshold:
                continue
            error_type = "mislocalization" if max_iou >= args.low_iou_threshold else "tiny_miss"
            if gt["scale"] not in {"tiny", "small"} and error_type == "tiny_miss":
                error_type = "miss"
            rows.append(
                {
                    "error_type": error_type,
                    "group": record["group"],
                    "image_id": coco_id,
                    "source_image_id": record["image_id"],
                    "file_name": record["file_name"],
                    "image_path": str(record["image_path"]),
                    "overlay_path": overlay_for(record, coco_id, detections),
                    "score": max((float(item["score"]) for item in detections), default=0.0),
                    "max_iou": max_iou,
                    "object_scale": gt["scale"],
                    "gt_count": len(gt_boxes),
                    "detection_count": len(detections),
                    "gt_boxes": json.dumps([item["box"] for item in gt_boxes], ensure_ascii=False),
                    "detections": json.dumps(detections, ensure_ascii=False),
                }
            )

        for detection, detection_box in zip(detections, detection_boxes):
            max_iou = max((iou(gt["box"], detection_box) for gt in gt_boxes), default=0.0)
            if max_iou >= args.low_iou_threshold:
                continue
            rows.append(
                {
                    "error_type": "background_false_positive",
                    "group": record["group"],
                    "image_id": coco_id,
                    "source_image_id": record["image_id"],
                    "file_name": record["file_name"],
                    "image_path": str(record["image_path"]),
                    "overlay_path": overlay_for(record, coco_id, detections),
                    "score": float(detection["score"]),
                    "max_iou": max_iou,
                    "object_scale": "",
                    "gt_count": len(gt_boxes),
                    "detection_count": len(detections),
                    "gt_boxes": json.dumps([item["box"] for item in gt_boxes], ensure_ascii=False),
                    "detections": json.dumps([detection], ensure_ascii=False),
                }
            )

    rows.sort(key=lambda row: (row["error_type"], -float(row["score"]), row["group"], row["file_name"]))
    return rows


def main() -> None:
    args = parse_args()
    records = limit_records(load_records(args.data_root, args.split), args.max_images)
    coco_ids = list(range(1, len(records) + 1))
    with args.predictions.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{args.predictions} must contain a JSON list")
    predictions = sanitize_predictions(payload, coco_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_errors(records, coco_ids, predictions, args)

    csv_path = args.output_dir / f"errors_{args.split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "error_type",
            "group",
            "image_id",
            "source_image_id",
            "file_name",
            "image_path",
            "overlay_path",
            "score",
            "max_iou",
            "object_scale",
            "gt_count",
            "detection_count",
            "gt_boxes",
            "detections",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {error_type: sum(1 for row in rows if row["error_type"] == error_type) for error_type in sorted({row["error_type"] for row in rows})}
    summary = {"rows": len(rows), "counts": counts, "csv": str(csv_path)}
    (args.output_dir / f"errors_{args.split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
