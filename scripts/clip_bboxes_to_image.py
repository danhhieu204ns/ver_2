#!/usr/bin/env python3
"""Clip object detection bounding boxes to image boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def as_number(value: Any) -> float:
    return float(value)


def format_number(value: float) -> int | float:
    if value.is_integer():
        return int(value)
    return round(value, 6)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "annotation_file",
        "group",
        "image_id",
        "file_name",
        "object_index",
        "category",
        "image_width",
        "image_height",
        "old_bbox",
        "new_bbox",
        "old_area",
        "new_area",
        "area_delta",
        "action",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def backup_annotations(annotation_files: list[Path], backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for annotation_file in annotation_files:
        destination = backup_dir / annotation_file.parent.name / annotation_file.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(annotation_file, destination)


def process_annotation_file(annotation_file: Path, dry_run: bool) -> list[dict[str, Any]]:
    with annotation_file.open(encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(f"{annotation_file} must contain a JSON list")

    changes: list[dict[str, Any]] = []
    group = annotation_file.parent.name

    for record in records:
        width = as_number(record["width"])
        height = as_number(record["height"])
        objects = record.get("objects") or []
        if not isinstance(objects, list):
            continue

        for object_index, obj in enumerate(objects):
            if not isinstance(obj, dict):
                continue

            bbox = obj.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            x, y, bbox_width, bbox_height = [as_number(value) for value in bbox]
            old_x1 = x
            old_y1 = y
            old_x2 = x + bbox_width
            old_y2 = y + bbox_height

            new_x1 = clip(old_x1, 0.0, width)
            new_y1 = clip(old_y1, 0.0, height)
            new_x2 = clip(old_x2, 0.0, width)
            new_y2 = clip(old_y2, 0.0, height)
            new_width = new_x2 - new_x1
            new_height = new_y2 - new_y1

            outside = old_x1 < 0 or old_y1 < 0 or old_x2 > width or old_y2 > height
            if not outside:
                continue

            old_area = bbox_width * bbox_height
            new_area = max(0.0, new_width) * max(0.0, new_height)
            if new_width <= 0 or new_height <= 0:
                action = "drop_non_positive_after_clip"
            else:
                action = "clip"
                new_bbox = [
                    format_number(new_x1),
                    format_number(new_y1),
                    format_number(new_width),
                    format_number(new_height),
                ]
                if not dry_run:
                    obj["bbox"] = new_bbox

            changes.append(
                {
                    "annotation_file": str(annotation_file),
                    "group": group,
                    "image_id": record.get("image_id", ""),
                    "file_name": record.get("file_name", ""),
                    "object_index": object_index,
                    "category": obj.get("category", ""),
                    "image_width": format_number(width),
                    "image_height": format_number(height),
                    "old_bbox": json.dumps(bbox, ensure_ascii=False),
                    "new_bbox": json.dumps(
                        [
                            format_number(new_x1),
                            format_number(new_y1),
                            format_number(max(0.0, new_width)),
                            format_number(max(0.0, new_height)),
                        ],
                        ensure_ascii=False,
                    ),
                    "old_area": round(old_area, 6),
                    "new_area": round(new_area, 6),
                    "area_delta": round(new_area - old_area, 6),
                    "action": action,
                }
            )

    if changes and not dry_run:
        annotation_file.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=Path("data"), type=Path, help="Dataset root containing annotation files.")
    parser.add_argument(
        "--report",
        default=Path("results/dataset_statistics/bbox_clip_report.csv"),
        type=Path,
        help="CSV report path for changed boxes.",
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        type=Path,
        help="Directory for annotation backups. Defaults to a timestamped folder beside the report.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without modifying annotations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_files = sorted(args.data_root.glob("*/annotations.json"))
    if not annotation_files:
        raise FileNotFoundError(f"No annotations.json files found under {args.data_root}")

    backup_dir = args.backup_dir
    if backup_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_dir = args.report.parent / "annotation_backups" / timestamp

    if not args.dry_run:
        backup_annotations(annotation_files, backup_dir)

    changes: list[dict[str, Any]] = []
    for annotation_file in annotation_files:
        changes.extend(process_annotation_file(annotation_file, args.dry_run))

    write_csv(args.report, changes)
    summary = {
        "changed_bboxes": len(changes),
        "backup_dir": "" if args.dry_run else str(backup_dir),
        "report": str(args.report),
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
