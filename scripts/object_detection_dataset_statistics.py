#!/usr/bin/env python3
"""Compute object detection dataset statistics from annotation JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {k: None for k in ["min", "p25", "median", "mean", "p75", "max"]}

    ordered = sorted(values)

    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * p
        lower = math.floor(pos)
        upper = math.ceil(pos)
        if lower == upper:
            return ordered[int(pos)]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)

    return {
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": median(ordered),
        "mean": mean(ordered),
        "p75": percentile(0.75),
        "max": ordered[-1],
    }


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def summarize_numeric(values: list[float]) -> dict[str, float | None]:
    summary = quantiles(values)
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in summary.items()}


def counter_to_sorted_dict(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda item: (str(item[0]), item[1]))}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def discover_annotation_files(data_root: Path) -> list[Path]:
    return sorted(data_root.glob("*/annotations.json"))


def list_image_files(group_dir: Path) -> set[str]:
    return {
        path.name
        for path in group_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def load_records(annotation_path: Path) -> list[dict[str, Any]]:
    with annotation_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{annotation_path} must contain a JSON list")
    return payload


def inspect_image(path: Path) -> tuple[int | None, int | None, str | None]:
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height), None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, None, f"{type(exc).__name__}: {exc}"


def add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    issue_type: str,
    group: str,
    file_name: str,
    details: str,
    image_id: str = "",
    object_index: int | str = "",
) -> None:
    issues.append(
        {
            "severity": severity,
            "issue_type": issue_type,
            "group": group,
            "file_name": file_name,
            "image_id": image_id,
            "object_index": object_index,
            "details": details,
        }
    )


def build_stats(data_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    annotation_files = discover_annotation_files(data_root)
    if not annotation_files:
        raise FileNotFoundError(f"No annotations.json files found under {data_root}")

    image_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    validation_issues: list[dict[str, Any]] = []
    group_image_files: dict[str, set[str]] = {}
    seen_image_ids: Counter[str] = Counter()
    seen_file_keys: Counter[str] = Counter()

    for annotation_path in annotation_files:
        group = annotation_path.parent.name
        group_image_files[group] = list_image_files(annotation_path.parent)
        records = load_records(annotation_path)

        annotated_files: set[str] = set()
        for row_index, record in enumerate(records):
            image_id = str(record.get("image_id", ""))
            file_name = str(record.get("file_name", ""))
            annotated_files.add(file_name)
            seen_image_ids[image_id] += 1
            seen_file_keys[f"{group}/{file_name}"] += 1

            ann_width = record.get("width")
            ann_height = record.get("height")
            image_path = annotation_path.parent / file_name
            image_exists = image_path.is_file()
            actual_width: int | None = None
            actual_height: int | None = None
            image_error = ""

            if image_exists:
                actual_width, actual_height, error = inspect_image(image_path)
                if error:
                    image_error = error
                    add_issue(validation_issues, "error", "unreadable_image", group, file_name, error, image_id)
                elif ann_width != actual_width or ann_height != actual_height:
                    add_issue(
                        validation_issues,
                        "warning",
                        "dimension_mismatch",
                        group,
                        file_name,
                        f"annotation={ann_width}x{ann_height}, actual={actual_width}x{actual_height}",
                        image_id,
                    )
            else:
                add_issue(validation_issues, "error", "missing_image", group, file_name, "image referenced by annotation is missing", image_id)

            objects = record.get("objects") or []
            if not isinstance(objects, list):
                add_issue(validation_issues, "error", "invalid_objects_field", group, file_name, "objects is not a list", image_id)
                objects = []

            width_num = float(ann_width or 0)
            height_num = float(ann_height or 0)
            image_area = width_num * height_num
            image_rows.append(
                {
                    "group": group,
                    "image_id": image_id,
                    "file_name": file_name,
                    "annotation_width": ann_width,
                    "annotation_height": ann_height,
                    "actual_width": actual_width if actual_width is not None else "",
                    "actual_height": actual_height if actual_height is not None else "",
                    "image_exists": image_exists,
                    "image_error": image_error,
                    "split": record.get("split", ""),
                    "source_type": record.get("source_type", ""),
                    "domain_type": record.get("domain_type", ""),
                    "object_count": len(objects),
                    "is_positive": len(objects) > 0,
                    "row_index": row_index,
                }
            )

            for object_index, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    add_issue(
                        validation_issues,
                        "error",
                        "invalid_object",
                        group,
                        file_name,
                        "object entry is not a dictionary",
                        image_id,
                        object_index,
                    )
                    continue

                category = str(obj.get("category", ""))
                bbox = obj.get("bbox")
                bbox_valid = isinstance(bbox, list) and len(bbox) == 4
                if not bbox_valid:
                    add_issue(validation_issues, "error", "invalid_bbox_format", group, file_name, f"bbox={bbox!r}", image_id, object_index)
                    continue

                try:
                    x, y, bbox_width, bbox_height = [float(value) for value in bbox]
                except (TypeError, ValueError):
                    add_issue(validation_issues, "error", "invalid_bbox_value", group, file_name, f"bbox={bbox!r}", image_id, object_index)
                    continue

                bbox_area = bbox_width * bbox_height
                x2 = x + bbox_width
                y2 = y + bbox_height
                normalized_width = safe_ratio(bbox_width, width_num)
                normalized_height = safe_ratio(bbox_height, height_num)
                normalized_area = safe_ratio(bbox_area, image_area)
                normalized_x_center = safe_ratio(x + bbox_width / 2, width_num)
                normalized_y_center = safe_ratio(y + bbox_height / 2, height_num)

                if bbox_width <= 0 or bbox_height <= 0:
                    add_issue(validation_issues, "error", "non_positive_bbox", group, file_name, f"bbox={bbox!r}", image_id, object_index)
                if width_num <= 0 or height_num <= 0:
                    add_issue(validation_issues, "error", "invalid_image_dimensions", group, file_name, f"width={ann_width}, height={ann_height}", image_id, object_index)
                elif x < 0 or y < 0 or x2 > width_num or y2 > height_num:
                    add_issue(
                        validation_issues,
                        "warning",
                        "bbox_outside_image",
                        group,
                        file_name,
                        f"bbox={bbox!r}, image={ann_width}x{ann_height}",
                        image_id,
                        object_index,
                    )

                object_rows.append(
                    {
                        "group": group,
                        "image_id": image_id,
                        "file_name": file_name,
                        "object_index": object_index,
                        "category": category,
                        "bbox_x": x,
                        "bbox_y": y,
                        "bbox_width": bbox_width,
                        "bbox_height": bbox_height,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "bbox_area": bbox_area,
                        "bbox_area_ratio": normalized_area,
                        "bbox_width_ratio": normalized_width,
                        "bbox_height_ratio": normalized_height,
                        "bbox_center_x_ratio": normalized_x_center,
                        "bbox_center_y_ratio": normalized_y_center,
                        "bbox_type": obj.get("bbox_type", ""),
                        "visibility": obj.get("visibility", ""),
                        "quality": obj.get("quality", ""),
                    }
                )

        unannotated = sorted(group_image_files[group] - annotated_files)
        for file_name in unannotated:
            add_issue(validation_issues, "warning", "unannotated_image", group, file_name, "image file has no annotation record")

    for image_id, count in seen_image_ids.items():
        if image_id and count > 1:
            add_issue(validation_issues, "warning", "duplicate_image_id", "", "", f"image_id={image_id!r} appears {count} times", image_id)

    for file_key, count in seen_file_keys.items():
        if count > 1:
            group, file_name = file_key.split("/", 1)
            add_issue(validation_issues, "warning", "duplicate_file_annotation", group, file_name, f"{file_key} appears {count} times")

    stats = aggregate_stats(data_root, annotation_files, image_rows, object_rows, validation_issues, group_image_files)
    return stats, image_rows, object_rows, validation_issues


def aggregate_stats(
    data_root: Path,
    annotation_files: list[Path],
    image_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    validation_issues: list[dict[str, Any]],
    group_image_files: dict[str, set[str]],
) -> dict[str, Any]:
    groups = sorted(group_image_files)
    category_counter = Counter(row["category"] for row in object_rows)
    issue_counter = Counter(row["issue_type"] for row in validation_issues)
    severity_counter = Counter(row["severity"] for row in validation_issues)

    def rows_for_group(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
        return [row for row in rows if row.get("group") == group]

    def summarize_group(group: str) -> dict[str, Any]:
        images = rows_for_group(image_rows, group)
        objects = rows_for_group(object_rows, group)
        issue_rows = rows_for_group(validation_issues, group)
        object_counts = [int(row["object_count"]) for row in images]
        widths = [float(row["annotation_width"]) for row in images if row.get("annotation_width")]
        heights = [float(row["annotation_height"]) for row in images if row.get("annotation_height")]
        aspect_ratios = [w / h for w, h in zip(widths, heights) if h]
        return {
            "annotation_records": len(images),
            "image_files": len(group_image_files[group]),
            "positive_images": sum(1 for row in images if row["is_positive"]),
            "negative_images": sum(1 for row in images if not row["is_positive"]),
            "total_objects": len(objects),
            "objects_per_image": summarize_numeric(object_counts),
            "categories": counter_to_sorted_dict(Counter(row["category"] for row in objects)),
            "splits": counter_to_sorted_dict(Counter(row.get("split", "") for row in images)),
            "source_types": counter_to_sorted_dict(Counter(row.get("source_type", "") for row in images)),
            "domain_types": counter_to_sorted_dict(Counter(row.get("domain_type", "") for row in images)),
            "image_width": summarize_numeric(widths),
            "image_height": summarize_numeric(heights),
            "image_aspect_ratio": summarize_numeric(aspect_ratios),
            "validation_issues": counter_to_sorted_dict(Counter(row["issue_type"] for row in issue_rows)),
        }

    bbox_by_category: dict[str, dict[str, Any]] = {}
    for category in sorted(category_counter):
        rows = [row for row in object_rows if row["category"] == category]
        bbox_by_category[category] = {
            "count": len(rows),
            "width_px": summarize_numeric([float(row["bbox_width"]) for row in rows]),
            "height_px": summarize_numeric([float(row["bbox_height"]) for row in rows]),
            "area_px": summarize_numeric([float(row["bbox_area"]) for row in rows]),
            "width_ratio": summarize_numeric([float(row["bbox_width_ratio"]) for row in rows if row["bbox_width_ratio"] is not None]),
            "height_ratio": summarize_numeric([float(row["bbox_height_ratio"]) for row in rows if row["bbox_height_ratio"] is not None]),
            "area_ratio": summarize_numeric([float(row["bbox_area_ratio"]) for row in rows if row["bbox_area_ratio"] is not None]),
            "bbox_types": counter_to_sorted_dict(Counter(row.get("bbox_type", "") for row in rows)),
            "visibility": counter_to_sorted_dict(Counter(row.get("visibility", "") for row in rows)),
            "quality": counter_to_sorted_dict(Counter(row.get("quality", "") for row in rows)),
        }

    total_images = len(image_rows)
    total_image_files = sum(len(files) for files in group_image_files.values())
    positive_images = sum(1 for row in image_rows if row["is_positive"])
    object_counts = [int(row["object_count"]) for row in image_rows]
    widths = [float(row["annotation_width"]) for row in image_rows if row.get("annotation_width")]
    heights = [float(row["annotation_height"]) for row in image_rows if row.get("annotation_height")]
    aspect_ratios = [w / h for w, h in zip(widths, heights) if h]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "annotation_files": [str(path) for path in annotation_files],
        "totals": {
            "annotation_records": total_images,
            "image_files": total_image_files,
            "positive_images": positive_images,
            "negative_images": total_images - positive_images,
            "total_objects": len(object_rows),
            "classes": len(category_counter),
            "validation_issues": len(validation_issues),
        },
        "groups": {group: summarize_group(group) for group in groups},
        "categories": counter_to_sorted_dict(category_counter),
        "splits": counter_to_sorted_dict(Counter(row.get("split", "") for row in image_rows)),
        "source_types": counter_to_sorted_dict(Counter(row.get("source_type", "") for row in image_rows)),
        "domain_types": counter_to_sorted_dict(Counter(row.get("domain_type", "") for row in image_rows)),
        "objects_per_image": summarize_numeric(object_counts),
        "image_width": summarize_numeric(widths),
        "image_height": summarize_numeric(heights),
        "image_aspect_ratio": summarize_numeric(aspect_ratios),
        "bbox_by_category": bbox_by_category,
        "validation": {
            "by_severity": counter_to_sorted_dict(severity_counter),
            "by_issue_type": counter_to_sorted_dict(issue_counter),
        },
    }


def flatten_bbox_summary(stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, summary in stats["bbox_by_category"].items():
        rows.append(
            {
                "category": category,
                "count": summary["count"],
                "width_px_min": summary["width_px"]["min"],
                "width_px_mean": summary["width_px"]["mean"],
                "width_px_median": summary["width_px"]["median"],
                "width_px_max": summary["width_px"]["max"],
                "height_px_min": summary["height_px"]["min"],
                "height_px_mean": summary["height_px"]["mean"],
                "height_px_median": summary["height_px"]["median"],
                "height_px_max": summary["height_px"]["max"],
                "area_px_min": summary["area_px"]["min"],
                "area_px_mean": summary["area_px"]["mean"],
                "area_px_median": summary["area_px"]["median"],
                "area_px_max": summary["area_px"]["max"],
                "area_ratio_mean": summary["area_ratio"]["mean"],
                "area_ratio_median": summary["area_ratio"]["median"],
                "bbox_types": json.dumps(summary["bbox_types"], ensure_ascii=False),
                "visibility": json.dumps(summary["visibility"], ensure_ascii=False),
                "quality": json.dumps(summary["quality"], ensure_ascii=False),
            }
        )
    return rows


def group_summary_rows(stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, summary in stats["groups"].items():
        rows.append(
            {
                "group": group,
                "annotation_records": summary["annotation_records"],
                "image_files": summary["image_files"],
                "positive_images": summary["positive_images"],
                "negative_images": summary["negative_images"],
                "total_objects": summary["total_objects"],
                "objects_per_image_mean": summary["objects_per_image"]["mean"],
                "image_width_median": summary["image_width"]["median"],
                "image_height_median": summary["image_height"]["median"],
                "categories": json.dumps(summary["categories"], ensure_ascii=False),
                "source_types": json.dumps(summary["source_types"], ensure_ascii=False),
                "domain_types": json.dumps(summary["domain_types"], ensure_ascii=False),
                "validation_issues": json.dumps(summary["validation_issues"], ensure_ascii=False),
            }
        )
    return rows


def category_count_rows(object_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    categories = sorted(Counter(row["category"] for row in object_rows))
    groups = sorted(Counter(row["group"] for row in object_rows))
    for category in categories:
        category_rows = [row for row in object_rows if row["category"] == category]
        out = {"category": category, "total": len(category_rows)}
        for group in groups:
            out[group] = sum(1 for row in category_rows if row["group"] == group)
        rows.append(out)
    return rows


def render_markdown(stats: dict[str, Any]) -> str:
    totals = stats["totals"]
    lines = [
        "# Object Detection Dataset Statistics",
        "",
        f"Generated at UTC: `{stats['generated_at_utc']}`",
        "",
        "## Totals",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Annotation records | {totals['annotation_records']} |",
        f"| Image files | {totals['image_files']} |",
        f"| Positive images | {totals['positive_images']} |",
        f"| Negative images | {totals['negative_images']} |",
        f"| Objects | {totals['total_objects']} |",
        f"| Classes | {totals['classes']} |",
        f"| Validation issues | {totals['validation_issues']} |",
        "",
        "## Groups",
        "",
        "| Group | Records | Image files | Positive | Negative | Objects | Median size |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for group, summary in stats["groups"].items():
        width = summary["image_width"]["median"]
        height = summary["image_height"]["median"]
        median_size = f"{width}x{height}"
        lines.append(
            f"| {group} | {summary['annotation_records']} | {summary['image_files']} | "
            f"{summary['positive_images']} | {summary['negative_images']} | "
            f"{summary['total_objects']} | {median_size} |"
        )

    lines.extend(
        [
            "",
            "## Categories",
            "",
            "| Category | Objects |",
            "|---|---:|",
        ]
    )
    for category, count in stats["categories"].items():
        lines.append(f"| {category} | {count} |")

    lines.extend(
        [
            "",
            "## Validation",
            "",
            "| Issue type | Count |",
            "|---|---:|",
        ]
    )
    for issue_type, count in stats["validation"]["by_issue_type"].items():
        lines.append(f"| {issue_type} | {count} |")

    lines.extend(
        [
            "",
            "## BBox Summary By Category",
            "",
            "| Category | Count | Width mean | Height mean | Area ratio mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category, summary in stats["bbox_by_category"].items():
        lines.append(
            f"| {category} | {summary['count']} | {summary['width_px']['mean']} | "
            f"{summary['height_px']['mean']} | {summary['area_ratio']['mean']} |"
        )

    return "\n".join(lines) + "\n"


def save_outputs(
    output_dir: Path,
    stats: dict[str, Any],
    image_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    validation_issues: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "dataset_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "dataset_statistics.md").write_text(render_markdown(stats), encoding="utf-8")

    write_csv(
        output_dir / "group_summary.csv",
        group_summary_rows(stats),
        [
            "group",
            "annotation_records",
            "image_files",
            "positive_images",
            "negative_images",
            "total_objects",
            "objects_per_image_mean",
            "image_width_median",
            "image_height_median",
            "categories",
            "source_types",
            "domain_types",
            "validation_issues",
        ],
    )

    category_rows = category_count_rows(object_rows)
    category_fields = ["category", "total"] + sorted({key for row in category_rows for key in row if key not in {"category", "total"}})
    write_csv(output_dir / "category_counts.csv", category_rows, category_fields)

    write_csv(
        output_dir / "bbox_summary_by_category.csv",
        flatten_bbox_summary(stats),
        [
            "category",
            "count",
            "width_px_min",
            "width_px_mean",
            "width_px_median",
            "width_px_max",
            "height_px_min",
            "height_px_mean",
            "height_px_median",
            "height_px_max",
            "area_px_min",
            "area_px_mean",
            "area_px_median",
            "area_px_max",
            "area_ratio_mean",
            "area_ratio_median",
            "bbox_types",
            "visibility",
            "quality",
        ],
    )

    write_csv(
        output_dir / "image_summary.csv",
        image_rows,
        [
            "group",
            "image_id",
            "file_name",
            "annotation_width",
            "annotation_height",
            "actual_width",
            "actual_height",
            "image_exists",
            "image_error",
            "split",
            "source_type",
            "domain_type",
            "object_count",
            "is_positive",
            "row_index",
        ],
    )

    write_csv(
        output_dir / "object_annotations.csv",
        object_rows,
        [
            "group",
            "image_id",
            "file_name",
            "object_index",
            "category",
            "bbox_x",
            "bbox_y",
            "bbox_width",
            "bbox_height",
            "bbox_x2",
            "bbox_y2",
            "bbox_area",
            "bbox_area_ratio",
            "bbox_width_ratio",
            "bbox_height_ratio",
            "bbox_center_x_ratio",
            "bbox_center_y_ratio",
            "bbox_type",
            "visibility",
            "quality",
        ],
    )

    write_csv(
        output_dir / "validation_issues.csv",
        validation_issues,
        ["severity", "issue_type", "group", "file_name", "image_id", "object_index", "details"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", type=Path, help="Dataset root containing group folders.")
    parser.add_argument(
        "--output-dir",
        default=Path("results/dataset_statistics"),
        type=Path,
        help="Directory where statistics files are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats, image_rows, object_rows, validation_issues = build_stats(args.data_root)
    save_outputs(args.output_dir, stats, image_rows, object_rows, validation_issues)
    print(f"Wrote statistics to {args.output_dir}")
    print(json.dumps(stats["totals"], indent=2))


if __name__ == "__main__":
    main()
