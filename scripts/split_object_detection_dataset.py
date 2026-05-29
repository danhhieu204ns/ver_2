#!/usr/bin/env python3
"""Create a stratified train/val/test split for the object detection dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def parse_ratio(value: str) -> dict[str, float]:
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Ratio must have exactly 3 comma-separated values: train,val,test")
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("Ratio sum must be positive")
    return {split: part / total for split, part in zip(SPLITS, parts)}


def target_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    train = round(total * ratios["train"])
    val = round(total * ratios["val"])
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8f}".rstrip("0").rstrip(".")


def bbox_area_bucket(area_ratio: float | None) -> str:
    if area_ratio is None:
        return "unknown"
    if area_ratio < 0.01:
        return "tiny"
    if area_ratio < 0.05:
        return "small"
    if area_ratio < 0.25:
        return "medium"
    return "large"


def min_bbox_area_ratio(record: dict[str, Any]) -> float | None:
    width = float(record.get("width") or 0)
    height = float(record.get("height") or 0)
    image_area = width * height
    if image_area <= 0:
        return None

    ratios: list[float] = []
    for obj in record.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        bbox = obj.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            bbox_width = float(bbox[2])
            bbox_height = float(bbox[3])
        except (TypeError, ValueError):
            continue
        if bbox_width <= 0 or bbox_height <= 0:
            continue
        ratios.append((bbox_width * bbox_height) / image_area)

    if not ratios:
        return None
    return min(ratios)


def record_stratum(group: str, record: dict[str, Any]) -> tuple[str, float | None]:
    objects = record.get("objects") or []
    if not objects:
        return f"{group}:negative", None

    area_ratio = min_bbox_area_ratio(record)
    bucket = bbox_area_bucket(area_ratio)
    return f"{group}:positive:{bucket}", area_ratio


def load_annotation(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON list")
    return records


def backup_annotations(annotation_files: list[Path], backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for annotation_file in annotation_files:
        destination = backup_dir / annotation_file.parent.name / annotation_file.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(annotation_file, destination)


def allocate_strata_counts(
    stratum_sizes: dict[str, int],
    ratios: dict[str, float],
    split_targets: dict[str, int],
) -> dict[str, dict[str, int]]:
    allocation: dict[str, dict[str, int]] = {}
    row_remaining: dict[str, int] = {}
    column_remaining = dict(split_targets)
    fractions: dict[str, dict[str, float]] = {}

    for stratum, size in sorted(stratum_sizes.items()):
        allocation[stratum] = {}
        fractions[stratum] = {}
        assigned = 0
        for split in SPLITS:
            desired = size * ratios[split]
            count = math.floor(desired)
            allocation[stratum][split] = count
            fractions[stratum][split] = desired - count
            assigned += count
            column_remaining[split] -= count
        row_remaining[stratum] = size - assigned

    for stratum in sorted(stratum_sizes):
        while row_remaining[stratum] > 0:
            candidates = [split for split in SPLITS if column_remaining[split] > 0]
            if not candidates:
                raise RuntimeError("Unable to allocate split counts exactly")
            split = max(
                candidates,
                key=lambda item: (
                    fractions[stratum][item],
                    ratios[item],
                    column_remaining[item],
                ),
            )
            allocation[stratum][split] += 1
            column_remaining[split] -= 1
            row_remaining[stratum] -= 1

    if any(value != 0 for value in column_remaining.values()):
        raise RuntimeError(f"Split allocation mismatch: {column_remaining}")

    return allocation


def assign_group_splits(
    group: str,
    records: list[dict[str, Any]],
    ratios: dict[str, float],
    rng: random.Random,
) -> list[dict[str, Any]]:
    indexed_records: list[dict[str, Any]] = []
    strata: dict[str, list[int]] = defaultdict(list)

    for index, record in enumerate(records):
        stratum, area_ratio = record_stratum(group, record)
        object_count = len(record.get("objects") or [])
        indexed_records.append(
            {
                "index": index,
                "stratum": stratum,
                "min_bbox_area_ratio": area_ratio,
                "object_count": object_count,
                "split": "",
            }
        )
        strata[stratum].append(index)

    for indices in strata.values():
        indices.sort(key=lambda idx: str(records[idx].get("file_name", "")))
        rng.shuffle(indices)

    allocation = allocate_strata_counts(
        {stratum: len(indices) for stratum, indices in strata.items()},
        ratios,
        target_counts(len(records), ratios),
    )

    assignments_by_index = {item["index"]: item for item in indexed_records}
    for stratum, indices in sorted(strata.items()):
        cursor = 0
        for split in SPLITS:
            count = allocation[stratum][split]
            for index in indices[cursor : cursor + count]:
                records[index]["split"] = split
                assignments_by_index[index]["split"] = split
            cursor += count

    return [assignments_by_index[index] for index in range(len(records))]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_reports(
    all_records: dict[Path, list[dict[str, Any]]],
    assignment_meta: dict[Path, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    assignment_rows: list[dict[str, Any]] = []
    summary_counter: Counter[tuple[str, str]] = Counter()
    object_counter: Counter[tuple[str, str]] = Counter()
    positive_counter: Counter[tuple[str, str]] = Counter()
    stratum_counter: Counter[tuple[str, str, str]] = Counter()
    stratum_object_counter: Counter[tuple[str, str, str]] = Counter()

    for annotation_file, records in all_records.items():
        group = annotation_file.parent.name
        for record, meta in zip(records, assignment_meta[annotation_file]):
            split = record["split"]
            object_count = len(record.get("objects") or [])
            key = (group, split)
            stratum_key = (group, meta["stratum"], split)
            summary_counter[key] += 1
            object_counter[key] += object_count
            if object_count > 0:
                positive_counter[key] += 1
            stratum_counter[stratum_key] += 1
            stratum_object_counter[stratum_key] += object_count
            assignment_rows.append(
                {
                    "group": group,
                    "image_id": record.get("image_id", ""),
                    "file_name": record.get("file_name", ""),
                    "split": split,
                    "stratum": meta["stratum"],
                    "object_count": object_count,
                    "min_bbox_area_ratio": format_float(meta["min_bbox_area_ratio"]),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for group in sorted({key[0] for key in summary_counter}):
        for split in SPLITS:
            key = (group, split)
            images = summary_counter[key]
            positives = positive_counter[key]
            summary_rows.append(
                {
                    "group": group,
                    "split": split,
                    "images": images,
                    "positive_images": positives,
                    "negative_images": images - positives,
                    "objects": object_counter[key],
                }
            )

    stratum_rows: list[dict[str, Any]] = []
    for group, stratum, split in sorted(stratum_counter):
        key = (group, stratum, split)
        stratum_rows.append(
            {
                "group": group,
                "stratum": stratum,
                "split": split,
                "images": stratum_counter[key],
                "objects": stratum_object_counter[key],
            }
        )

    return assignment_rows, summary_rows, stratum_rows


def write_split_lists(output_dir: Path, assignment_rows: list[dict[str, Any]]) -> None:
    for split in SPLITS:
        rows = [
            f"data/{row['group']}/{row['file_name']}"
            for row in sorted(assignment_rows, key=lambda item: (item["group"], item["file_name"]))
            if row["split"] == split
        ]
        (output_dir / f"{split}_images.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=Path("data"), type=Path, help="Dataset root containing group folders.")
    parser.add_argument(
        "--output-dir",
        default=Path("results/dataset_split"),
        type=Path,
        help="Directory where split reports are written.",
    )
    parser.add_argument("--ratio", default=parse_ratio("0.70,0.15,0.15"), type=parse_ratio, help="Train,val,test ratio.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for deterministic splitting.")
    parser.add_argument(
        "--backup-dir",
        default=None,
        type=Path,
        help="Directory for annotation backups. Defaults to a timestamped folder under output-dir.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate reports without writing split fields.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_files = sorted(args.data_root.glob("*/annotations.json"))
    if not annotation_files:
        raise FileNotFoundError(f"No annotations.json files found under {args.data_root}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = args.backup_dir or (output_dir / "annotation_backups" / timestamp)

    all_records = {path: load_annotation(path) for path in annotation_files}
    if not args.dry_run:
        backup_annotations(annotation_files, backup_dir)

    assignment_meta: dict[Path, list[dict[str, Any]]] = {}
    for annotation_file, records in all_records.items():
        group = annotation_file.parent.name
        rng = random.Random(f"{args.seed}:{group}")
        assignment_meta[annotation_file] = assign_group_splits(group, records, args.ratio, rng)

    assignment_rows, summary_rows, stratum_rows = build_reports(all_records, assignment_meta)

    if not args.dry_run:
        for annotation_file, records in all_records.items():
            annotation_file.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_csv(
        output_dir / "split_assignments.csv",
        assignment_rows,
        ["group", "image_id", "file_name", "split", "stratum", "object_count", "min_bbox_area_ratio"],
    )
    write_csv(output_dir / "split_summary.csv", summary_rows, ["group", "split", "images", "positive_images", "negative_images", "objects"])
    write_csv(output_dir / "split_strata_summary.csv", stratum_rows, ["group", "stratum", "split", "images", "objects"])
    write_split_lists(output_dir, assignment_rows)

    totals = Counter(row["split"] for row in assignment_rows)
    result = {
        "dry_run": args.dry_run,
        "seed": args.seed,
        "ratio": args.ratio,
        "backup_dir": "" if args.dry_run else str(backup_dir),
        "output_dir": str(output_dir),
        "totals": {split: totals[split] for split in SPLITS},
    }
    (output_dir / "split_config.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
