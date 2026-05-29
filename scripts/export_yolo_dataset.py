#!/usr/bin/env python3
"""Export the local detection annotations to an Ultralytics YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
CLASS_NAME = "nine_dash_line"


def load_annotation(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def safe_output_name(group: str, file_name: str) -> str:
    return f"{group}__{Path(file_name).name}"


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bbox_to_yolo_line(bbox: list[Any], image_width: float, image_height: float) -> str | None:
    if image_width <= 0 or image_height <= 0:
        return None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        x, y, width, height = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    x1 = clip(x, 0.0, image_width)
    y1 = clip(y, 0.0, image_height)
    x2 = clip(x + width, 0.0, image_width)
    y2 = clip(y + height, 0.0, image_height)
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    norm_width = clipped_width / image_width
    norm_height = clipped_height / image_height
    return f"0 {x_center:.8f} {y_center:.8f} {norm_width:.8f} {norm_height:.8f}"


def link_or_copy_image(source: Path, destination: Path, copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy:
        shutil.copy2(source, destination)
    else:
        os.symlink(source.resolve(), destination)


def write_dataset_yaml(output_dir: Path) -> None:
    yaml_text = "\n".join(
        [
            f"path: {output_dir.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            f"  0: {CLASS_NAME}",
            "",
        ]
    )
    (output_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def discover_records(data_root: Path, splits: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for annotation_path in sorted(data_root.glob("*/annotations.json")):
        group = annotation_path.parent.name
        for record in load_annotation(annotation_path):
            split = str(record.get("split", ""))
            if split not in splits:
                continue
            file_name = str(record.get("file_name", ""))
            image_path = annotation_path.parent / file_name
            rows.append(
                {
                    "group": group,
                    "split": split,
                    "record": record,
                    "source_path": image_path,
                    "output_name": safe_output_name(group, file_name),
                }
            )
    rows.sort(key=lambda item: (item["split"], item["group"], item["output_name"]))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=Path("data"), type=Path, help="Dataset root containing group folders.")
    parser.add_argument(
        "--output-dir",
        default=Path("results/baselines/yolo_dataset"),
        type=Path,
        help="Output directory for the YOLO dataset.",
    )
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated splits to export.")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of creating symlinks.")
    parser.add_argument("--overwrite", action="store_true", help="Remove output-dir before exporting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    output_dir = args.output_dir
    splits = {split.strip() for split in args.splits.split(",") if split.strip()}
    unknown_splits = splits - set(SPLITS)
    if unknown_splits:
        raise ValueError(f"Unsupported split(s): {sorted(unknown_splits)}")

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty. Use --overwrite to rebuild it.")

    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    records = discover_records(data_root, splits)
    if not records:
        raise FileNotFoundError(f"No annotation records found under {data_root} for splits {sorted(splits)}")

    manifest_rows: list[dict[str, Any]] = []
    split_counter: Counter[str] = Counter()
    object_counter: Counter[str] = Counter()
    skipped_objects = 0

    for item in records:
        record = item["record"]
        split = item["split"]
        source_path = item["source_path"]
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing image referenced by annotation: {source_path}")

        image_name = item["output_name"]
        label_name = f"{Path(image_name).stem}.txt"
        image_destination = output_dir / "images" / split / image_name
        label_destination = output_dir / "labels" / split / label_name

        link_or_copy_image(source_path, image_destination, args.copy)

        image_width = float(record.get("width") or 0)
        image_height = float(record.get("height") or 0)
        label_lines: list[str] = []
        for obj in record.get("objects") or []:
            if not isinstance(obj, dict):
                skipped_objects += 1
                continue
            line = bbox_to_yolo_line(obj.get("bbox"), image_width, image_height)
            if line is None:
                skipped_objects += 1
                continue
            label_lines.append(line)

        label_destination.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        split_counter[split] += 1
        object_counter[split] += len(label_lines)
        manifest_rows.append(
            {
                "split": split,
                "group": item["group"],
                "source_path": str(source_path),
                "image_path": str(image_destination),
                "label_path": str(label_destination),
                "objects": len(label_lines),
            }
        )

    write_dataset_yaml(output_dir)
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["split", "group", "source_path", "image_path", "label_path", "objects"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        split: {"images": split_counter[split], "objects": object_counter[split]}
        for split in SPLITS
        if split in splits
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": summary, "skipped_objects": skipped_objects}, indent=2))


if __name__ == "__main__":
    main()
