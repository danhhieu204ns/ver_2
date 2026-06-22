#!/usr/bin/env python3
"""Create dataset roots for the Experiment 2 training-data ablation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


GROUPS = ("positive", "negative_in_domain", "negative_out_domain")
SPLITS = ("train", "val", "test")
VARIANTS: dict[str, dict[str, Any]] = {
    "positive_only": {
        "training_data": "Positive only",
        "train_groups": {"positive"},
    },
    "positive_negative_out_domain": {
        "training_data": "Positive + negative out-domain",
        "train_groups": {"positive", "negative_out_domain"},
    },
    "positive_negative_in_domain": {
        "training_data": "Positive + negative in-domain",
        "train_groups": {"positive", "negative_in_domain"},
    },
    "positive_both_negatives": {
        "training_data": "Positive + both negatives",
        "train_groups": {"positive", "negative_in_domain", "negative_out_domain"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=Path("data"), type=Path, help="Source dataset root.")
    parser.add_argument(
        "--output-root",
        default=Path("results/dataset_ablation/data"),
        type=Path,
        help="Output root containing one generated dataset root per variant.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=tuple(VARIANTS),
        help="Ablation variants to generate.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to materialize images inside each generated dataset root.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove existing variant directories first.")
    return parser.parse_args()


def load_annotation(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def variant_records(records: list[dict[str, Any]], group: str, train_groups: set[str]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for record in records:
        split = str(record.get("split", ""))
        if split == "train" and group not in train_groups:
            continue
        filtered.append(record)
    return filtered


def remove_existing_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def materialize_image(source: Path, destination: Path, image_mode: str, overwrite: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing source image: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if overwrite:
            remove_existing_path(destination)
        else:
            return

    if image_mode == "symlink":
        destination.symlink_to(source.resolve())
    elif image_mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported image mode: {image_mode}")


def summarize_records(records_by_group: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    split_counter: Counter[str] = Counter()
    group_counter: Counter[tuple[str, str]] = Counter()
    positive_counter: Counter[tuple[str, str]] = Counter()
    object_counter: Counter[tuple[str, str]] = Counter()

    for group, records in records_by_group.items():
        for record in records:
            split = str(record.get("split", ""))
            objects = record.get("objects") or []
            split_counter[split] += 1
            group_counter[(split, group)] += 1
            object_counter[(split, group)] += len(objects)
            if objects:
                positive_counter[(split, group)] += 1

    summary: dict[str, Any] = {}
    for split in SPLITS:
        groups: dict[str, Any] = {}
        for group in GROUPS:
            images = group_counter[(split, group)]
            positives = positive_counter[(split, group)]
            groups[group] = {
                "images": images,
                "positive_images": positives,
                "negative_images": images - positives,
                "objects": object_counter[(split, group)],
            }
        summary[split] = {
            "images": split_counter[split],
            "positive_images": sum(groups[group]["positive_images"] for group in GROUPS),
            "negative_images": sum(groups[group]["negative_images"] for group in GROUPS),
            "objects": sum(groups[group]["objects"] for group in GROUPS),
            "groups": groups,
        }
    return summary


def build_variant(
    source_records: dict[str, list[dict[str, Any]]],
    data_root: Path,
    output_root: Path,
    variant: str,
    image_mode: str,
    overwrite: bool,
) -> dict[str, Any]:
    config = VARIANTS[variant]
    train_groups = set(config["train_groups"])
    variant_root = output_root / variant

    if variant_root.exists() and overwrite:
        shutil.rmtree(variant_root)
    variant_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    records_by_group: dict[str, list[dict[str, Any]]] = {}
    for group in GROUPS:
        records = variant_records(source_records[group], group, train_groups)
        records_by_group[group] = records

        group_dir = variant_root / group
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / "annotations.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        for record in records:
            file_name = str(record.get("file_name", ""))
            if not file_name:
                raise ValueError(f"{data_root / group / 'annotations.json'} has a record without file_name")
            materialize_image(data_root / group / file_name, group_dir / file_name, image_mode, overwrite)
            split = str(record.get("split", ""))
            objects = record.get("objects") or []
            manifest_rows.append(
                {
                    "variant": variant,
                    "training_data": config["training_data"],
                    "group": group,
                    "split": split,
                    "included_in_train": split == "train" and group in train_groups,
                    "image_id": record.get("image_id", ""),
                    "file_name": file_name,
                    "object_count": len(objects),
                }
            )

    summary = {
        "variant": variant,
        "training_data": config["training_data"],
        "train_groups": sorted(train_groups),
        "data_root": str(variant_root),
        "image_mode": image_mode,
        "splits": summarize_records(records_by_group),
    }

    (variant_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(
        variant_root / "manifest.csv",
        manifest_rows,
        ["variant", "training_data", "group", "split", "included_in_train", "image_id", "file_name", "object_count"],
    )
    return summary


def main() -> None:
    args = parse_args()
    source_records: dict[str, list[dict[str, Any]]] = {}
    for group in GROUPS:
        annotation_path = args.data_root / group / "annotations.json"
        source_records[group] = load_annotation(annotation_path)

    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        build_variant(
            source_records=source_records,
            data_root=args.data_root,
            output_root=args.output_root,
            variant=variant,
            image_mode=args.image_mode,
            overwrite=args.overwrite,
        )
        for variant in args.variants
    ]
    (args.output_root / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variants": len(summaries), "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
