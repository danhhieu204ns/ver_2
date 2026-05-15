import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


CATEGORIES = {
    "positive_real": {
        "label_type": "positive_real",
        "domain_default": "map_sea",
        "is_synthetic": False,
        "is_hard_negative": False,
    },
    "positive_synthetic": {
        "label_type": "synthetic_pos",
        "domain_default": "map_sea",
        "is_synthetic": True,
        "is_hard_negative": False,
    },
    "negative_in_domain": {
        "label_type": "in_domain_neg",
        "domain_default": "map_sea",
        "is_synthetic": False,
        "is_hard_negative": False,
    },
    "negative_out_domain": {
        "label_type": "out_domain_neg",
        "domain_default": "non_map",
        "is_synthetic": False,
        "is_hard_negative": False,
    },
    "negative_hard": {
        "label_type": "hard_neg",
        "domain_default": "hard_negative",
        "is_synthetic": False,
        "is_hard_negative": True,
    },
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_CATEGORIES = [{"id": 1, "name": "nine_dash_line", "supercategory": "object"}]
SPLITS = ("train", "val", "test_standard", "test_robustness")


def stable_int(text):
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def stable_bucket(text):
    return stable_int(text) % 10000 / 10000.0


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def numeric_suffix(text):
    match = re.search(r"(\d+)$", Path(text).stem)
    return match.group(1) if match else None


def get_synthetic_parent_info(record):
    parent_fields = ("synthetic_parent_id", "parent_id", "source_id", "original_image_id", "base_image_id")
    for field in parent_fields:
        value = record.get(field)
        if value:
            return str(value), "known", field

    return "", "missing", ""


def infer_source_id(category_name, record):
    existing = record.get("source_id")
    if existing:
        return str(existing)

    if category_name == "positive_synthetic":
        parent_id, parent_status, _ = get_synthetic_parent_info(record)
        if parent_status == "known":
            return parent_id
        return str(record.get("image_id") or Path(record.get("file_name", "")).stem)

    return str(record.get("image_id") or Path(record.get("file_name", "")).stem)


def choose_split(category_name, source_id):
    # Synthetic positives are kept train-only until real parent provenance exists.
    # Without parent provenance, putting synthetic images into val/test would
    # contaminate evaluation. This does not prove absence of train-test leakage
    # against real parents; the leakage report makes that limitation explicit.
    if category_name == "positive_synthetic":
        return "train"

    bucket = stable_bucket(source_id)
    if bucket < 0.70:
        return "train"
    if bucket < 0.80:
        return "val"
    if bucket < 0.90:
        return "test_standard"
    return "test_robustness"


def load_records(category_dir):
    ann_path = category_dir / "annotations.json"
    if not ann_path.exists():
        return [], f"missing annotation file: {ann_path}"
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{ann_path} must be a JSON list")
    return data, None


def list_images(category_dir):
    return {
        path.name: path
        for path in category_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def validate_and_clip_bbox(bbox, width, height):
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None, "bbox_not_xywh"
    try:
        x, y, w, h = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None, "bbox_non_numeric"

    if width <= 0 or height <= 0:
        return None, "invalid_image_size"
    if w <= 0 or h <= 0:
        return None, "bbox_non_positive_size"

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + w)
    y2 = min(float(height), y + h)
    clipped_w = x2 - x1
    clipped_h = y2 - y1

    if clipped_w <= 0 or clipped_h <= 0:
        return None, "bbox_outside_image"
    if clipped_w < 1.0 or clipped_h < 1.0:
        return None, "bbox_too_small_after_clip"

    issue = None
    if x < 0 or y < 0 or x + w > width or y + h > height:
        issue = "bbox_clipped_to_image"

    return [round(x1, 4), round(y1, 4), round(clipped_w, 4), round(clipped_h, 4)], issue


def object_size_label(bbox, width, height):
    area = bbox[2] * bbox[3]
    image_area = max(float(width * height), 1.0)
    ratio = area / image_area
    if ratio < 0.01:
        return "small"
    if ratio < 0.05:
        return "medium"
    return "large"


def build_coco(records):
    images = []
    annotations = []
    ann_id = 1

    for image_id, record in enumerate(records, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": record["relative_path"],
                "width": record["width"],
                "height": record["height"],
                "source_id": record["source_id"],
                "label_type": record["label_type"],
                "domain_type": record["domain_type"],
                "split": record["split"],
                "synthetic_parent_id": record["synthetic_parent_id"],
                "parent_provenance_status": record["parent_provenance_status"],
            }
        )

        for obj in record["objects"]:
            bbox = obj["bbox"]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 4),
                    "iscrowd": 0,
                    "bbox_type": obj.get("bbox_type", ""),
                    "visibility": obj.get("visibility", ""),
                    "quality": obj.get("quality", ""),
                }
            )
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": COCO_CATEGORIES}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_metadata_csv(path, records):
    fieldnames = [
        "image_id",
        "file_name",
        "relative_path",
        "split",
        "label_type",
        "source_type",
        "domain_type",
        "source_id",
        "synthetic_parent_id",
        "parent_provenance_status",
        "parent_provenance_field",
        "is_synthetic",
        "is_hard_negative",
        "has_bbox",
        "object_count",
        "object_size",
        "width",
        "height",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in fieldnames})


def write_split_files(output_dir, records):
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        paths = sorted(
            [record["relative_path"] for record in records if record["split"] == split],
            key=natural_key,
        )
        (split_dir / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")


def build_leakage_report(records):
    source_to_splits = {}
    for record in records:
        source_to_splits.setdefault(record["source_id"], set()).add(record["split"])

    source_id_split_collisions = [
        {"source_id": source_id, "splits": sorted(splits)}
        for source_id, splits in sorted(source_to_splits.items())
        if len(splits) > 1
    ]

    real_records = [
        record
        for record in records
        if record["label_type"] == "positive_real"
    ]
    real_lookup = {}
    for record in real_records:
        keys = {
            record["source_id"],
            record["image_id"],
            Path(record["file_name"]).stem,
        }
        for key in keys:
            if key:
                real_lookup[str(key)] = record

    synthetic_records = [
        record
        for record in records
        if record["label_type"] == "synthetic_pos"
    ]
    synthetic_missing_parent = [
        {
            "image_id": record["image_id"],
            "file_name": record["file_name"],
            "split": record["split"],
        }
        for record in synthetic_records
        if record["parent_provenance_status"] != "known"
    ]

    synthetic_parent_not_found = []
    synthetic_train_parent_eval_leaks = []
    synthetic_parent_split_collisions = []
    parent_to_splits = {}

    for record in synthetic_records:
        parent_id = record["synthetic_parent_id"]
        if not parent_id:
            continue

        parent_to_splits.setdefault(parent_id, set()).add(record["split"])
        parent = real_lookup.get(parent_id)
        if parent is None:
            synthetic_parent_not_found.append(
                {
                    "synthetic_image_id": record["image_id"],
                    "synthetic_file_name": record["file_name"],
                    "synthetic_split": record["split"],
                    "synthetic_parent_id": parent_id,
                }
            )
            continue

        parent_to_splits.setdefault(parent_id, set()).add(parent["split"])
        if record["split"] == "train" and parent["split"] != "train":
            synthetic_train_parent_eval_leaks.append(
                {
                    "synthetic_image_id": record["image_id"],
                    "synthetic_file_name": record["file_name"],
                    "synthetic_split": record["split"],
                    "parent_image_id": parent["image_id"],
                    "parent_file_name": parent["file_name"],
                    "parent_split": parent["split"],
                    "synthetic_parent_id": parent_id,
                }
            )

    for parent_id, splits in sorted(parent_to_splits.items()):
        if len(splits) > 1:
            synthetic_parent_split_collisions.append(
                {"synthetic_parent_id": parent_id, "splits": sorted(splits)}
            )

    return {
        "split_strategy": "source_id_hash_for_non_synthetic; synthetic_train_only_until_parent_provenance_exists",
        "source_id_split_collisions": source_id_split_collisions,
        "synthetic_missing_parent": synthetic_missing_parent,
        "synthetic_parent_not_found": synthetic_parent_not_found,
        "synthetic_train_parent_eval_leaks": synthetic_train_parent_eval_leaks,
        "synthetic_parent_split_collisions": synthetic_parent_split_collisions,
        "verdict": {
            "non_synthetic_source_id_leakage": len(source_id_split_collisions) == 0,
            "synthetic_parent_leakage_checkable": len(synthetic_missing_parent) == 0,
            "synthetic_train_parent_eval_leakage": len(synthetic_train_parent_eval_leaks) == 0,
        },
    }


def write_summary_md(path, summary, report):
    lines = [
        "# Dataset Audit Summary",
        "",
        "## Counts",
        "",
        "| Category | Images on disk | Annotation records | Missing annotation | Annotation without image | Valid objects | Invalid objects |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, item in summary["category_counts"].items():
        lines.append(
            "| {category} | {images_on_disk} | {annotation_records} | {missing_annotation} | "
            "{annotation_without_image} | {valid_objects} | {invalid_objects} |".format(
                category=category,
                **item,
            )
        )

    lines.extend(
        [
            "",
            "## Split Counts",
            "",
            "| Split | Images | Objects |",
            "| --- | ---: | ---: |",
        ]
    )
    for split in SPLITS:
        split_item = summary["split_counts"].get(split, {"images": 0, "objects": 0})
        lines.append(f"| {split} | {split_item['images']} | {split_item['objects']} |")

    lines.extend(
        [
            "",
            "## Split By Category",
            "",
            "| Split | Category | Images | Objects |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for split in SPLITS:
        for category in sorted(CATEGORIES):
            item = summary["split_category_counts"].get(split, {}).get(category, {"images": 0, "objects": 0})
            lines.append(f"| {split} | {category} | {item['images']} | {item['objects']} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Images missing annotation: {len(report['missing_annotations'])}",
            f"- Annotation records without image file: {len(report['annotation_without_images'])}",
            f"- Invalid bbox objects removed from COCO export: {len(report['invalid_objects'])}",
            f"- Bboxes clipped to image boundary but kept: {len(report['clipped_objects'])}",
            f"- Source IDs appearing in more than one split: {len(report['leakage']['source_id_split_collisions'])}",
            f"- Synthetic images without known parent provenance: {len(report['leakage']['synthetic_missing_parent'])}",
            f"- Synthetic train images whose known parent is in val/test: {len(report['leakage']['synthetic_train_parent_eval_leaks'])}",
            "- `positive_synthetic` is assigned to train-only; synthetic images should not be used for validation/test metrics.",
        ]
    )
    if len(report["leakage"]["synthetic_missing_parent"]) == 0:
        lines.append("- Synthetic parent leakage is fully checkable: every synthetic image has known parent provenance.")
    else:
        lines.append(
            "- Synthetic parent leakage is not fully checkable until each synthetic image has a real "
            "`synthetic_parent_id`/`parent_id` that matches a `positive_real` image."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_dataset(data_dir, output_dir):
    enriched_records = []
    report = {
        "missing_annotations": [],
        "annotation_without_images": [],
        "invalid_objects": [],
        "clipped_objects": [],
        "positive_without_valid_bbox": [],
        "negative_with_objects": [],
        "duplicate_image_ids": [],
        "annotation_load_errors": [],
        "leakage": {},
    }
    summary = {
        "category_counts": {},
        "split_counts": {},
        "split_category_counts": {},
    }
    seen_image_ids = Counter()

    for category_name in sorted(CATEGORIES):
        category_dir = data_dir / category_name
        if not category_dir.exists():
            continue

        config = CATEGORIES[category_name]
        disk_images = list_images(category_dir)
        records, load_error = load_records(category_dir)
        if load_error:
            report["annotation_load_errors"].append(load_error)
            records = []

        by_file_name = {record.get("file_name"): record for record in records}
        missing_annotations = sorted(set(disk_images) - set(by_file_name), key=natural_key)
        annotation_without_images = sorted(set(by_file_name) - set(disk_images), key=natural_key)

        for file_name in missing_annotations:
            report["missing_annotations"].append({"category": category_name, "file_name": file_name})
        for file_name in annotation_without_images:
            report["annotation_without_images"].append({"category": category_name, "file_name": file_name})

        valid_object_count = 0
        invalid_object_count = 0

        for record in sorted(records, key=lambda item: natural_key(item.get("file_name", ""))):
            file_name = record.get("file_name", "")
            if file_name not in disk_images:
                continue

            image_id = str(record.get("image_id") or Path(file_name).stem)
            seen_image_ids[image_id] += 1
            width = int(record.get("width") or 0)
            height = int(record.get("height") or 0)
            source_id = infer_source_id(category_name, record)
            if config["is_synthetic"]:
                synthetic_parent_id, parent_status, parent_field = get_synthetic_parent_info(record)
            else:
                synthetic_parent_id, parent_status, parent_field = "", "not_synthetic", ""
            split = record.get("split") or choose_split(category_name, source_id)
            domain_type = record.get("domain_type") or config["domain_default"]
            source_type = record.get("source_type") or category_name

            valid_objects = []
            object_size_counts = Counter()
            for object_index, obj in enumerate(record.get("objects") or []):
                bbox, issue = validate_and_clip_bbox(obj.get("bbox"), width, height)
                if bbox is None:
                    invalid_object_count += 1
                    report["invalid_objects"].append(
                        {
                            "category": category_name,
                            "image_id": image_id,
                            "file_name": file_name,
                            "object_index": object_index,
                            "reason": issue,
                            "bbox": obj.get("bbox"),
                        }
                    )
                    continue

                if issue == "bbox_clipped_to_image":
                    report["clipped_objects"].append(
                        {
                            "category": category_name,
                            "image_id": image_id,
                            "file_name": file_name,
                            "object_index": object_index,
                            "bbox": obj.get("bbox"),
                            "clipped_bbox": bbox,
                        }
                    )

                valid_object = dict(obj)
                valid_object["bbox"] = bbox
                valid_objects.append(valid_object)
                valid_object_count += 1
                object_size_counts[object_size_label(bbox, width, height)] += 1

            if category_name.startswith("positive") and not valid_objects:
                report["positive_without_valid_bbox"].append({"category": category_name, "file_name": file_name})
            if category_name.startswith("negative") and valid_objects:
                report["negative_with_objects"].append({"category": category_name, "file_name": file_name})

            if object_size_counts:
                object_size = object_size_counts.most_common(1)[0][0]
            else:
                object_size = ""

            enriched_records.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "relative_path": f"{category_name}/{file_name}",
                    "split": split,
                    "label_type": config["label_type"],
                    "source_type": source_type,
                    "domain_type": domain_type,
                    "source_id": source_id,
                    "synthetic_parent_id": synthetic_parent_id,
                    "parent_provenance_status": parent_status,
                    "parent_provenance_field": parent_field,
                    "is_synthetic": config["is_synthetic"],
                    "is_hard_negative": config["is_hard_negative"],
                    "has_bbox": bool(valid_objects),
                    "object_count": len(valid_objects),
                    "object_size": object_size,
                    "width": width,
                    "height": height,
                    "objects": valid_objects,
                }
            )

        summary["category_counts"][category_name] = {
            "images_on_disk": len(disk_images),
            "annotation_records": len(records),
            "missing_annotation": len(missing_annotations),
            "annotation_without_image": len(annotation_without_images),
            "valid_objects": valid_object_count,
            "invalid_objects": invalid_object_count,
        }

    report["duplicate_image_ids"] = [
        {"image_id": image_id, "count": count}
        for image_id, count in sorted(seen_image_ids.items())
        if count > 1
    ]
    report["leakage"] = build_leakage_report(enriched_records)

    for split in SPLITS:
        records = [record for record in enriched_records if record["split"] == split]
        summary["split_counts"][split] = {
            "images": len(records),
            "objects": sum(record["object_count"] for record in records),
        }
        summary["split_category_counts"][split] = {}
        for category in sorted(CATEGORIES):
            category_records = [
                record for record in records if record["relative_path"].startswith(f"{category}/")
            ]
            summary["split_category_counts"][split][category] = {
                "images": len(category_records),
                "objects": sum(record["object_count"] for record in category_records),
            }

    split_dir = output_dir / "coco"
    for split in SPLITS:
        records = [record for record in enriched_records if record["split"] == split]
        write_json(split_dir / f"{split}.json", build_coco(records))

    write_json(split_dir / "all.json", build_coco(enriched_records))
    write_json(output_dir / "custom_annotations_with_metadata.json", enriched_records)
    write_json(output_dir / "audit_report.json", {"summary": summary, "report": report})
    write_metadata_csv(output_dir / "metadata.csv", enriched_records)
    write_split_files(output_dir, enriched_records)
    write_summary_md(output_dir / "audit_summary.md", summary, report)

    return summary, report


def main():
    parser = argparse.ArgumentParser(description="Audit and export the nine-dash-line dataset.")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--output-dir", default=Path("data") / "audit", type=Path)
    args = parser.parse_args()

    summary, report = audit_dataset(args.data_dir, args.output_dir)
    print(json.dumps({"summary": summary, "issues": {key: len(value) for key, value in report.items()}}, indent=2))


if __name__ == "__main__":
    main()
