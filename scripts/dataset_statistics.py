import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test_standard", "test_robustness")
CATEGORY_ORDER = (
    "positive_real",
    "in_domain_neg",
    "out_domain_neg",
    "synthetic_pos",
    "hard_neg",
)
CATEGORY_DISPLAY = {
    "positive_real": "Positive real",
    "in_domain_neg": "In-domain negative",
    "out_domain_neg": "Out-domain negative",
    "synthetic_pos": "Synthetic positive",
    "hard_neg": "Hard negative",
}


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bool_text(value):
    return "Yes" if value else "No"


def percentile(sorted_values, q):
    if not sorted_values:
        return ""
    if len(sorted_values) == 1:
        return round(sorted_values[0], 6)
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    value = sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    return round(value, 6)


def describe(values):
    values = sorted(values)
    if not values:
        return {
            "count": 0,
            "min": "",
            "p25": "",
            "median": "",
            "mean": "",
            "p75": "",
            "max": "",
        }
    return {
        "count": len(values),
        "min": round(values[0], 6),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "mean": round(sum(values) / len(values), 6),
        "p75": percentile(values, 0.75),
        "max": round(values[-1], 6),
    }


def object_size_from_area_ratio(area_ratio):
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


def load_metadata(metadata_path):
    rows = read_csv(metadata_path)
    for row in rows:
        row["width"] = as_int(row.get("width"))
        row["height"] = as_int(row.get("height"))
        row["object_count"] = as_int(row.get("object_count"))
    return rows


def load_bbox_rows(coco_path):
    coco = read_json(coco_path)
    images = {image["id"]: image for image in coco.get("images", [])}
    rows = []
    for ann in coco.get("annotations", []):
        image = images.get(ann.get("image_id"))
        if not image:
            continue
        x, y, w, h = [as_float(value) for value in ann.get("bbox", [0, 0, 0, 0])]
        width = as_float(image.get("width"))
        height = as_float(image.get("height"))
        image_area = max(width * height, 1.0)
        area = as_float(ann.get("area"), w * h)
        area_ratio = area / image_area
        rows.append(
            {
                "image_id": image.get("id"),
                "file_name": image.get("file_name", ""),
                "split": image.get("split", ""),
                "label_type": image.get("label_type", ""),
                "domain_type": image.get("domain_type", ""),
                "image_width": round(width, 6),
                "image_height": round(height, 6),
                "bbox_x": round(x, 6),
                "bbox_y": round(y, 6),
                "bbox_width": round(w, 6),
                "bbox_height": round(h, 6),
                "bbox_area": round(area, 6),
                "bbox_area_ratio": round(area_ratio, 8),
                "object_size": object_size_from_area_ratio(area_ratio),
            }
        )
    return rows


def build_dataset_summary(metadata_rows):
    grouped = defaultdict(list)
    for row in metadata_rows:
        grouped[row["label_type"]].append(row)

    rows = []
    for label_type in CATEGORY_ORDER:
        items = grouped.get(label_type, [])
        splits = {row["split"] for row in items}
        has_any_bbox = any(row["object_count"] > 0 for row in items)
        bbox_text = "Yes" if has_any_bbox else "No"
        if label_type == "hard_neg":
            bbox_text = "No / pseudo-box"
        rows.append(
            {
                "category": CATEGORY_DISPLAY[label_type],
                "label_type": label_type,
                "images": len(items),
                "instances": sum(row["object_count"] for row in items),
                "bbox": bbox_text,
                "used_in_train": bool_text("train" in splits),
                "used_in_val": bool_text("val" in splits),
                "used_in_test": bool_text(
                    "test_standard" in splits or "test_robustness" in splits
                ),
            }
        )
    return rows


def build_split_category_counts(metadata_rows):
    grouped = defaultdict(lambda: {"images": 0, "instances": 0})
    for row in metadata_rows:
        key = (row["split"], row["label_type"])
        grouped[key]["images"] += 1
        grouped[key]["instances"] += row["object_count"]

    rows = []
    for split in SPLITS:
        for label_type in CATEGORY_ORDER:
            item = grouped[(split, label_type)]
            rows.append(
                {
                    "split": split,
                    "category": CATEGORY_DISPLAY[label_type],
                    "label_type": label_type,
                    "images": item["images"],
                    "instances": item["instances"],
                }
            )
    return rows


def build_resolution_distribution(metadata_rows):
    counts = Counter((row["width"], row["height"]) for row in metadata_rows)
    rows = []
    for (width, height), count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 6) if height else "",
                "megapixels": round(width * height / 1_000_000, 6),
                "images": count,
            }
        )
    return rows


def build_domain_distribution(metadata_rows):
    grouped = defaultdict(lambda: {"images": 0, "instances": 0})
    for row in metadata_rows:
        key = (row["domain_type"], row["label_type"], row["split"])
        grouped[key]["images"] += 1
        grouped[key]["instances"] += row["object_count"]

    rows = []
    for (domain_type, label_type, split), item in sorted(grouped.items()):
        rows.append(
            {
                "domain_type": domain_type,
                "category": CATEGORY_DISPLAY.get(label_type, label_type),
                "label_type": label_type,
                "split": split,
                "images": item["images"],
                "instances": item["instances"],
            }
        )
    return rows


def build_object_size_distribution(bbox_rows):
    grouped = defaultdict(int)
    for row in bbox_rows:
        grouped[(row["split"], row["label_type"], row["object_size"])] += 1

    rows = []
    for split in SPLITS:
        for label_type in ("positive_real", "synthetic_pos"):
            for object_size in ("small", "medium", "large"):
                rows.append(
                    {
                        "split": split,
                        "category": CATEGORY_DISPLAY[label_type],
                        "label_type": label_type,
                        "object_size": object_size,
                        "instances": grouped[(split, label_type, object_size)],
                    }
                )
    return rows


def build_bbox_summary(bbox_rows):
    grouped = defaultdict(lambda: {"width": [], "height": [], "area_ratio": []})
    for row in bbox_rows:
        keys = [
            ("all", "all"),
            (row["split"], "all"),
            (row["split"], row["label_type"]),
            ("all", row["label_type"]),
        ]
        for key in keys:
            grouped[key]["width"].append(as_float(row["bbox_width"]))
            grouped[key]["height"].append(as_float(row["bbox_height"]))
            grouped[key]["area_ratio"].append(as_float(row["bbox_area_ratio"]))

    rows = []
    for (split, label_type), values in sorted(grouped.items()):
        width_stats = describe(values["width"])
        height_stats = describe(values["height"])
        area_stats = describe(values["area_ratio"])
        rows.append(
            {
                "split": split,
                "label_type": label_type,
                "instances": width_stats["count"],
                "bbox_width_min": width_stats["min"],
                "bbox_width_p25": width_stats["p25"],
                "bbox_width_median": width_stats["median"],
                "bbox_width_mean": width_stats["mean"],
                "bbox_width_p75": width_stats["p75"],
                "bbox_width_max": width_stats["max"],
                "bbox_height_min": height_stats["min"],
                "bbox_height_p25": height_stats["p25"],
                "bbox_height_median": height_stats["median"],
                "bbox_height_mean": height_stats["mean"],
                "bbox_height_p75": height_stats["p75"],
                "bbox_height_max": height_stats["max"],
                "bbox_area_ratio_min": area_stats["min"],
                "bbox_area_ratio_p25": area_stats["p25"],
                "bbox_area_ratio_median": area_stats["median"],
                "bbox_area_ratio_mean": area_stats["mean"],
                "bbox_area_ratio_p75": area_stats["p75"],
                "bbox_area_ratio_max": area_stats["max"],
            }
        )
    return rows


def build_image_summary(metadata_rows):
    grouped = defaultdict(lambda: {"width": [], "height": [], "megapixels": [], "images": 0})
    for row in metadata_rows:
        width = row["width"]
        height = row["height"]
        for key in [("all", "all"), (row["split"], "all"), (row["split"], row["label_type"])]:
            grouped[key]["width"].append(width)
            grouped[key]["height"].append(height)
            grouped[key]["megapixels"].append(width * height / 1_000_000)
            grouped[key]["images"] += 1

    rows = []
    for (split, label_type), values in sorted(grouped.items()):
        width_stats = describe(values["width"])
        height_stats = describe(values["height"])
        mp_stats = describe(values["megapixels"])
        rows.append(
            {
                "split": split,
                "label_type": label_type,
                "images": values["images"],
                "width_min": width_stats["min"],
                "width_median": width_stats["median"],
                "width_mean": width_stats["mean"],
                "width_max": width_stats["max"],
                "height_min": height_stats["min"],
                "height_median": height_stats["median"],
                "height_mean": height_stats["mean"],
                "height_max": height_stats["max"],
                "megapixels_min": mp_stats["min"],
                "megapixels_median": mp_stats["median"],
                "megapixels_mean": mp_stats["mean"],
                "megapixels_max": mp_stats["max"],
            }
        )
    return rows


def write_all_statistics(metadata_path, coco_path, output_dir):
    metadata_rows = load_metadata(metadata_path)
    bbox_rows = load_bbox_rows(coco_path)

    outputs = [
        (
            "dataset_summary.csv",
            [
                "category",
                "label_type",
                "images",
                "instances",
                "bbox",
                "used_in_train",
                "used_in_val",
                "used_in_test",
            ],
            build_dataset_summary(metadata_rows),
        ),
        (
            "split_category_counts.csv",
            ["split", "category", "label_type", "images", "instances"],
            build_split_category_counts(metadata_rows),
        ),
        (
            "image_resolution_distribution.csv",
            ["width", "height", "aspect_ratio", "megapixels", "images"],
            build_resolution_distribution(metadata_rows),
        ),
        (
            "image_size_summary.csv",
            [
                "split",
                "label_type",
                "images",
                "width_min",
                "width_median",
                "width_mean",
                "width_max",
                "height_min",
                "height_median",
                "height_mean",
                "height_max",
                "megapixels_min",
                "megapixels_median",
                "megapixels_mean",
                "megapixels_max",
            ],
            build_image_summary(metadata_rows),
        ),
        (
            "bbox_instances.csv",
            [
                "image_id",
                "file_name",
                "split",
                "label_type",
                "domain_type",
                "image_width",
                "image_height",
                "bbox_x",
                "bbox_y",
                "bbox_width",
                "bbox_height",
                "bbox_area",
                "bbox_area_ratio",
                "object_size",
            ],
            bbox_rows,
        ),
        (
            "bbox_size_summary.csv",
            [
                "split",
                "label_type",
                "instances",
                "bbox_width_min",
                "bbox_width_p25",
                "bbox_width_median",
                "bbox_width_mean",
                "bbox_width_p75",
                "bbox_width_max",
                "bbox_height_min",
                "bbox_height_p25",
                "bbox_height_median",
                "bbox_height_mean",
                "bbox_height_p75",
                "bbox_height_max",
                "bbox_area_ratio_min",
                "bbox_area_ratio_p25",
                "bbox_area_ratio_median",
                "bbox_area_ratio_mean",
                "bbox_area_ratio_p75",
                "bbox_area_ratio_max",
            ],
            build_bbox_summary(bbox_rows),
        ),
        (
            "object_size_distribution.csv",
            ["split", "category", "label_type", "object_size", "instances"],
            build_object_size_distribution(bbox_rows),
        ),
        (
            "domain_distribution.csv",
            ["domain_type", "category", "label_type", "split", "images", "instances"],
            build_domain_distribution(metadata_rows),
        ),
    ]

    written = []
    for filename, fieldnames, rows in outputs:
        path = output_dir / filename
        write_csv(path, fieldnames, rows)
        written.append(path)
    return written


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export dataset statistics CSV files from audited metadata and COCO annotations."
    )
    parser.add_argument("--metadata", default=Path("data/audit/metadata.csv"), type=Path)
    parser.add_argument("--coco", default=Path("data/audit/coco/all.json"), type=Path)
    parser.add_argument("--output-dir", default=Path("data/audit/statistics"), type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    written = write_all_statistics(args.metadata, args.coco, args.output_dir)
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
