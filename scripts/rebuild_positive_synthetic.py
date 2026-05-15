import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


SYNTHETIC_PATTERNS = (
    re.compile(r"^(nine_dash_\d+)(?:_jpg)?(?:\.rf\.[^.]+)?_synth_(\d+)\.jpg$"),
    re.compile(r"^(nine_dash_\d+)_synth_(\d+)(?:_jpg)?(?:\.rf\.[^.]+)?\.jpg$"),
)


def parse_synthetic_key(file_name):
    for pattern in SYNTHETIC_PATTERNS:
        match = pattern.match(file_name)
        if match:
            parent_id, synth_idx = match.groups()
            return parent_id, int(synth_idx), f"{parent_id}_synth_{int(synth_idx)}"
    return "", -1, ""


def load_parent_splits(metadata_path):
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        row["image_id"]: row["split"]
        for row in rows
        if row.get("label_type") == "positive_real"
    }


def load_coco_synthetic(coco_path):
    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    annotations_by_image_id = defaultdict(list)
    for ann in coco.get("annotations", []):
        if int(ann.get("category_id", -1)) == 1:
            annotations_by_image_id[int(ann["image_id"])].append(ann)

    images_by_key = {}
    for image in coco.get("images", []):
        candidates = [image.get("file_name", "")]
        extra_name = (image.get("extra") or {}).get("name")
        if extra_name:
            candidates.insert(0, extra_name)

        parsed = None
        for candidate in candidates:
            parent_id, synth_idx, key = parse_synthetic_key(candidate)
            if key:
                parsed = parent_id, synth_idx, key
                break

        if not parsed:
            continue

        anns = annotations_by_image_id.get(int(image["id"]), [])
        if not anns:
            continue

        parent_id, synth_idx, key = parsed
        images_by_key[key] = {
            "parent_id": parent_id,
            "synth_idx": synth_idx,
            "width": int(image["width"]),
            "height": int(image["height"]),
            "annotations": anns,
            "coco_file_name": image.get("file_name", ""),
            "coco_extra_name": extra_name or "",
        }

    return images_by_key


def natural_synthetic_sort(item):
    source_path, parent_id, synth_idx, key = item
    parent_num = int(parent_id.rsplit("_", 1)[1])
    return parent_num, synth_idx, source_path.name


def rebuild(synthetic_dir, coco_path, metadata_path, output_dir):
    parent_splits = load_parent_splits(metadata_path)
    coco_synthetic = load_coco_synthetic(coco_path)

    source_files = []
    skipped = []
    for path in sorted(synthetic_dir.iterdir()):
        if not path.is_file():
            continue

        parent_id, synth_idx, key = parse_synthetic_key(path.name)
        if not key:
            skipped.append({"file_name": path.name, "reason": "unparseable_synthetic_filename"})
            continue

        if key not in coco_synthetic:
            skipped.append({"file_name": path.name, "reason": "missing_bbox_in_coco_train", "parent_id": parent_id})
            continue

        parent_split = parent_splits.get(parent_id)
        if not parent_split:
            skipped.append({"file_name": path.name, "reason": "parent_missing_in_positive_real", "parent_id": parent_id})
            continue

        if parent_split != "train":
            skipped.append(
                {
                    "file_name": path.name,
                    "reason": "parent_not_train_leakage",
                    "parent_id": parent_id,
                    "parent_split": parent_split,
                }
            )
            continue

        source_files.append((path, parent_id, synth_idx, key))

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, item in enumerate(sorted(source_files, key=natural_synthetic_sort), start=1):
        source_path, parent_id, synth_idx, key = item
        new_file_name = f"positive_synthetic_{index}.jpg"
        shutil.copy2(source_path, output_dir / new_file_name)

        coco_image = coco_synthetic[key]
        objects = []
        for ann in coco_image["annotations"]:
            objects.append(
                {
                    "category": "nine_dash_line",
                    "bbox": [round(float(value), 4) for value in ann["bbox"]],
                    "bbox_type": "visible_instance",
                    "visibility": "full",
                    "quality": "high",
                }
            )

        records.append(
            {
                "image_id": f"positive_synthetic_{index}",
                "file_name": new_file_name,
                "width": coco_image["width"],
                "height": coco_image["height"],
                "split": "",
                "source_type": "positive_synthetic",
                "domain_type": "map_sea",
                "source_id": parent_id,
                "synthetic_parent_id": parent_id,
                "parent_provenance_status": "known",
                "parent_provenance_field": "filename",
                "original_file_name": source_path.name,
                "coco_file_name": coco_image["coco_file_name"],
                "coco_extra_name": coco_image["coco_extra_name"],
                "objects": objects,
            }
        )

    with (output_dir / "annotations.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")

    summary = {
        "synthetic_source_files": len([p for p in synthetic_dir.iterdir() if p.is_file()]),
        "coco_synthetic_with_bbox": len(coco_synthetic),
        "kept_train_safe": len(records),
        "skipped": len(skipped),
        "skipped_by_reason": {},
    }
    for item in skipped:
        reason = item["reason"]
        summary["skipped_by_reason"][reason] = summary["skipped_by_reason"].get(reason, 0) + 1

    report = {"summary": summary, "skipped": skipped}
    with (output_dir / "rebuild_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Rebuild train-safe positive_synthetic dataset.")
    parser.add_argument("--synthetic-dir", type=Path, default=Path("synthetic"))
    parser.add_argument(
        "--coco-train",
        type=Path,
        default=Path("nine-dash-line-coco 2.v5i.coco") / "train" / "_annotations.coco.json",
    )
    parser.add_argument("--metadata", type=Path, default=Path("data") / "audit" / "metadata.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "positive_synthetic")
    args = parser.parse_args()

    report = rebuild(args.synthetic_dir, args.coco_train, args.metadata, args.output_dir)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
