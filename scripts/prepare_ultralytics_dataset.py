import argparse
import json
import os
import shutil
from pathlib import Path


SPLITS = ("train", "val", "test_standard", "test_robustness")


def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_link_or_copy(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_category_mapping(coco):
    categories = sorted(coco.get("categories", []), key=lambda item: int(item["id"]))
    return {int(category["id"]): idx for idx, category in enumerate(categories)}


def coco_bbox_to_yolo(bbox, width, height):
    x, y, w, h = [float(v) for v in bbox]
    x1 = max(0.0, min(width, x))
    y1 = max(0.0, min(height, y))
    x2 = max(0.0, min(width, x + w))
    y2 = max(0.0, min(height, y + h))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0.0 or bh <= 0.0:
        return None
    xc = (x1 + bw / 2.0) / width
    yc = (y1 + bh / 2.0) / height
    return xc, yc, bw / width, bh / height


def convert_split(data_root, coco_path, output_root, split, mode):
    coco = load_coco(coco_path)
    cat_id_to_class = build_category_mapping(coco)
    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image = {image_id: [] for image_id in images}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    image_list_path = output_root / f"{split}.txt"
    with image_list_path.open("w", encoding="utf-8") as image_list:
        for image_id, image in images.items():
            rel = Path(image["file_name"])
            src = data_root / rel
            if not src.is_file():
                raise FileNotFoundError(f"Missing image referenced by COCO: {src}")

            dst_img = output_root / "images" / split / rel
            dst_label = output_root / "labels" / split / rel.with_suffix(".txt")
            safe_link_or_copy(src, dst_img, mode=mode)
            dst_label.parent.mkdir(parents=True, exist_ok=True)

            lines = []
            width = float(image["width"])
            height = float(image["height"])
            for ann in anns_by_image.get(image_id, []):
                if ann.get("iscrowd", 0):
                    continue
                category_id = int(ann["category_id"])
                if category_id not in cat_id_to_class:
                    raise KeyError(f"Annotation category_id={category_id} missing from categories in {coco_path}")
                converted = coco_bbox_to_yolo(ann["bbox"], width, height)
                if converted is None:
                    continue
                xc, yc, bw, bh = converted
                cls = cat_id_to_class[category_id]
                lines.append(f"{cls} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")

            dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            image_list.write(str(dst_img.resolve()) + "\n")

    return len(images), sum(len(v) for v in anns_by_image.values())


def write_yaml(output_root):
    yaml_path = output_root / "nine_dash_line.yaml"
    text = "\n".join(
        [
            f"path: {output_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test_standard",
            "names:",
            "  0: nine_dash_line",
            "",
        ]
    )
    yaml_path.write_text(text, encoding="utf-8")
    return yaml_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert audited COCO splits to Ultralytics YOLO/RT-DETR format."
    )
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument("--coco-dir", default="data/audit/coco", type=Path)
    parser.add_argument("--output-root", default="data/ultralytics_stage2", type=Path)
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="hardlink saves disk space and falls back to copy when unavailable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in SPLITS:
        coco_path = args.coco_dir / f"{split}.json"
        images, annotations = convert_split(
            data_root=args.data_root,
            coco_path=coco_path,
            output_root=args.output_root,
            split=split,
            mode=args.mode,
        )
        summary[split] = {"images": images, "annotations": annotations}

    yaml_path = write_yaml(args.output_root)
    summary_path = args.output_root / "conversion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {yaml_path}")
    print(f"Wrote {summary_path}")
    for split, row in summary.items():
        print(f"{split}: {row['images']} images, {row['annotations']} boxes")


if __name__ == "__main__":
    main()
