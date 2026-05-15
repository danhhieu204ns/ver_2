import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps


LABEL_ORDER = (
    "positive_real",
    "in_domain_neg",
    "out_domain_neg",
    "hard_neg",
    "synthetic_pos",
)
LABEL_NAMES = {
    "positive_real": "Positive real",
    "in_domain_neg": "In-domain negative",
    "out_domain_neg": "Out-domain negative",
    "hard_neg": "Hard negative",
    "synthetic_pos": "Synthetic positive",
}
LABEL_COLORS = {
    "positive_real": "#2f6f9f",
    "in_domain_neg": "#5b8c5a",
    "out_domain_neg": "#8a6f3d",
    "hard_neg": "#b35c44",
    "synthetic_pos": "#7a5aa6",
}


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def ensure_output(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)


def save_figure(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_image_resolution(rows, output_dir):
    top_rows = sorted(rows, key=lambda row: as_int(row["images"]), reverse=True)[:20]
    labels = [f"{row['width']}x{row['height']}" for row in top_rows]
    values = [as_int(row["images"]) for row in top_rows]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(labels, values, color="#4e79a7")
    ax.set_title("Distribution of Image Resolution")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("# Images")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir / "01_image_resolution_distribution.png")


def plot_bbox_width_height(rows, output_dir):
    widths = [as_float(row["bbox_width"]) for row in rows]
    heights = [as_float(row["bbox_height"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(widths, bins=40, color="#4e79a7", edgecolor="white")
    axes[0].set_title("BBox Width")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("# Instances")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].hist(heights, bins=40, color="#59a14f", edgecolor="white")
    axes[1].set_title("BBox Height")
    axes[1].set_xlabel("Height (px)")
    axes[1].set_ylabel("# Instances")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Distribution of BBox Width and Height")
    save_figure(fig, output_dir / "02_bbox_width_height_distribution.png")


def plot_bbox_area_ratio(rows, output_dir):
    ratios = [as_float(row["bbox_area_ratio"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.hist(ratios, bins=50, color="#f28e2b", edgecolor="white")
    ax.axvline(0.01, color="#444444", linestyle="--", linewidth=1.2, label="small < 0.01")
    ax.axvline(0.05, color="#777777", linestyle=":", linewidth=1.5, label="medium < 0.05")
    ax.set_title("Distribution of BBox Area Ratio")
    ax.set_xlabel("BBox area / image area")
    ax.set_ylabel("# Instances")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_dir / "03_bbox_area_ratio_distribution.png")


def plot_object_size(rows, output_dir):
    counts = defaultdict(int)
    for row in rows:
        counts[(row["label_type"], row["object_size"])] += as_int(row["instances"])

    sizes = ("small", "medium", "large")
    labels = ["Positive real", "Synthetic positive"]
    x = range(len(labels))
    bottoms = [0, 0]
    colors = {"small": "#76b7b2", "medium": "#edc948", "large": "#e15759"}

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for size in sizes:
        values = [
            counts[("positive_real", size)],
            counts[("synthetic_pos", size)],
        ]
        ax.bar(x, values, bottom=bottoms, label=size.title(), color=colors[size])
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_title("Distribution of Object Size")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("# Instances")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_dir / "04_object_size_distribution.png")


def plot_domain_distribution(rows, output_dir):
    counts = defaultdict(int)
    for row in rows:
        counts[(row["domain_type"], row["label_type"])] += as_int(row["images"])

    domains = sorted({domain for domain, _ in counts})
    x = range(len(domains))
    bottoms = [0] * len(domains)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label_type in LABEL_ORDER:
        values = [counts[(domain, label_type)] for domain in domains]
        if not any(values):
            continue
        ax.bar(
            x,
            values,
            bottom=bottoms,
            label=LABEL_NAMES[label_type],
            color=LABEL_COLORS[label_type],
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_title("Domain Distribution")
    ax.set_xticks(list(x), domains)
    ax.set_xlabel("Domain")
    ax.set_ylabel("# Images")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_dir / "05_domain_distribution.png")


def build_bbox_lookup(bbox_rows):
    lookup = defaultdict(list)
    for row in bbox_rows:
        lookup[row["file_name"]].append(
            (
                as_float(row["bbox_x"]),
                as_float(row["bbox_y"]),
                as_float(row["bbox_width"]),
                as_float(row["bbox_height"]),
            )
        )
    return lookup


def choose_gallery_rows(metadata_rows):
    chosen = []
    for label_type in LABEL_ORDER:
        candidates = [
            row
            for row in metadata_rows
            if row["label_type"] == label_type and row.get("split") in {"train", "val", "test_standard"}
        ]
        candidates = sorted(candidates, key=lambda row: (row.get("split", ""), row.get("file_name", "")))
        if candidates:
            chosen.append(candidates[0])
    return chosen


def show_image_with_optional_boxes(ax, image_path, boxes, title):
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        ax.imshow(image)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    for x, y, w, h in boxes:
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="#ffcc33", facecolor="none")
        ax.add_patch(rect)


def plot_example_gallery(metadata_rows, bbox_rows, data_root, output_dir):
    bbox_lookup = build_bbox_lookup(bbox_rows)
    gallery_rows = choose_gallery_rows(metadata_rows)

    fig, axes = plt.subplots(1, len(gallery_rows), figsize=(15, 3.6))
    if len(gallery_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, gallery_rows):
        relative_path = row["relative_path"]
        image_path = data_root / relative_path
        boxes = bbox_lookup.get(relative_path, [])
        show_image_with_optional_boxes(ax, image_path, boxes, LABEL_NAMES[row["label_type"]])

    fig.suptitle("Example Gallery")
    save_figure(fig, output_dir / "06_example_gallery.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset statistics figures.")
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--metadata", default=Path("data/audit/metadata.csv"), type=Path)
    parser.add_argument("--statistics-dir", default=Path("data/audit/statistics"), type=Path)
    parser.add_argument("--output-dir", default=Path("data/audit/figures"), type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output(args.output_dir)

    metadata_rows = read_csv(args.metadata)
    resolution_rows = read_csv(args.statistics_dir / "image_resolution_distribution.csv")
    bbox_rows = read_csv(args.statistics_dir / "bbox_instances.csv")
    object_size_rows = read_csv(args.statistics_dir / "object_size_distribution.csv")
    domain_rows = read_csv(args.statistics_dir / "domain_distribution.csv")

    plot_image_resolution(resolution_rows, args.output_dir)
    plot_bbox_width_height(bbox_rows, args.output_dir)
    plot_bbox_area_ratio(bbox_rows, args.output_dir)
    plot_object_size(object_size_rows, args.output_dir)
    plot_domain_distribution(domain_rows, args.output_dir)
    plot_example_gallery(metadata_rows, bbox_rows, args.data_root, args.output_dir)

    for path in sorted(args.output_dir.glob("*.png")):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
