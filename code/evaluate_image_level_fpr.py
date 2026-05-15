import argparse
import json
from pathlib import Path

from detection_common import (
    choose_threshold_for_target_fpr,
    load_coco_dict,
    summarize_image_level,
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute image-level moderation metrics from COCO-format detections."
    )
    parser.add_argument("--val-coco", default="data/audit/coco/val.json", type=Path)
    parser.add_argument("--val-preds", required=True, type=Path)
    parser.add_argument("--test-coco", default="data/audit/coco/test_robustness.json", type=Path)
    parser.add_argument("--test-preds", required=True, type=Path)
    parser.add_argument("--target-val-fpr", type=float, default=0.01)
    parser.add_argument("--output", default="results/stage2_baselines/image_level_metrics.json", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    val_coco = load_coco_dict(args.val_coco)
    val_preds = load_json(args.val_preds)
    test_coco = load_coco_dict(args.test_coco)
    test_preds = load_json(args.test_preds)

    threshold = choose_threshold_for_target_fpr(val_coco, val_preds, args.target_val_fpr)
    result = {
        "threshold_source": "validation",
        "target_val_fpr": args.target_val_fpr,
        "selected_threshold": threshold,
        "val": summarize_image_level(val_coco, val_preds, threshold),
        "test": summarize_image_level(test_coco, test_preds, threshold),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Selected threshold on val: {threshold:.6f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
