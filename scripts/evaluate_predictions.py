#!/usr/bin/env python3
"""Evaluate COCO-style baseline predictions with object and moderation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline_eval_utils import (
    SPLITS,
    build_detection_metrics,
    limit_records,
    load_records,
    sanitize_predictions,
    write_image_scores_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=Path("data"), type=Path, help="Dataset root containing group folders.")
    parser.add_argument("--split", default="test", choices=SPLITS, help="Dataset split to evaluate.")
    parser.add_argument("--predictions", required=True, type=Path, help="COCO-style predictions JSON.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for metrics and image score CSV.")
    parser.add_argument("--score-threshold", default=0.25, type=float, help="Fixed score threshold for FPPI/FPR reporting.")
    parser.add_argument("--max-images", default=None, type=int, help="Optional smoke-test image limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = limit_records(load_records(args.data_root, args.split), args.max_images)
    coco_ids = list(range(1, len(records) + 1))

    with args.predictions.open(encoding="utf-8") as handle:
        raw_predictions = json.load(handle)
    if not isinstance(raw_predictions, list):
        raise ValueError(f"{args.predictions} must contain a JSON list")

    predictions = sanitize_predictions(raw_predictions, coco_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_detection_metrics(records, coco_ids, predictions, args.split, args.score_threshold)

    (args.output_dir / f"predictions_{args.split}.json").write_text(
        json.dumps(predictions, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / f"metrics_{args.split}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_image_scores_csv(args.output_dir / f"image_scores_{args.split}.csv", records, coco_ids, predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
