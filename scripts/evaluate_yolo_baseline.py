#!/usr/bin/env python3
"""Run a trained Ultralytics YOLO detector and emit baseline metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from baseline_eval_utils import (
    CLASS_ID,
    SPLITS,
    build_detection_metrics,
    limit_records,
    load_records,
    write_image_scores_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="Trained YOLO weights, usually weights/best.pt.")
    parser.add_argument("--data-root", default=Path("data"), type=Path, help="Dataset root containing group folders.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for predictions and metrics.")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=SPLITS, help="Splits to evaluate.")
    parser.add_argument("--imgsz", default=960, type=int, help="YOLO inference image size.")
    parser.add_argument("--batch", default=16, type=int, help="YOLO inference batch size.")
    parser.add_argument("--conf", default=0.001, type=float, help="Minimum confidence retained for evaluation.")
    parser.add_argument("--iou", default=0.7, type=float, help="NMS IoU used by YOLO inference.")
    parser.add_argument("--max-det", default=300, type=int, help="Maximum detections per image.")
    parser.add_argument("--score-threshold", default=0.25, type=float, help="Fixed score threshold for FPPI/FPR reporting.")
    parser.add_argument("--device", default=None, help="Optional Ultralytics device string.")
    parser.add_argument("--workers", default=8, type=int, help="Ultralytics dataloader workers.")
    parser.add_argument("--max-images", default=None, type=int, help="Optional smoke-test image limit per split.")
    parser.add_argument(
        "--predict-chunk-size",
        default=128,
        type=int,
        help="Number of images passed to each Ultralytics predict call. Use <=0 to predict a split in one call.",
    )
    return parser.parse_args()


def predict_split(model: YOLO, records: list[dict[str, Any]], coco_ids: list[int], args: argparse.Namespace) -> list[dict[str, Any]]:
    image_paths = [str(record["image_path"]) for record in records]
    kwargs: dict[str, Any] = {
        "source": image_paths,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "workers": args.workers,
        "stream": True,
        "verbose": False,
    }
    if args.device is not None:
        kwargs["device"] = args.device

    chunk_size = (
        args.predict_chunk_size
        if args.predict_chunk_size and args.predict_chunk_size > 0
        else max(1, len(image_paths))
    )
    predictions: list[dict[str, Any]] = []
    for chunk_start in range(0, len(image_paths), chunk_size):
        kwargs["source"] = image_paths[chunk_start : chunk_start + chunk_size]
        for result_offset, result in enumerate(model.predict(**kwargs)):
            coco_id = coco_ids[chunk_start + result_offset]
            boxes = result.boxes
            if boxes is None:
                continue
            for box, score, cls_id in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
                if int(cls_id) != 0:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box]
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                if width <= 0 or height <= 0:
                    continue
                predictions.append(
                    {
                        "image_id": coco_id,
                        "category_id": CLASS_ID,
                        "bbox": [x1, y1, width, height],
                        "score": float(score),
                    }
                )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predictions


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_metrics: dict[str, Any] = {}

    for split in args.splits:
        records = limit_records(load_records(args.data_root, split), args.max_images)
        coco_ids = list(range(1, len(records) + 1))
        predictions = predict_split(model, records, coco_ids, args)
        metrics = build_detection_metrics(records, coco_ids, predictions, split, args.score_threshold)
        final_metrics[split] = metrics

        (args.output_dir / f"predictions_{split}.json").write_text(
            json.dumps(predictions, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / f"metrics_{split}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_image_scores_csv(args.output_dir / f"image_scores_{split}.csv", records, coco_ids, predictions)

    (args.output_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
