#!/usr/bin/env python3
"""Shared evaluation utilities for clean baseline runs."""

from __future__ import annotations

import csv
import contextlib
import io
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ModuleNotFoundError:  # Image-level metric tests do not require COCO.
    COCO = None  # type: ignore[assignment,misc]
    COCOeval = None  # type: ignore[assignment,misc]


CLASS_ID = 1
CLASS_NAME = "nine_dash_line"
SPLITS = ("train", "val", "test")
RELATIVE_SCALE_BUCKETS = {
    "tiny": (0.0, 0.01),
    "small": (0.01, 0.05),
    "medium": (0.05, 0.25),
    "large": (0.25, math.inf),
}


def require_pycocotools() -> None:
    if COCO is None or COCOeval is None:
        raise ModuleNotFoundError("pycocotools is required for COCO object-detection metrics; install requirements.txt")


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_annotation(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def normalized_record(record: dict[str, Any], annotation_path: Path, group: str) -> dict[str, Any]:
    file_name = str(record.get("file_name", ""))
    return {
        "group": group,
        "split": str(record.get("split", "")),
        "image_id": str(record.get("image_id", "")),
        "file_name": file_name,
        "image_path": annotation_path.parent / file_name,
        "width": int(record.get("width") or 0),
        "height": int(record.get("height") or 0),
        "objects": record.get("objects") or [],
    }


def load_records(data_root: Path, split: str) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")

    records: list[dict[str, Any]] = []
    for annotation_path in sorted(data_root.glob("*/annotations.json")):
        group = annotation_path.parent.name
        for record in load_annotation(annotation_path):
            if str(record.get("split", "")) != split:
                continue
            item = normalized_record(record, annotation_path, group)
            if not item["image_path"].is_file():
                raise FileNotFoundError(f"Missing image referenced by annotation: {item['image_path']}")
            records.append(item)
    records.sort(key=lambda item: (item["group"], item["file_name"]))
    return records


def limit_records(records: list[dict[str, Any]], max_images: int | None) -> list[dict[str, Any]]:
    if max_images is None or len(records) <= max_images:
        return records

    positives = [record for record in records if record["objects"]]
    negatives = [record for record in records if not record["objects"]]
    selected: list[dict[str, Any]] = []

    if positives and negatives and max_images >= 2:
        positive_target = min(len(positives), max(1, max_images // 2))
        negative_target = min(len(negatives), max_images - positive_target)
        selected.extend(positives[:positive_target])
        selected.extend(negatives[:negative_target])
    else:
        selected.extend(records[:max_images])

    if len(selected) < max_images:
        selected_ids = {id(record) for record in selected}
        for record in records:
            if id(record) in selected_ids:
                continue
            selected.append(record)
            if len(selected) == max_images:
                break

    selected.sort(key=lambda item: (item["group"], item["file_name"]))
    return selected


def bbox_xywh_to_xyxy(bbox: Any, image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x, y, width, height = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    x1 = clip(x, 0.0, float(image_width))
    y1 = clip(y, 0.0, float(image_height))
    x2 = clip(x + width, 0.0, float(image_width))
    y2 = clip(y + height, 0.0, float(image_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def valid_prediction(prediction: dict[str, Any], valid_image_ids: set[int]) -> dict[str, Any] | None:
    try:
        image_id = int(prediction["image_id"])
        category_id = int(prediction.get("category_id", CLASS_ID))
        bbox = [float(value) for value in prediction["bbox"]]
        score = float(prediction["score"])
    except (KeyError, TypeError, ValueError):
        return None

    if image_id not in valid_image_ids or category_id != CLASS_ID or len(bbox) != 4:
        return None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    if any(not math.isfinite(value) for value in bbox) or bbox[2] <= 0 or bbox[3] <= 0:
        return None
    return {"image_id": image_id, "category_id": CLASS_ID, "bbox": bbox, "score": score}


def sanitize_predictions(predictions: Iterable[dict[str, Any]], coco_ids: list[int]) -> list[dict[str, Any]]:
    valid_image_ids = set(coco_ids)
    clean: list[dict[str, Any]] = []
    for prediction in predictions:
        item = valid_prediction(prediction, valid_image_ids)
        if item is not None:
            clean.append(item)
    return clean


def coco_dataset_from_records(records: list[dict[str, Any]], coco_ids: list[int] | None = None) -> COCO:
    require_pycocotools()
    if coco_ids is None:
        coco_ids = list(range(1, len(records) + 1))

    dataset: dict[str, Any] = {
        "info": {"description": "nine-dash-line baseline evaluation dataset"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": CLASS_ID, "name": CLASS_NAME}],
    }
    annotation_id = 1
    for record, coco_id in zip(records, coco_ids):
        dataset["images"].append(
            {
                "id": coco_id,
                "file_name": f"{record['group']}/{record['file_name']}",
                "width": record["width"],
                "height": record["height"],
            }
        )
        for obj in record["objects"]:
            if not isinstance(obj, dict):
                continue
            box = bbox_xywh_to_xyxy(obj.get("bbox"), record["width"], record["height"])
            if box is None:
                continue
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            dataset["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": coco_id,
                    "category_id": CLASS_ID,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco = COCO()
    coco.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def coco_metrics_from_stats(stats: np.ndarray) -> dict[str, float | None]:
    names = [
        "mAP",
        "mAP50",
        "mAP75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    return {name: (float(value) if float(value) >= 0.0 else None) for name, value in zip(names, stats.tolist())}


def empty_coco_metrics(coco_gt: COCO) -> dict[str, float | None]:
    """COCO metrics for zero detections, preserving unavailable area buckets."""

    areas = [float(annotation["area"]) for annotation in coco_gt.dataset["annotations"]]
    if not areas:
        return {name: None for name in coco_metrics_from_stats(np.zeros(12)).keys()}

    has_small = any(area <= 32.0**2 for area in areas)
    has_medium = any(32.0**2 <= area <= 96.0**2 for area in areas)
    has_large = any(area >= 96.0**2 for area in areas)
    return {
        "mAP": 0.0,
        "mAP50": 0.0,
        "mAP75": 0.0,
        "mAP_small": 0.0 if has_small else None,
        "mAP_medium": 0.0 if has_medium else None,
        "mAP_large": 0.0 if has_large else None,
        "AR1": 0.0,
        "AR10": 0.0,
        "AR100": 0.0,
        "AR_small": 0.0 if has_small else None,
        "AR_medium": 0.0 if has_medium else None,
        "AR_large": 0.0 if has_large else None,
    }


def run_coco_eval(
    records: list[dict[str, Any]],
    coco_ids: list[int],
    predictions: list[dict[str, Any]],
) -> dict[str, float | None]:
    require_pycocotools()
    if len(records) != len(coco_ids) or len(set(coco_ids)) != len(coco_ids):
        raise ValueError("records and coco_ids must have equal lengths and unique COCO image IDs")
    coco_gt = coco_dataset_from_records(records, coco_ids)
    gt_count = len(coco_gt.dataset["annotations"])
    if not predictions or gt_count == 0:
        return empty_coco_metrics(coco_gt)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = coco_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return coco_metrics_from_stats(coco_eval.stats)


def bbox_area_ratio(obj: dict[str, Any], record: dict[str, Any]) -> float | None:
    box = bbox_xywh_to_xyxy(obj.get("bbox"), record["width"], record["height"])
    image_area = float(record["width"] * record["height"])
    if box is None or image_area <= 0:
        return None
    x1, y1, x2, y2 = box
    return ((x2 - x1) * (y2 - y1)) / image_area


def valid_object_count(record: dict[str, Any]) -> int:
    return sum(
        bbox_xywh_to_xyxy(obj.get("bbox"), record["width"], record["height"]) is not None
        for obj in record["objects"]
        if isinstance(obj, dict)
    )


def scale_bucket_for_object(obj: dict[str, Any], record: dict[str, Any]) -> str | None:
    ratio = bbox_area_ratio(obj, record)
    if ratio is None:
        return None
    for name, (low, high) in RELATIVE_SCALE_BUCKETS.items():
        if low <= ratio < high:
            return name
    return None


def predictions_for_scale_bucket(
    records: list[dict[str, Any]],
    coco_ids: list[int],
    predictions: list[dict[str, Any]],
    bucket: str,
) -> list[dict[str, Any]]:
    """Keep detections whose clipped relative area belongs to the bucket."""

    record_by_id = dict(zip(coco_ids, records))
    filtered: list[dict[str, Any]] = []
    for prediction in predictions:
        record = record_by_id[int(prediction["image_id"])]
        box = bbox_xywh_to_xyxy(prediction["bbox"], record["width"], record["height"])
        image_area = float(record["width"] * record["height"])
        if box is None or image_area <= 0:
            continue
        x1, y1, x2, y2 = box
        ratio = ((x2 - x1) * (y2 - y1)) / image_area
        low, high = RELATIVE_SCALE_BUCKETS[bucket]
        if low <= ratio < high:
            filtered.append(prediction)
    return filtered


def records_for_scale_bucket(records: list[dict[str, Any]], bucket: str) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for record in records:
        objects = [
            obj
            for obj in record["objects"]
            if isinstance(obj, dict) and scale_bucket_for_object(obj, record) == bucket
        ]
        item = dict(record)
        item["objects"] = objects
        filtered.append(item)
    return filtered


def relative_scale_metrics(
    records: list[dict[str, Any]],
    coco_ids: list[int],
    predictions: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, float | None]]:
    nested: dict[str, dict[str, Any]] = {}
    flat: dict[str, float | None] = {}

    for bucket in RELATIVE_SCALE_BUCKETS:
        bucket_records = records_for_scale_bucket(records, bucket)
        object_count = sum(len(record["objects"]) for record in bucket_records)
        image_count = sum(1 for record in bucket_records if record["objects"])
        if object_count == 0:
            metrics: dict[str, Any] = {"objects": 0, "positive_images": 0, "detections": 0}
            for name in ("mAP", "mAP50", "mAP75", "AR100"):
                metrics[name] = None
        else:
            bucket_predictions = predictions_for_scale_bucket(records, coco_ids, predictions, bucket)
            metrics = run_coco_eval(bucket_records, coco_ids, bucket_predictions)
            metrics["objects"] = object_count
            metrics["positive_images"] = image_count
            metrics["detections"] = len(bucket_predictions)

        nested[bucket] = metrics
        flat[f"mAP_{bucket}_rel"] = metrics.get("mAP")
        flat[f"mAP50_{bucket}_rel"] = metrics.get("mAP50")
        flat[f"mAP75_{bucket}_rel"] = metrics.get("mAP75")
        flat[f"AR100_{bucket}_rel"] = metrics.get("AR100")

    flat["mAP_tiny"] = nested["tiny"].get("mAP")
    return nested, flat


def image_scores_from_predictions(
    records: list[dict[str, Any]],
    coco_ids: list[int],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    max_scores = {coco_id: 0.0 for coco_id in coco_ids}
    detection_counts = Counter({coco_id: 0 for coco_id in coco_ids})
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        score = float(prediction["score"])
        if image_id not in max_scores:
            continue
        max_scores[image_id] = max(max_scores[image_id], score)
        detection_counts[image_id] += 1

    rows: list[dict[str, Any]] = []
    for record, coco_id in zip(records, coco_ids):
        rows.append(
            {
                "image_id": coco_id,
                "source_image_id": record["image_id"],
                "group": record["group"],
                "file_name": record["file_name"],
                "image_path": str(record["image_path"]),
                "label": 1 if valid_object_count(record) else 0,
                "object_count": valid_object_count(record),
                "score": max_scores[coco_id],
                "detections": detection_counts[coco_id],
            }
        )
    return rows


def binary_counts(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    negative_by_group: Counter[str] = Counter()
    fp_by_group: Counter[str] = Counter()

    for row in rows:
        label = int(row["label"])
        # A no-detection image has score 0 for ranking, but is never positive at
        # threshold 0 because there is no actual detection to threshold.
        predicted = int(row.get("detections", 0)) > 0 and float(row["score"]) >= threshold
        if label:
            if predicted:
                tp += 1
            else:
                fn += 1
        else:
            negative_by_group[str(row["group"])] += 1
            if predicted:
                fp += 1
                fp_by_group[str(row["group"])] += 1
            else:
                tn += 1

    positives = tp + fn
    negatives = fp + tn
    tpr = tp / positives if positives else None
    fpr = fp / negatives if negatives else None
    by_group = {
        group: {
            "negative_images": int(negative_by_group[group]),
            "false_positive_images": int(fp_by_group[group]),
            "fpr": fp_by_group[group] / negative_by_group[group] if negative_by_group[group] else None,
        }
        for group in sorted(negative_by_group)
    }
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "by_negative_group": by_group,
    }


def candidate_thresholds(rows: list[dict[str, Any]]) -> list[float]:
    scores = sorted({float(row["score"]) for row in rows}, reverse=True)
    no_positive_threshold = (max(scores) + 1.0) if scores else 1.0
    thresholds = [no_positive_threshold] + scores + [0.0]
    deduped: list[float] = []
    seen: set[float] = set()
    for threshold in thresholds:
        if threshold in seen:
            continue
        seen.add(threshold)
        deduped.append(threshold)
    return deduped


def auroc_score(rows: list[dict[str, Any]]) -> float | None:
    positives = sum(1 for row in rows if int(row["label"]) == 1)
    negatives = len(rows) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted((float(row["score"]), int(row["label"])) for row in rows)
    rank_sum_pos = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum_pos += average_rank * sum(label for _, label in ordered[index:end])
        index = end

    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(rows: list[dict[str, Any]]) -> float | None:
    positives = sum(1 for row in rows if int(row["label"]) == 1)
    if positives == 0:
        return None

    # Aggregate equal-score samples before integrating the PR curve. Ranking
    # positives ahead of negatives inside a tie would make AP depend on file order.
    counts_by_score: dict[float, list[int]] = {}
    for row in rows:
        score = float(row["score"])
        counts = counts_by_score.setdefault(score, [0, 0])
        counts[0] += int(row["label"] == 1)
        counts[1] += 1

    true_positives = 0
    predicted_positives = 0
    ap = 0.0
    for score in sorted(counts_by_score, reverse=True):
        positive_count, total_count = counts_by_score[score]
        true_positives += positive_count
        predicted_positives += total_count
        precision = true_positives / predicted_positives
        ap += (positive_count / positives) * precision
    return ap


def operation_at_target_tpr(rows: list[dict[str, Any]], target_tpr: float) -> dict[str, Any]:
    candidates = [binary_counts(rows, threshold) for threshold in candidate_thresholds(rows)]
    feasible = [item for item in candidates if item["tpr"] is not None and item["tpr"] >= target_tpr]
    if not feasible:
        return {"target_tpr": target_tpr, "threshold": None, "tpr": None, "fpr": None, "by_negative_group": {}}
    best = min(feasible, key=lambda item: (float("inf") if item["fpr"] is None else item["fpr"], -float(item["threshold"])))
    best["target_tpr"] = target_tpr
    return best


def recall_at_fpr(rows: list[dict[str, Any]], max_fpr: float) -> dict[str, Any]:
    candidates = [binary_counts(rows, threshold) for threshold in candidate_thresholds(rows)]
    feasible = [item for item in candidates if item["fpr"] is not None and item["fpr"] <= max_fpr]
    if not feasible:
        return {"max_fpr": max_fpr, "threshold": None, "recall": None, "fpr": None}
    best = max(feasible, key=lambda item: (-1.0 if item["tpr"] is None else item["tpr"], item["threshold"]))
    return {
        "max_fpr": max_fpr,
        "threshold": best["threshold"],
        "recall": best["tpr"],
        "fpr": best["fpr"],
        "tp": best["tp"],
        "fp": best["fp"],
        "tn": best["tn"],
        "fn": best["fn"],
    }


def group_fpr(operation: dict[str, Any], group: str) -> float | None:
    group_metrics = operation.get("by_negative_group", {}).get(group, {})
    return group_metrics.get("fpr")


def image_level_metrics(
    records: list[dict[str, Any]],
    coco_ids: list[int],
    predictions: list[dict[str, Any]],
    fixed_threshold: float,
    target_tpr: float = 0.95,
) -> tuple[dict[str, Any], dict[str, float | None]]:
    rows = image_scores_from_predictions(records, coco_ids, predictions)
    positives = sum(1 for row in rows if int(row["label"]) == 1)
    negatives = len(rows) - positives
    op_95 = operation_at_target_tpr(rows, target_tpr)
    fixed = binary_counts(rows, fixed_threshold)
    recall_1 = recall_at_fpr(rows, 0.01)
    recall_5 = recall_at_fpr(rows, 0.05)

    nested = {
        "positive_images": positives,
        "negative_images": negatives,
        "auroc": auroc_score(rows),
        "average_precision": average_precision(rows),
        "fpr_at_target_tpr": op_95,
        "recall_at_fpr": {
            "0.01": recall_1,
            "0.05": recall_5,
        },
        "fixed_threshold": fixed,
    }
    flat = {
        "image_AUROC": nested["auroc"],
        "image_AP": nested["average_precision"],
        "FPR@95TPR": op_95.get("fpr"),
        "threshold@95TPR": op_95.get("threshold"),
        "TPR@95TPR": op_95.get("tpr"),
        "Recall@FPR=1%": recall_1.get("recall"),
        "Recall@FPR=5%": recall_5.get("recall"),
        "in_domain_FPR@95TPR": group_fpr(op_95, "negative_in_domain"),
        "out_domain_FPR@95TPR": group_fpr(op_95, "negative_out_domain"),
        "in_domain_FPR@fixed": group_fpr(fixed, "negative_in_domain"),
        "out_domain_FPR@fixed": group_fpr(fixed, "negative_out_domain"),
    }
    return nested, flat


def false_positive_detection_metrics(
    records: list[dict[str, Any]],
    coco_ids: list[int],
    predictions: list[dict[str, Any]],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    record_by_id = dict(zip(coco_ids, records))
    negative_images_by_group = Counter(record["group"] for record in records if not valid_object_count(record))
    fp_by_group: Counter[str] = Counter()
    positive_predictions_by_group: Counter[str] = Counter()

    for prediction in predictions:
        if float(prediction["score"]) < threshold:
            continue
        record = record_by_id[int(prediction["image_id"])]
        positive_predictions_by_group[record["group"]] += 1
        if not valid_object_count(record):
            fp_by_group[record["group"]] += 1

    total_negative_images = sum(negative_images_by_group.values())
    total_false_positives = sum(fp_by_group.values())
    metrics = {
        "total_negative_images": total_negative_images,
        "total_false_positives": total_false_positives,
        "fppi": total_false_positives / total_negative_images if total_negative_images else 0.0,
        "by_group": {
            group: {
                "negative_images": negative_images_by_group[group],
                "false_positives": fp_by_group[group],
                "fppi": fp_by_group[group] / negative_images_by_group[group] if negative_images_by_group[group] else 0.0,
            }
            for group in sorted(negative_images_by_group)
        },
    }
    return metrics, dict(sorted(positive_predictions_by_group.items()))


def build_detection_metrics(
    records: list[dict[str, Any]],
    coco_ids: list[int],
    predictions: list[dict[str, Any]],
    split: str,
    fixed_threshold: float,
) -> dict[str, Any]:
    if len(records) != len(coco_ids) or len(set(coco_ids)) != len(coco_ids):
        raise ValueError("records and coco_ids must have equal lengths and unique COCO image IDs")
    clean_predictions = sanitize_predictions(predictions, coco_ids)
    metrics: dict[str, Any] = run_coco_eval(records, coco_ids, clean_predictions)
    scale_nested, scale_flat = relative_scale_metrics(records, coco_ids, clean_predictions)
    image_nested, image_flat = image_level_metrics(records, coco_ids, clean_predictions, fixed_threshold)
    fp_metrics, group_prediction_counts = false_positive_detection_metrics(records, coco_ids, clean_predictions, fixed_threshold)

    metrics.update(scale_flat)
    metrics.update(image_flat)
    metrics["split"] = split
    metrics["images"] = len(records)
    metrics["ground_truth_objects"] = sum(valid_object_count(record) for record in records)
    metrics["input_predictions"] = len(predictions)
    metrics["detections"] = len(clean_predictions)
    metrics["invalid_predictions_dropped"] = len(predictions) - len(clean_predictions)
    metrics["fppi_threshold"] = fixed_threshold
    metrics["false_positives_on_negatives"] = fp_metrics
    metrics["predictions_above_fppi_threshold_by_group"] = group_prediction_counts
    metrics["relative_scale_metrics"] = scale_nested
    metrics["image_level"] = image_nested
    return metrics


def write_image_scores_csv(
    path: Path,
    records: list[dict[str, Any]],
    coco_ids: list[int],
    predictions: list[dict[str, Any]],
) -> None:
    rows = image_scores_from_predictions(records, coco_ids, predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "image_id",
            "source_image_id",
            "group",
            "file_name",
            "image_path",
            "label",
            "object_count",
            "score",
            "detections",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
