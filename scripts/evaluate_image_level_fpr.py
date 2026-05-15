import argparse
import json
from pathlib import Path


NEGATIVE_LABEL_TYPES = {"in_domain_neg", "out_domain_neg", "hard_neg"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def image_key(image):
    return str(image.get("id"))


def max_score_by_image(predictions):
    scores = {}
    for pred in predictions:
        img_id = str(pred.get("image_id"))
        score = float(pred.get("score", 0.0))
        scores[img_id] = max(scores.get(img_id, 0.0), score)
    return scores


def summarize(coco, predictions, threshold):
    scores = max_score_by_image(predictions)
    positive_image_ids = {ann["image_id"] for ann in coco.get("annotations", [])}
    rows = {}
    for image in coco.get("images", []):
        label_type = image.get("label_type", "")
        has_gt = image["id"] in positive_image_ids
        is_positive = has_gt or label_type in {"positive_real", "synthetic_pos"}
        group = "positive" if is_positive else label_type or "negative"
        row = rows.setdefault(
            group,
            {"images": 0, "flagged": 0, "false_positive": 0, "true_positive": 0},
        )
        row["images"] += 1
        flagged = scores.get(str(image["id"]), 0.0) >= threshold
        if flagged:
            row["flagged"] += 1
            if is_positive:
                row["true_positive"] += 1
            else:
                row["false_positive"] += 1

    for group, row in rows.items():
        denom = max(1, row["images"])
        row["flag_rate"] = row["flagged"] / denom
        row["fpr"] = row["false_positive"] / denom
        row["recall_image"] = row["true_positive"] / denom
    return rows


def choose_threshold(coco, predictions, target_fpr):
    scores = max_score_by_image(predictions)
    max_score = max(scores.values(), default=1.0)
    candidates = sorted({0.0, 1.0, max_score + 1e-6, *scores.values()})
    best = 1.0
    best_recall = -1.0
    for threshold in candidates:
        summary = summarize(coco, predictions, threshold)
        neg_images = 0
        false_pos = 0
        for group, row in summary.items():
            if group in NEGATIVE_LABEL_TYPES or group == "negative":
                neg_images += row["images"]
                false_pos += row["false_positive"]
        fpr = false_pos / max(1, neg_images)
        recall = summary.get("positive", {}).get("recall_image", 0.0)
        if fpr <= target_fpr and recall > best_recall:
            best = threshold
            best_recall = recall
    return best


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute image-level moderation metrics from COCO-format detections."
    )
    parser.add_argument("--val-coco", default="data/audit/coco/val.json", type=Path)
    parser.add_argument("--val-preds", required=True, type=Path)
    parser.add_argument("--test-coco", default="data/audit/coco/test_robustness.json", type=Path)
    parser.add_argument("--test-preds", required=True, type=Path)
    parser.add_argument("--target-val-fpr", type=float, default=0.01)
    parser.add_argument("--output", default="results/minimal_baselines/image_level_metrics.json", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    val_coco = load_json(args.val_coco)
    val_preds = load_json(args.val_preds)
    test_coco = load_json(args.test_coco)
    test_preds = load_json(args.test_preds)

    threshold = choose_threshold(val_coco, val_preds, args.target_val_fpr)
    result = {
        "threshold_source": "validation",
        "target_val_fpr": args.target_val_fpr,
        "selected_threshold": threshold,
        "val": summarize(val_coco, val_preds, threshold),
        "test": summarize(test_coco, test_preds, threshold),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Selected threshold on val: {threshold:.6f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
