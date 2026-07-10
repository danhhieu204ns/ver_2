#!/usr/bin/env python3
"""Build the HN-SARD GT/context-scale ablation table from final metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SCALES = ["1.0", "1.25", "1.5", "2.0"]

FIELDS = [
    "run",
    "gt_scale",
    "split",
    "protocol_name",
    "model",
    "anchor_preset",
    "AP",
    "AP50",
    "AP75",
    "AP_tiny",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR100",
    "FPR_in_domain",
    "FPR_out_domain",
    "FPR@95TPR",
    "Recall@FPR=1%",
    "Recall@FPR=5%",
    "image_AUROC",
    "image_AP",
    "negative_fppi",
    "detections",
    "lambda_pos",
    "lambda_con",
    "scale_aware",
    "positive_proposal_iou_threshold",
    "contrastive_warmup_epochs",
    "teacher_crop_size",
    "teacher_context_scale",
    "metrics_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        default=Path("results/hnsard_gt_scale_ablation/runs"),
        type=Path,
        help="Root containing one run directory per GT/context scale.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("results/hnsard_gt_scale_ablation/tables"),
        type=Path,
        help="Output directory for CSV and markdown tables.",
    )
    parser.add_argument("--split", default="test", help="Split to summarize from final_metrics.json.")
    parser.add_argument("--scales", nargs="+", default=DEFAULT_SCALES, help="GT/context scales to summarize.")
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include expected scales with blank metrics when final_metrics.json is missing.",
    )
    return parser.parse_args()


def safe_token(value: str) -> str:
    return value.replace(".", "p").replace("-", "m")


def run_name_for_scale(scale: str) -> str:
    return f"gt_scale_{safe_token(scale)}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def load_split_metrics(path: Path, split: str) -> dict[str, Any] | None:
    payload = load_json(path)
    if split in payload and isinstance(payload[split], dict):
        return payload[split]
    if payload.get("split") == split:
        return payload
    return None


def metric_value(metrics: dict[str, Any], field: str) -> Any:
    metric_map = {
        "AP": "mAP",
        "AP50": "mAP50",
        "AP75": "mAP75",
        "AP_tiny": "mAP_tiny",
        "AP_small": "mAP_small_rel",
        "AP_medium": "mAP_medium_rel",
        "AP_large": "mAP_large_rel",
        "FPR_in_domain": "in_domain_FPR@95TPR",
        "FPR_out_domain": "out_domain_FPR@95TPR",
    }
    if field == "negative_fppi":
        return metrics.get("false_positives_on_negatives", {}).get("fppi")
    return metrics.get(metric_map.get(field, field))


def empty_row(run: str, scale: str, split: str, metrics_path: Path) -> dict[str, Any]:
    row = {
        "run": run,
        "gt_scale": scale,
        "split": split,
        "teacher_context_scale": scale,
        "metrics_path": str(metrics_path),
    }
    for field in FIELDS:
        row.setdefault(field, "")
    return row


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scale in args.scales:
        run = run_name_for_scale(scale)
        run_dir = args.result_root / run
        metrics_path = run_dir / "final_metrics.json"
        if not metrics_path.is_file():
            if args.include_missing:
                rows.append(empty_row(run, scale, args.split, metrics_path))
            continue

        metrics = load_split_metrics(metrics_path, args.split)
        if metrics is None:
            if args.include_missing:
                rows.append(empty_row(run, scale, args.split, metrics_path))
            continue

        config = load_json(run_dir / "config.json") if (run_dir / "config.json").is_file() else {}
        context_scale = config.get("teacher_context_scale", scale)
        row = {
            "run": run,
            "gt_scale": context_scale,
            "split": args.split,
            "protocol_name": config.get("protocol_name"),
            "model": config.get("model"),
            "anchor_preset": config.get("anchor_preset"),
            "metrics_path": str(metrics_path),
            "lambda_pos": config.get("lambda_pos"),
            "lambda_con": config.get("lambda_con"),
            "scale_aware": config.get("scale_aware"),
            "positive_proposal_iou_threshold": config.get("positive_proposal_iou_threshold"),
            "contrastive_warmup_epochs": config.get("contrastive_warmup_epochs"),
            "teacher_crop_size": config.get("teacher_crop_size"),
            "teacher_context_scale": context_scale,
        }
        for field in FIELDS:
            if field in row:
                continue
            row[field] = metric_value(metrics, field)
        rows.append(row)
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    display_fields = [
        "gt_scale",
        "AP",
        "AP_tiny",
        "AP_small",
        "FPR_in_domain",
        "FPR@95TPR",
        "Recall@FPR=1%",
        "negative_fppi",
    ]
    headers = {
        "gt_scale": "GT scale",
        "AP": "AP",
        "AP_tiny": "AP_tiny",
        "AP_small": "AP_small",
        "FPR_in_domain": "FPR in-domain",
        "FPR@95TPR": "FPR@95TPR",
        "Recall@FPR=1%": "R@1%FPR",
        "negative_fppi": "Negative FPPI",
    }
    lines = [
        "# HN-SARD GT Scale Ablation Table",
        "",
        "| " + " | ".join(headers[field] for field in display_fields) + " |",
        "| " + " | ".join("---" for _ in display_fields) + " |",
    ]
    for row in rows:
        values = ["" if row.get(field) is None else str(row.get(field, "")) for field in display_fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = collect_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / f"gt_scale_ablation_{args.split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.output_dir / f"gt_scale_ablation_{args.split}.md"
    md_path.write_text(render_markdown(rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
