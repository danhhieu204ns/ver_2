#!/usr/bin/env python3
"""Build summary tables for an HN-SARD loss-weight grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "run",
    "split",
    "protocol_name",
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
    "image_AUROC",
    "image_AP",
    "negative_fppi",
    "detections",
    "lambda_pos",
    "lambda_con",
    "scale_aware",
    "positive_proposal_iou_threshold",
    "contrastive_warmup_epochs",
    "metrics_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        default=Path("results/hnsard_loss_weight_grid_predpos_default/runs"),
        type=Path,
        help="Root containing one run directory per loss-weight setting.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("results/hnsard_loss_weight_grid_predpos_default/tables"),
        type=Path,
        help="Output directory for CSV and markdown tables.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to summarize, or 'all' to include every split in final_metrics.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


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


def split_items(metrics_payload: dict[str, Any], split: str) -> list[tuple[str, dict[str, Any]]]:
    if split == "all":
        return [
            (name, metrics)
            for name, metrics in metrics_payload.items()
            if isinstance(metrics, dict)
        ]
    metrics = metrics_payload.get(split)
    if isinstance(metrics, dict):
        return [(split, metrics)]
    if metrics_payload.get("split") == split:
        return [(split, metrics_payload)]
    return []


def collect_rows(result_root: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in result_root.iterdir() if path.is_dir()):
        metrics_path = run_dir / "final_metrics.json"
        if not metrics_path.is_file():
            continue

        config = load_json(run_dir / "config.json") if (run_dir / "config.json").is_file() else {}
        metrics_payload = load_json(metrics_path)
        for split_name, metrics in split_items(metrics_payload, split):
            row = {
                "run": run_dir.name,
                "split": split_name,
                "protocol_name": config.get("protocol_name"),
                "anchor_preset": config.get("anchor_preset"),
                "metrics_path": str(metrics_path),
                "lambda_pos": config.get("lambda_pos"),
                "lambda_con": config.get("lambda_con"),
                "scale_aware": config.get("scale_aware"),
                "positive_proposal_iou_threshold": config.get("positive_proposal_iou_threshold"),
                "contrastive_warmup_epochs": config.get("contrastive_warmup_epochs"),
            }
            for field in FIELDS:
                if field not in row:
                    row[field] = metric_value(metrics, field)
            rows.append(row)
    rows.sort(key=lambda row: (row["split"], float(row["lambda_pos"]), float(row["lambda_con"]), row["run"]))
    return rows


def render_markdown(rows: list[dict[str, Any]], title: str) -> str:
    display_fields = [
        "run",
        "lambda_pos",
        "lambda_con",
        "AP",
        "AP_tiny",
        "AP_small",
        "FPR_in_domain",
        "FPR@95TPR",
        "negative_fppi",
    ]
    headers = {
        "run": "Run",
        "lambda_pos": "lambda_pos",
        "lambda_con": "lambda_con",
        "AP": "AP",
        "AP_tiny": "AP_tiny",
        "AP_small": "AP_small",
        "FPR_in_domain": "FPR in-domain",
        "FPR@95TPR": "FPR@95TPR",
        "negative_fppi": "Negative FPPI",
    }
    lines = [
        f"# {title}",
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
    rows = collect_rows(args.result_root, args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / f"loss_weight_grid_{args.split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.output_dir / f"loss_weight_grid_{args.split}.md"
    md_path.write_text(
        render_markdown(rows, f"HN-SARD Loss Weight Grid ({args.split})"),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
