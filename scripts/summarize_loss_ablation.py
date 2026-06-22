#!/usr/bin/env python3
"""Build the Stage 3 HN-SARD loss-ablation table from final metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VARIANTS: dict[str, str] = {
    "baseline": "Baseline Faster R-CNN",
    "l_pos": "+ L_pos",
    "l_pos_scale": "+ L_pos + scale-aware",
    "l_pos_scale_l_con": "+ L_pos + scale-aware + L_con",
    "full_hnsard": "Full HN-SARD",
}

FIELDS = [
    "variant",
    "method",
    "split",
    "AP",
    "AP50",
    "AP75",
    "AP_tiny",
    "AP_small",
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
    "contrastive_warmup_epochs",
    "metrics_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        default=Path("results/hnsard_loss_ablation/runs"),
        type=Path,
        help="Root containing one run directory per loss-ablation variant.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("results/hnsard_loss_ablation/tables"),
        type=Path,
        help="Output directory for CSV and markdown tables.",
    )
    parser.add_argument("--split", default="test", help="Split to summarize from final_metrics.json.")
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include expected variants with blank metrics when final_metrics.json is missing.",
    )
    return parser.parse_args()


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
        "FPR_in_domain": "in_domain_FPR@95TPR",
        "FPR_out_domain": "out_domain_FPR@95TPR",
    }
    if field == "negative_fppi":
        return metrics.get("false_positives_on_negatives", {}).get("fppi")
    return metrics.get(metric_map.get(field, field))


def empty_row(variant: str, split: str, metrics_path: Path) -> dict[str, Any]:
    row = {
        "variant": variant,
        "method": VARIANTS[variant],
        "split": split,
        "metrics_path": str(metrics_path),
    }
    for field in FIELDS:
        row.setdefault(field, "")
    return row


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, method in VARIANTS.items():
        run_dir = args.result_root / variant
        metrics_path = run_dir / "final_metrics.json"
        if not metrics_path.is_file():
            if args.include_missing:
                rows.append(empty_row(variant, args.split, metrics_path))
            continue

        metrics = load_split_metrics(metrics_path, args.split)
        if metrics is None:
            if args.include_missing:
                rows.append(empty_row(variant, args.split, metrics_path))
            continue

        config = load_json(run_dir / "config.json") if (run_dir / "config.json").is_file() else {}
        row = {
            "variant": variant,
            "method": method,
            "split": args.split,
            "metrics_path": str(metrics_path),
            "lambda_pos": config.get("lambda_pos"),
            "lambda_con": config.get("lambda_con"),
            "scale_aware": config.get("scale_aware"),
            "contrastive_warmup_epochs": config.get("contrastive_warmup_epochs"),
        }
        for field in FIELDS:
            if field in row:
                continue
            row[field] = metric_value(metrics, field)
        rows.append(row)
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    display_fields = [
        "method",
        "AP",
        "AP_tiny",
        "AP_small",
        "FPR_in_domain",
        "FPR@95TPR",
        "negative_fppi",
    ]
    headers = {
        "method": "Variant",
        "AP": "AP",
        "AP_tiny": "AP_tiny",
        "AP_small": "AP_small",
        "FPR_in_domain": "FPR in-domain",
        "FPR@95TPR": "FPR@95TPR",
        "negative_fppi": "Negative FPPI",
    }
    lines = [
        "# HN-SARD Loss Ablation Table",
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

    csv_path = args.output_dir / f"loss_ablation_{args.split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.output_dir / f"loss_ablation_{args.split}.md"
    md_path.write_text(render_markdown(rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
