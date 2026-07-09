#!/usr/bin/env python3
"""Build a final table across all HN-SARD loss-ablation experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPERIMENTS = [
    {
        "name": "hnsard_loss_ablation",
        "label": "Loss ablation (micro)",
        "runs": Path("results_full/results/hnsard_loss_ablation/runs"),
    },
    {
        "name": "hnsard_loss_ablation_lowloss_default",
        "label": "Loss ablation low-loss default",
        "runs": Path("results_full/results/hnsard_loss_ablation_lowloss_default/runs"),
    },
    {
        "name": "hnsard_loss_weight_grid_predpos_default",
        "label": "Loss weight grid predpos default",
        "runs": Path("results_full/results/hnsard_loss_weight_grid_predpos_default/runs"),
    },
]

METHODS = {
    "baseline": "Baseline Faster R-CNN",
    "l_pos": "+ L_pos",
    "l_pos_scale": "+ L_pos + scale-aware",
    "l_pos_scale_l_con": "+ L_pos + scale-aware + L_con",
    "full_hnsard": "Full HN-SARD",
}

FIELDS = [
    "rank_AP",
    "experiment",
    "experiment_label",
    "setting",
    "method",
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
    parser.add_argument("--split", default="test", help="Split to collect from final_metrics.json.")
    parser.add_argument(
        "--output-dir",
        default=Path("results_full/results/hnsard_loss_ablation_all/tables"),
        type=Path,
        help="Output directory for the final CSV and markdown tables.",
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
        "AP_medium": "mAP_medium_rel",
        "AP_large": "mAP_large_rel",
        "FPR_in_domain": "in_domain_FPR@95TPR",
        "FPR_out_domain": "out_domain_FPR@95TPR",
    }
    if field == "negative_fppi":
        return metrics.get("false_positives_on_negatives", {}).get("fppi")
    return metrics.get(metric_map.get(field, field))


def ranked_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    ap = row.get("AP")
    return (
        -(float(ap) if ap is not None else -1.0),
        row["experiment"],
        row["setting"],
    )


def collect_rows(split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        runs_root = experiment["runs"]
        if not runs_root.is_dir():
            continue
        for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            metrics_path = run_dir / "final_metrics.json"
            if not metrics_path.is_file():
                continue
            metrics = load_split_metrics(metrics_path, split)
            if metrics is None:
                continue
            config = load_json(run_dir / "config.json") if (run_dir / "config.json").is_file() else {}
            row = {
                "experiment": experiment["name"],
                "experiment_label": experiment["label"],
                "setting": run_dir.name,
                "method": METHODS.get(run_dir.name, run_dir.name),
                "split": split,
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
    rows.sort(key=ranked_sort_key)
    for index, row in enumerate(rows, start=1):
        row["rank_AP"] = index
    return rows


def render_markdown(rows: list[dict[str, Any]], split: str) -> str:
    display_fields = [
        "rank_AP",
        "experiment_label",
        "setting",
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
        "rank_AP": "Rank",
        "experiment_label": "Experiment",
        "setting": "Setting",
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
        f"# Final HN-SARD Loss Ablation Table ({split})",
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
    rows = collect_rows(args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / f"loss_ablation2_all_{args.split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.output_dir / f"loss_ablation2_all_{args.split}.md"
    md_path.write_text(render_markdown(rows, args.split), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
