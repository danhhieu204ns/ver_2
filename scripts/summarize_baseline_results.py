#!/usr/bin/env python3
"""Build the phase-1 baseline table from final_metrics.json files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "method",
    "split",
    "protocol_name",
    "mAP",
    "mAP50",
    "mAP75",
    "mAP_tiny",
    "mAP_small_rel",
    "mAP_medium_rel",
    "mAP_large_rel",
    "AR100",
    "image_AUROC",
    "image_AP",
    "FPR@95TPR",
    "Recall@FPR=1%",
    "Recall@FPR=5%",
    "in_domain_FPR@95TPR",
    "out_domain_FPR@95TPR",
    "negative_fppi",
    "detections",
    "metrics_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=Path("results/baselines"), type=Path, help="Baseline result root.")
    parser.add_argument("--output-dir", default=Path("results/baselines/tables"), type=Path, help="Output directory.")
    parser.add_argument("--split", default="test", help="Split to summarize from final_metrics.json.")
    parser.add_argument("--include-smoke", action="store_true", help="Include smoke-test result directories.")
    parser.add_argument("--allow-mixed-protocols", action="store_true", help="Allow missing or mixed protocol_name values.")
    return parser.parse_args()


def metric_value(metrics: dict[str, Any], key: str) -> Any:
    if key == "negative_fppi":
        return metrics.get("false_positives_on_negatives", {}).get("fppi")
    return metrics.get(key)


def method_name(metrics_path: Path, result_root: Path) -> str:
    parent = metrics_path.parent
    try:
        return str(parent.relative_to(result_root))
    except ValueError:
        return parent.name


def load_split_metrics(path: Path, split: str) -> dict[str, Any] | None:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and split in payload and isinstance(payload[split], dict):
        return payload[split]
    if isinstance(payload, dict) and payload.get("split") == split:
        return payload
    return None


def collect_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(args.result_root.rglob("final_metrics.json")):
        rel = str(metrics_path.relative_to(args.result_root))
        if not args.include_smoke and "smoke" in rel:
            continue
        if "yolo_dataset" in rel:
            continue
        metrics = load_split_metrics(metrics_path, args.split)
        if metrics is None:
            continue
        config_path = metrics_path.parent / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        row = {
            "method": method_name(metrics_path, args.result_root),
            "split": args.split,
            "protocol_name": config.get("protocol_name"),
            "metrics_path": str(metrics_path),
        }
        for field in FIELDS:
            if field in row:
                continue
            row[field] = metric_value(metrics, field)
        rows.append(row)
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    table_fields = [field for field in FIELDS if field != "metrics_path"]
    lines = [
        "# Baseline Table",
        "",
        "| " + " | ".join(table_fields) + " |",
        "| " + " | ".join("---" for _ in table_fields) + " |",
    ]
    for row in rows:
        values = ["" if row.get(field) is None else str(row.get(field)) for field in table_fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = collect_rows(args)
    protocols = {row.get("protocol_name") for row in rows}
    if rows and not args.allow_mixed_protocols and (None in protocols or len(protocols) != 1):
        raise RuntimeError(f"Missing or mixed experiment protocols: {sorted(map(str, protocols))}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / f"baseline_table_{args.split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.output_dir / f"baseline_table_{args.split}.md"
    md_path.write_text(render_markdown(rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
