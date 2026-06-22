#!/usr/bin/env python3
"""Audit completed runs against the canonical cross-experiment protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED: dict[str, Any] = {
    "protocol_name": "canonical_v2",
    "epochs": 50,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "workers": 8,
    "seed": 42,
    "model_score_threshold": 0.001,
    "fppi_threshold": 0.25,
    "hflip_prob": 0.5,
    "aug_brightness": 0.2,
    "aug_saturation": 0.2,
    "aug_hue": 0.015,
    "patience": 15,
}


ALIASES: dict[str, tuple[str, ...]] = {
    "protocol_name": ("protocol_name",),
    "epochs": ("training_epochs", "epochs"),
    "train_batch_size": ("training_batch_size", "batch_size"),
    "eval_batch_size": ("batch", "eval_batch_size"),
    "workers": ("workers",),
    "seed": ("seed",),
    "model_score_threshold": ("conf", "model_score_threshold"),
    "fppi_threshold": ("score_threshold", "fppi_threshold"),
    "hflip_prob": ("hflip_prob",),
    "aug_brightness": ("aug_brightness",),
    "aug_saturation": ("aug_saturation",),
    "aug_hue": ("aug_hue",),
    "patience": ("patience",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[
            Path("results/baselines"),
            Path("results/dataset_ablation/runs"),
            Path("results/hnsard_loss_ablation/runs"),
        ],
        help="Result roots searched recursively for completed-run config.json files.",
    )
    return parser.parse_args()


def canonical_value(config: dict[str, Any], field: str) -> Any:
    for key in ALIASES[field]:
        if key in config:
            return config[key]
    return None


def equivalent(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    return actual == expected


def completed_configs(roots: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for metrics_path in root.rglob("final_metrics.json"):
            config_path = metrics_path.parent / "config.json"
            if config_path.is_file():
                paths.add(config_path)
    return sorted(paths)


def main() -> None:
    args = parse_args()
    configs = completed_configs(args.roots)
    mismatches: list[dict[str, Any]] = []
    for path in configs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload if isinstance(payload, dict) else {}
        for field, expected in EXPECTED.items():
            actual = canonical_value(config, field)
            if not equivalent(actual, expected):
                mismatches.append(
                    {"config": str(path), "field": field, "expected": expected, "actual": actual}
                )

    report = {"configs_checked": len(configs), "mismatches": mismatches}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
