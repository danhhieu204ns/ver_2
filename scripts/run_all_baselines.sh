#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_TORCHVISION="${RUN_TORCHVISION:-1}"
RUN_YOLO="${RUN_YOLO:-1}"
RUN_DETR="${RUN_DETR:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

if [[ "$RUN_TORCHVISION" == "1" ]]; then
  bash scripts/run_torchvision_baselines.sh
fi

if [[ "$RUN_YOLO" == "1" ]]; then
  bash scripts/run_yolo_baselines.sh
fi

if [[ "$RUN_DETR" == "1" ]]; then
  bash scripts/run_detr_baseline.sh
fi

if [[ "$RUN_SUMMARY" == "1" ]]; then
  "${PYTHON:-.venv/bin/python}" scripts/summarize_baseline_results.py \
    --result-root results/baselines \
    --output-dir results/baselines/tables \
    --split test
fi
