#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-data}"
OUT_ROOT="${OUT_ROOT:-results/baselines}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"
DETR_EPOCHS="${DETR_EPOCHS:-50}"
DETR_BATCH="${DETR_BATCH:-16}"
DETR_EVAL_BATCH="${DETR_EVAL_BATCH:-16}"
DETR_LR="${DETR_LR:-0.0001}"
DETR_BACKBONE_LR="${DETR_BACKBONE_LR:-0.00001}"
DETR_EVAL_EVERY="${DETR_EVAL_EVERY:-1}"
PRINT_FREQ="${PRINT_FREQ:-50}"

"$PYTHON" scripts/train_detr_baseline.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_ROOT/detr_r50" \
  --epochs "$DETR_EPOCHS" \
  --batch-size "$DETR_BATCH" \
  --eval-batch-size "$DETR_EVAL_BATCH" \
  --lr "$DETR_LR" \
  --backbone-lr "$DETR_BACKBONE_LR" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --eval-every "$DETR_EVAL_EVERY" \
  --print-freq "$PRINT_FREQ"
