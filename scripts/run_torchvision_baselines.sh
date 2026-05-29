#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-data}"
OUT_ROOT="${OUT_ROOT:-results/baselines}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"
EPOCHS_TORCHVISION="${EPOCHS_TORCHVISION:-20}"
EVAL_EVERY="${EVAL_EVERY:-1}"
PRINT_FREQ="${PRINT_FREQ:-50}"
FRCNN_BATCH="${FRCNN_BATCH:-16}"
FRCNN_EVAL_BATCH="${FRCNN_EVAL_BATCH:-16}"
SSD_BATCH="${SSD_BATCH:-16}"
SSD_EVAL_BATCH="${SSD_EVAL_BATCH:-16}"
LR_FRCNN="${LR_FRCNN:-0.005}"
LR_SSD="${LR_SSD:-0.002}"

run_torchvision_baseline() {
  local model="$1"
  local output_name="$2"
  local batch_size="$3"
  local eval_batch_size="$4"
  local lr="$5"

  "$PYTHON" scripts/train_faster_rcnn_baseline.py \
    --model "$model" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUT_ROOT/$output_name" \
    --epochs "$EPOCHS_TORCHVISION" \
    --batch-size "$batch_size" \
    --eval-batch-size "$eval_batch_size" \
    --lr "$lr" \
    --lr-step-size 5 \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --eval-every "$EVAL_EVERY" \
    --print-freq "$PRINT_FREQ"
}

run_torchvision_baseline "fasterrcnn_r50" "fasterrcnn_r50" "$FRCNN_BATCH" "$FRCNN_EVAL_BATCH" "$LR_FRCNN"
run_torchvision_baseline "fasterrcnn_r101" "fasterrcnn_r101" "$FRCNN_BATCH" "$FRCNN_EVAL_BATCH" "$LR_FRCNN"
run_torchvision_baseline "ssd300_vgg16" "ssd300_vgg16" "$SSD_BATCH" "$SSD_EVAL_BATCH" "$LR_SSD"
