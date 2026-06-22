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
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FRCNN_BATCH="${FRCNN_BATCH:-8}"
FRCNN_EVAL_BATCH="${FRCNN_EVAL_BATCH:-8}"
SSD_BATCH="${SSD_BATCH:-8}"
SSD_EVAL_BATCH="${SSD_EVAL_BATCH:-8}"
LR_FRCNN="${LR_FRCNN:-0.005}"
LR_SSD="${LR_SSD:-0.002}"
SSD_CLIP_GRAD_NORM="${SSD_CLIP_GRAD_NORM:-10.0}"

has_final_result() {
  local output_dir="$1"
  [[ "$SKIP_COMPLETED" == "1" && -s "$output_dir/final_metrics.json" ]]
}

run_torchvision_baseline() {
  local model="$1"
  local output_name="$2"
  local batch_size="$3"
  local eval_batch_size="$4"
  local lr="$5"
  shift 5
  local extra_args=("$@")
  local output_dir="$OUT_ROOT/$output_name"

  if has_final_result "$output_dir"; then
    echo "Skipping $output_name: found $output_dir/final_metrics.json"
    return
  fi

  "$PYTHON" scripts/train_faster_rcnn_baseline.py \
    --model "$model" \
    --data-root "$DATA_ROOT" \
    --output-dir "$output_dir" \
    --epochs "$EPOCHS_TORCHVISION" \
    --batch-size "$batch_size" \
    --eval-batch-size "$eval_batch_size" \
    --lr "$lr" \
    --lr-step-size 5 \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --eval-every "$EVAL_EVERY" \
    --print-freq "$PRINT_FREQ" \
    "${extra_args[@]}"
}

run_torchvision_baseline "fasterrcnn_r50" "fasterrcnn_r50" "$FRCNN_BATCH" "$FRCNN_EVAL_BATCH" "$LR_FRCNN"
run_torchvision_baseline "fasterrcnn_r101" "fasterrcnn_r101" "$FRCNN_BATCH" "$FRCNN_EVAL_BATCH" "$LR_FRCNN"
run_torchvision_baseline "ssd300_vgg16" "ssd300_vgg16" "$SSD_BATCH" "$SSD_EVAL_BATCH" "$LR_SSD" \
  --clip-grad-norm "$SSD_CLIP_GRAD_NORM"
