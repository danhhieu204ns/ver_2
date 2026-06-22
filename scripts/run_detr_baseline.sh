#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/experiment_defaults.sh"

PYTHON="${PYTHON:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-data}"
OUT_ROOT="${OUT_ROOT:-results/baselines}"
DETR_EPOCHS="${DETR_EPOCHS:-$EXPERIMENT_EPOCHS}"
DETR_BATCH="${DETR_BATCH:-$TRAIN_BATCH_SIZE}"
DETR_EVAL_BATCH="${DETR_EVAL_BATCH:-$EVAL_BATCH_SIZE}"
DETR_LR="${DETR_LR:-0.0001}"
DETR_BACKBONE_LR="${DETR_BACKBONE_LR:-0.00001}"
DETR_EVAL_EVERY="${DETR_EVAL_EVERY:-$EVAL_EVERY}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
read -r -a EVAL_SPLIT_ARRAY <<< "$EVAL_SPLITS"

OUTPUT_DIR="$OUT_ROOT/detr_r50"
if [[ "$SKIP_COMPLETED" == "1" && -s "$OUTPUT_DIR/final_metrics.json" ]]; then
  echo "Skipping detr_r50: found $OUTPUT_DIR/final_metrics.json"
  exit 0
fi

"$PYTHON" scripts/train_detr_baseline.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$DETR_EPOCHS" \
  --batch-size "$DETR_BATCH" \
  --eval-batch-size "$DETR_EVAL_BATCH" \
  --lr "$DETR_LR" \
  --backbone-lr "$DETR_BACKBONE_LR" \
  --shortest-edge "$MIN_SIZE" \
  --longest-edge "$MAX_SIZE" \
  --model-score-threshold "$MODEL_SCORE_THRESHOLD" \
  --fppi-threshold "$FPPI_THRESHOLD" \
  --hflip-prob "$HFLIP_PROB" \
  --aug-brightness "$AUG_BRIGHTNESS" \
  --aug-saturation "$AUG_SATURATION" \
  --aug-hue "$AUG_HUE" \
  --protocol-name "$EXPERIMENT_PROTOCOL" \
  --eval-splits "${EVAL_SPLIT_ARRAY[@]}" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --eval-every "$DETR_EVAL_EVERY" \
  --print-freq "$PRINT_FREQ"
