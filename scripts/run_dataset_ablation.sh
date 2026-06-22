#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/experiment_defaults.sh"

PYTHON="${PYTHON:-.venv/bin/python}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-data}"
ABLATION_ROOT="${ABLATION_ROOT:-results/dataset_ablation}"
ABLATION_DATA_ROOT="${ABLATION_DATA_ROOT:-$ABLATION_ROOT/data}"
OUT_ROOT="${OUT_ROOT:-$ABLATION_ROOT/runs}"
VARIANTS="${VARIANTS:-positive_only positive_negative_out_domain positive_negative_in_domain positive_both_negatives}"

MODEL="${MODEL:-fasterrcnn_r50}"
EPOCHS_DATASET_ABLATION="${EPOCHS_DATASET_ABLATION:-$EXPERIMENT_EPOCHS}"
BATCH_SIZE="${BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
LR="${LR:-0.005}"
LR_STEP_SIZE="${LR_STEP_SIZE:-5}"
SUMMARY_SPLIT="${SUMMARY_SPLIT:-test}"
ERROR_SPLIT="${ERROR_SPLIT:-test}"
ERROR_DRAW_LIMIT="${ERROR_DRAW_LIMIT:-100}"

IMAGE_MODE="${IMAGE_MODE:-symlink}"
OVERWRITE_DATA="${OVERWRITE_DATA:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
COLLECT_ERRORS="${COLLECT_ERRORS:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
PATIENCE_ABLATION="${PATIENCE_ABLATION:-$PATIENCE}"
NO_PRETRAINED="${NO_PRETRAINED:-0}"
MAX_TRAIN_IMAGES="${MAX_TRAIN_IMAGES:-}"
MAX_VAL_IMAGES="${MAX_VAL_IMAGES:-}"
MAX_TEST_IMAGES="${MAX_TEST_IMAGES:-}"

read -r -a VARIANT_ARRAY <<< "$VARIANTS"
read -r -a EVAL_SPLIT_ARRAY <<< "$EVAL_SPLITS"

variants_to_run=()
for variant in "${VARIANT_ARRAY[@]}"; do
  final_metrics="$OUT_ROOT/$variant/final_metrics.json"
  if [[ "$DRY_RUN" != "1" && "$SKIP_COMPLETED" == "1" && -s "$final_metrics" ]]; then
    echo "Skipping $variant: found $final_metrics"
    continue
  fi
  variants_to_run+=("$variant")
done

create_args=(
  --data-root "$SOURCE_DATA_ROOT"
  --output-root "$ABLATION_DATA_ROOT"
  --image-mode "$IMAGE_MODE"
  --variants "${variants_to_run[@]}"
)
if [[ "$DRY_RUN" != "1" && "${#variants_to_run[@]}" -eq 0 ]]; then
  echo "All dataset-ablation variants already have final metrics."
elif [[ "$OVERWRITE_DATA" == "1" ]]; then
  create_args+=(--overwrite)
  "$PYTHON" scripts/create_dataset_ablation_variants.py "${create_args[@]}"
else
  "$PYTHON" scripts/create_dataset_ablation_variants.py "${create_args[@]}"
fi

for variant in "${variants_to_run[@]}"; do
  output_dir="$OUT_ROOT/$variant"

  train_args=(
    --model "$MODEL"
    --data-root "$ABLATION_DATA_ROOT/$variant"
    --output-dir "$output_dir"
    --epochs "$EPOCHS_DATASET_ABLATION"
    --batch-size "$BATCH_SIZE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --lr "$LR"
    --lr-step-size "$LR_STEP_SIZE"
    --workers "$WORKERS"
    --seed "$SEED"
    --eval-every "$EVAL_EVERY"
    --print-freq "$PRINT_FREQ"
    --fppi-threshold "$FPPI_THRESHOLD"
    --min-size "$MIN_SIZE"
    --max-size "$MAX_SIZE"
    --model-score-threshold "$MODEL_SCORE_THRESHOLD"
    --hflip-prob "$HFLIP_PROB"
    --aug-brightness "$AUG_BRIGHTNESS"
    --aug-saturation "$AUG_SATURATION"
    --aug-hue "$AUG_HUE"
    --protocol-name "$EXPERIMENT_PROTOCOL"
    --eval-splits "${EVAL_SPLIT_ARRAY[@]}"
    --patience "$PATIENCE_ABLATION"
  )

  if [[ -n "$MAX_TRAIN_IMAGES" ]]; then
    train_args+=(--max-train-images "$MAX_TRAIN_IMAGES")
  fi
  if [[ -n "$MAX_VAL_IMAGES" ]]; then
    train_args+=(--max-val-images "$MAX_VAL_IMAGES")
  fi
  if [[ -n "$MAX_TEST_IMAGES" ]]; then
    train_args+=(--max-test-images "$MAX_TEST_IMAGES")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    train_args+=(--dry-run)
  fi
  if [[ "$NO_PRETRAINED" == "1" ]]; then
    train_args+=(--no-pretrained)
  fi

  "$PYTHON" scripts/train_faster_rcnn_baseline.py "${train_args[@]}"

  predictions="$OUT_ROOT/$variant/final/predictions_${ERROR_SPLIT}.json"
  if [[ "$DRY_RUN" != "1" && "$COLLECT_ERRORS" == "1" && -f "$predictions" ]]; then
    "$PYTHON" scripts/collect_baseline_errors.py \
      --data-root "$ABLATION_DATA_ROOT/$variant" \
      --split "$ERROR_SPLIT" \
      --predictions "$predictions" \
      --output-dir "$ABLATION_ROOT/errors/$variant" \
      --score-threshold "$FPPI_THRESHOLD" \
      --draw-limit "$ERROR_DRAW_LIMIT"
  fi
done

if [[ "$RUN_SUMMARY" == "1" && "$DRY_RUN" != "1" ]]; then
  "$PYTHON" scripts/summarize_dataset_ablation.py \
    --result-root "$OUT_ROOT" \
    --output-dir "$ABLATION_ROOT/tables" \
    --split "$SUMMARY_SPLIT"
fi
