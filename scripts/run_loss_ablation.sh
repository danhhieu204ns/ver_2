#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-data}"
ABLATION_ROOT="${ABLATION_ROOT:-results/hnsard_loss_ablation}"
OUT_ROOT="${OUT_ROOT:-$ABLATION_ROOT/runs}"
VARIANTS="${VARIANTS:-baseline l_pos l_pos_scale l_pos_scale_l_con full_hnsard}"

MODEL="${MODEL:-fasterrcnn_r50}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"
EPOCHS_LOSS_ABLATION="${EPOCHS_LOSS_ABLATION:-20}"
EVAL_EVERY="${EVAL_EVERY:-1}"
PRINT_FREQ="${PRINT_FREQ:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LR="${LR:-0.005}"
LR_STEP_SIZE="${LR_STEP_SIZE:-5}"
FPPI_THRESHOLD="${FPPI_THRESHOLD:-0.25}"
MIN_SIZE="${MIN_SIZE:-800}"
MAX_SIZE="${MAX_SIZE:-1333}"
MODEL_SCORE_THRESHOLD="${MODEL_SCORE_THRESHOLD:-0.05}"
HFLIP_PROB="${HFLIP_PROB:-0.0}"
EVAL_SPLITS="${EVAL_SPLITS:-val test}"
SUMMARY_SPLIT="${SUMMARY_SPLIT:-test}"
ERROR_SPLIT="${ERROR_SPLIT:-test}"
ERROR_DRAW_LIMIT="${ERROR_DRAW_LIMIT:-100}"

TEACHER_BACKEND="${TEACHER_BACKEND:-transformers}"
TEACHER_MODEL="${TEACHER_MODEL:-facebook/dinov2-small}"
TEACHER_DIM="${TEACHER_DIM:-384}"
TEACHER_CROP_SIZE="${TEACHER_CROP_SIZE:-224}"
TEACHER_CONTEXT_SCALE="${TEACHER_CONTEXT_SCALE:-1.5}"
TEACHER_CROP_BATCH_SIZE="${TEACHER_CROP_BATCH_SIZE:-32}"
TEACHER_LOCAL_FILES_ONLY="${TEACHER_LOCAL_FILES_ONLY:-0}"
TEACHER_FP16="${TEACHER_FP16:-0}"

LAMBDA_POS="${LAMBDA_POS:-0.5}"
LAMBDA_CON="${LAMBDA_CON:-0.1}"
CONTRASTIVE_WARMUP_EPOCHS="${CONTRASTIVE_WARMUP_EPOCHS:-3}"
HARD_NEGATIVE_PRESELECT="${HARD_NEGATIVE_PRESELECT:-128}"
HARD_NEGATIVES_PER_IMAGE="${HARD_NEGATIVES_PER_IMAGE:-16}"
MAX_POSITIVE_REGIONS_PER_IMAGE="${MAX_POSITIVE_REGIONS_PER_IMAGE:-16}"
TEACHER_BANK_SIZE="${TEACHER_BANK_SIZE:-512}"

DRY_RUN="${DRY_RUN:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
COLLECT_ERRORS="${COLLECT_ERRORS:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
NO_PRETRAINED="${NO_PRETRAINED:-0}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"
MAX_TRAIN_IMAGES="${MAX_TRAIN_IMAGES:-}"
MAX_VAL_IMAGES="${MAX_VAL_IMAGES:-}"
MAX_TEST_IMAGES="${MAX_TEST_IMAGES:-}"

read -r -a VARIANT_ARRAY <<< "$VARIANTS"
read -r -a EVAL_SPLIT_ARRAY <<< "$EVAL_SPLITS"

has_final_result() {
  local output_dir="$1"
  [[ "$DRY_RUN" != "1" && "$SKIP_COMPLETED" == "1" && -s "$output_dir/final_metrics.json" ]]
}

base_train_args() {
  local output_dir="$1"
  local args=(
    --model "$MODEL"
    --data-root "$DATA_ROOT"
    --output-dir "$output_dir"
    --epochs "$EPOCHS_LOSS_ABLATION"
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
    --eval-splits "${EVAL_SPLIT_ARRAY[@]}"
    --teacher-model "$TEACHER_MODEL"
    --teacher-dim "$TEACHER_DIM"
    --teacher-crop-size "$TEACHER_CROP_SIZE"
    --teacher-context-scale "$TEACHER_CONTEXT_SCALE"
    --teacher-crop-batch-size "$TEACHER_CROP_BATCH_SIZE"
    --max-positive-regions-per-image "$MAX_POSITIVE_REGIONS_PER_IMAGE"
    --hard-negative-preselect "$HARD_NEGATIVE_PRESELECT"
    --hard-negatives-per-image "$HARD_NEGATIVES_PER_IMAGE"
    --teacher-bank-size "$TEACHER_BANK_SIZE"
  )

  if [[ -n "$MAX_TRAIN_IMAGES" ]]; then
    args+=(--max-train-images "$MAX_TRAIN_IMAGES")
  fi
  if [[ -n "$MAX_VAL_IMAGES" ]]; then
    args+=(--max-val-images "$MAX_VAL_IMAGES")
  fi
  if [[ -n "$MAX_TEST_IMAGES" ]]; then
    args+=(--max-test-images "$MAX_TEST_IMAGES")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry-run)
  fi
  if [[ "$NO_PRETRAINED" == "1" ]]; then
    args+=(--no-pretrained)
  fi
  if [[ "$SKIP_FINAL_EVAL" == "1" ]]; then
    args+=(--skip-final-eval)
  fi
  if [[ "$TEACHER_LOCAL_FILES_ONLY" == "1" ]]; then
    args+=(--teacher-local-files-only)
  fi
  if [[ "$TEACHER_FP16" == "1" ]]; then
    args+=(--teacher-fp16)
  fi

  printf '%s\n' "${args[@]}"
}

run_variant() {
  local variant="$1"
  local output_dir="$OUT_ROOT/$variant"
  if has_final_result "$output_dir"; then
    echo "Skipping $variant: found $output_dir/final_metrics.json"
    return
  fi

  mapfile -t train_args < <(base_train_args "$output_dir")
  case "$variant" in
    baseline)
      train_args+=(--teacher-backend none --lambda-pos 0 --lambda-con 0 --no-scale-aware --contrastive-warmup-epochs 0)
      ;;
    l_pos)
      train_args+=(--teacher-backend "$TEACHER_BACKEND" --lambda-pos "$LAMBDA_POS" --lambda-con 0 --no-scale-aware --contrastive-warmup-epochs 0)
      ;;
    l_pos_scale)
      train_args+=(--teacher-backend "$TEACHER_BACKEND" --lambda-pos "$LAMBDA_POS" --lambda-con 0 --scale-aware --contrastive-warmup-epochs 0)
      ;;
    l_pos_scale_l_con)
      train_args+=(--teacher-backend "$TEACHER_BACKEND" --lambda-pos "$LAMBDA_POS" --lambda-con "$LAMBDA_CON" --scale-aware --contrastive-warmup-epochs 0)
      ;;
    full_hnsard)
      train_args+=(--teacher-backend "$TEACHER_BACKEND" --lambda-pos "$LAMBDA_POS" --lambda-con "$LAMBDA_CON" --scale-aware --contrastive-warmup-epochs "$CONTRASTIVE_WARMUP_EPOCHS")
      ;;
    *)
      echo "Unknown loss-ablation variant: $variant" >&2
      exit 2
      ;;
  esac

  "$PYTHON" scripts/train_hnsard.py "${train_args[@]}"

  local predictions="$OUT_ROOT/$variant/final/predictions_${ERROR_SPLIT}.json"
  if [[ "$DRY_RUN" != "1" && "$COLLECT_ERRORS" == "1" && -f "$predictions" ]]; then
    "$PYTHON" scripts/collect_baseline_errors.py \
      --data-root "$DATA_ROOT" \
      --split "$ERROR_SPLIT" \
      --predictions "$predictions" \
      --output-dir "$ABLATION_ROOT/errors/$variant" \
      --score-threshold "$FPPI_THRESHOLD" \
      --draw-limit "$ERROR_DRAW_LIMIT"
  fi
}

for variant in "${VARIANT_ARRAY[@]}"; do
  run_variant "$variant"
done

if [[ "$RUN_SUMMARY" == "1" && "$DRY_RUN" != "1" ]]; then
  "$PYTHON" scripts/summarize_loss_ablation.py \
    --result-root "$OUT_ROOT" \
    --output-dir "$ABLATION_ROOT/tables" \
    --split "$SUMMARY_SPLIT"
fi
