#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/experiment_defaults.sh"

PYTHON="${PYTHON:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-data}"
GRID_ROOT="${GRID_ROOT:-results/hnsard_loss_weight_grid}"
OUT_ROOT="${OUT_ROOT:-$GRID_ROOT/runs}"
TABLE_DIR="${TABLE_DIR:-$GRID_ROOT/tables}"

MODEL="${MODEL:-fasterrcnn_r50}"
ANCHOR_PRESET="${ANCHOR_PRESET:-micro}"
EPOCHS_LOSS_GRID="${EPOCHS_LOSS_GRID:-$EXPERIMENT_EPOCHS}"
BATCH_SIZE="${BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
LR="${LR:-0.005}"
LR_STEP_SIZE="${LR_STEP_SIZE:-10}"
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

LAMBDA_POS_GRID="${LAMBDA_POS_GRID:-0.25 0.5 0.75 1.0}"
LAMBDA_CON_GRID="${LAMBDA_CON_GRID:-0 0.05 0.1 0.2}"
CONTRASTIVE_WARMUP_EPOCHS="${CONTRASTIVE_WARMUP_EPOCHS:-5}"
SCALE_AWARE="${SCALE_AWARE:-1}"
TINY_WEIGHT="${TINY_WEIGHT:-3.0}"
SMALL_WEIGHT="${SMALL_WEIGHT:-2.0}"
MEDIUM_WEIGHT="${MEDIUM_WEIGHT:-1.2}"
LARGE_WEIGHT="${LARGE_WEIGHT:-1.0}"
HARD_NEGATIVE_PRESELECT="${HARD_NEGATIVE_PRESELECT:-128}"
HARD_NEGATIVES_PER_IMAGE="${HARD_NEGATIVES_PER_IMAGE:-16}"
MAX_POSITIVE_REGIONS_PER_IMAGE="${MAX_POSITIVE_REGIONS_PER_IMAGE:-16}"
TEACHER_BANK_SIZE="${TEACHER_BANK_SIZE:-512}"

DRY_RUN="${DRY_RUN:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
COLLECT_ERRORS="${COLLECT_ERRORS:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
PATIENCE_GRID="${PATIENCE_GRID:-$PATIENCE}"
NO_PRETRAINED="${NO_PRETRAINED:-0}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"
MAX_TRAIN_IMAGES="${MAX_TRAIN_IMAGES:-}"
MAX_VAL_IMAGES="${MAX_VAL_IMAGES:-}"
MAX_TEST_IMAGES="${MAX_TEST_IMAGES:-}"

read -r -a LAMBDA_POS_ARRAY <<< "$LAMBDA_POS_GRID"
read -r -a LAMBDA_CON_ARRAY <<< "$LAMBDA_CON_GRID"
read -r -a EVAL_SPLIT_ARRAY <<< "$EVAL_SPLITS"

safe_token() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  echo "$value"
}

is_zero_weight() {
  [[ "$1" =~ ^0+([.]0+)?$ ]]
}

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
    --epochs "$EPOCHS_LOSS_GRID"
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
    --anchor-preset "$ANCHOR_PRESET"
    --model-score-threshold "$MODEL_SCORE_THRESHOLD"
    --hflip-prob "$HFLIP_PROB"
    --aug-brightness "$AUG_BRIGHTNESS"
    --aug-saturation "$AUG_SATURATION"
    --aug-hue "$AUG_HUE"
    --protocol-name "$EXPERIMENT_PROTOCOL"
    --eval-splits "${EVAL_SPLIT_ARRAY[@]}"
    --teacher-model "$TEACHER_MODEL"
    --teacher-dim "$TEACHER_DIM"
    --teacher-crop-size "$TEACHER_CROP_SIZE"
    --teacher-context-scale "$TEACHER_CONTEXT_SCALE"
    --teacher-crop-batch-size "$TEACHER_CROP_BATCH_SIZE"
    --tiny-weight "$TINY_WEIGHT"
    --small-weight "$SMALL_WEIGHT"
    --medium-weight "$MEDIUM_WEIGHT"
    --large-weight "$LARGE_WEIGHT"
    --max-positive-regions-per-image "$MAX_POSITIVE_REGIONS_PER_IMAGE"
    --hard-negative-preselect "$HARD_NEGATIVE_PRESELECT"
    --hard-negatives-per-image "$HARD_NEGATIVES_PER_IMAGE"
    --teacher-bank-size "$TEACHER_BANK_SIZE"
    --patience "$PATIENCE_GRID"
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

summarize_grid() {
  "$PYTHON" - "$OUT_ROOT" "$TABLE_DIR" "$SUMMARY_SPLIT" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


result_root = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
split = sys.argv[3]

fields = [
    "run",
    "split",
    "protocol_name",
    "model",
    "anchor_preset",
    "lambda_pos",
    "lambda_con",
    "scale_aware",
    "contrastive_warmup_epochs",
    "tiny_weight",
    "small_weight",
    "medium_weight",
    "large_weight",
    "AP",
    "AP50",
    "AP75",
    "AP_tiny",
    "AP_small",
    "FPR_in_domain",
    "FPR_out_domain",
    "FPR@95TPR",
    "image_AUROC",
    "image_AP",
    "negative_fppi",
    "detections",
    "metrics_path",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def split_metrics(path: Path, split_name: str) -> dict[str, Any] | None:
    payload = load_json(path)
    if split_name in payload and isinstance(payload[split_name], dict):
        return payload[split_name]
    if payload.get("split") == split_name:
        return payload
    return None


def metric_value(metrics: dict[str, Any], field: str) -> Any:
    metric_map = {
        "AP": "mAP",
        "AP50": "mAP50",
        "AP75": "mAP75",
        "AP_tiny": "mAP_tiny",
        "AP_small": "mAP_small_rel",
        "FPR_in_domain": "in_domain_FPR@95TPR",
        "FPR_out_domain": "out_domain_FPR@95TPR",
    }
    if field == "negative_fppi":
        return metrics.get("false_positives_on_negatives", {}).get("fppi")
    return metrics.get(metric_map.get(field, field))


def sortable_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


rows: list[dict[str, Any]] = []
if result_root.is_dir():
    for run_dir in sorted(path for path in result_root.iterdir() if path.is_dir()):
        metrics_path = run_dir / "final_metrics.json"
        metrics = split_metrics(metrics_path, split)
        if metrics is None:
            continue

        config = load_json(run_dir / "config.json")
        row = {
            "run": run_dir.name,
            "split": split,
            "protocol_name": config.get("protocol_name"),
            "model": config.get("model"),
            "anchor_preset": config.get("anchor_preset"),
            "lambda_pos": config.get("lambda_pos"),
            "lambda_con": config.get("lambda_con"),
            "scale_aware": config.get("scale_aware"),
            "contrastive_warmup_epochs": config.get("contrastive_warmup_epochs"),
            "tiny_weight": config.get("tiny_weight"),
            "small_weight": config.get("small_weight"),
            "medium_weight": config.get("medium_weight"),
            "large_weight": config.get("large_weight"),
            "metrics_path": str(metrics_path),
        }
        for field in fields:
            row.setdefault(field, metric_value(metrics, field))
        rows.append(row)

rows.sort(key=lambda item: (sortable_float(item.get("lambda_pos")), sortable_float(item.get("lambda_con")), item["run"]))

output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / f"loss_weight_grid_{split}.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

md_fields = ["run", "lambda_pos", "lambda_con", "AP", "AP_tiny", "AP_small", "FPR@95TPR", "FPR_in_domain", "negative_fppi"]
md_path = output_dir / f"loss_weight_grid_{split}.md"
lines = [
    "# HN-SARD Loss Weight Grid",
    "",
    "| " + " | ".join(md_fields) + " |",
    "| " + " | ".join("---" for _ in md_fields) + " |",
]
for row in rows:
    lines.append("| " + " | ".join("" if row.get(field) is None else str(row.get(field, "")) for field in md_fields) + " |")
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps({"rows": len(rows), "csv": str(csv_path), "markdown": str(md_path)}, indent=2))
PY
}

run_combo() {
  local lambda_pos="$1"
  local lambda_con="$2"
  local pos_token
  local con_token
  pos_token="$(safe_token "$lambda_pos")"
  con_token="$(safe_token "$lambda_con")"
  local run_name="lp${pos_token}_lc${con_token}"
  local output_dir="$OUT_ROOT/$run_name"

  if has_final_result "$output_dir"; then
    echo "Skipping $run_name: found $output_dir/final_metrics.json"
    return
  fi

  mapfile -t train_args < <(base_train_args "$output_dir")
  local resume_checkpoint="$output_dir/checkpoints/last.pt"

  if is_zero_weight "$lambda_pos" && is_zero_weight "$lambda_con"; then
    train_args+=(--teacher-backend none --lambda-pos 0 --lambda-con 0 --no-scale-aware --contrastive-warmup-epochs 0)
  else
    train_args+=(--teacher-backend "$TEACHER_BACKEND" --lambda-pos "$lambda_pos" --lambda-con "$lambda_con")
    if [[ "$SCALE_AWARE" == "1" ]]; then
      train_args+=(--scale-aware)
    else
      train_args+=(--no-scale-aware)
    fi
    train_args+=(--contrastive-warmup-epochs "$CONTRASTIVE_WARMUP_EPOCHS")
  fi

  if [[ "$AUTO_RESUME" == "1" && -s "$resume_checkpoint" ]]; then
    echo "Resuming $run_name from $resume_checkpoint"
    train_args+=(--resume "$resume_checkpoint")
  fi

  echo "Running $run_name: lambda_pos=$lambda_pos lambda_con=$lambda_con"
  "$PYTHON" scripts/train_hnsard.py "${train_args[@]}"

  local predictions="$output_dir/final/predictions_${ERROR_SPLIT}.json"
  if [[ "$DRY_RUN" != "1" && "$COLLECT_ERRORS" == "1" && -f "$predictions" ]]; then
    "$PYTHON" scripts/collect_baseline_errors.py \
      --data-root "$DATA_ROOT" \
      --split "$ERROR_SPLIT" \
      --predictions "$predictions" \
      --output-dir "$GRID_ROOT/errors/$run_name" \
      --score-threshold "$FPPI_THRESHOLD" \
      --draw-limit "$ERROR_DRAW_LIMIT"
  fi
}

for lambda_pos in "${LAMBDA_POS_ARRAY[@]}"; do
  for lambda_con in "${LAMBDA_CON_ARRAY[@]}"; do
    run_combo "$lambda_pos" "$lambda_con"
  done
done

if [[ "$RUN_SUMMARY" == "1" && "$DRY_RUN" != "1" ]]; then
  summarize_grid
fi
