#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
YOLO="${YOLO:-$ROOT_DIR/.venv/bin/yolo}"
DATA_ROOT="${DATA_ROOT:-data}"
OUT_ROOT="${OUT_ROOT:-results/baselines}"
YOLO_DATASET="${YOLO_DATASET:-$OUT_ROOT/yolo_dataset}"
YOLO_PROJECT="${YOLO_PROJECT:-$OUT_ROOT/yolo}"
SEED="${SEED:-42}"
YOLO_EPOCHS="${YOLO_EPOCHS:-100}"
YOLO_IMGSZ="${YOLO_IMGSZ:-960}"
YOLO_BATCH_S="${YOLO_BATCH_S:-32}"
YOLO_BATCH_M="${YOLO_BATCH_M:-32}"
YOLO_WORKERS="${YOLO_WORKERS:-16}"
YOLO_MODELS="${YOLO_MODELS:-yolov8s.pt yolov8m.pt yolov9s.pt yolov9m.pt yolo11s.pt yolo11m.pt yolo26n.pt}"
YOLO_EVAL_SPLITS="${YOLO_EVAL_SPLITS:-val test}"
YOLO_EVAL_CONF="${YOLO_EVAL_CONF:-0.001}"
YOLO_EVAL_SCORE_THRESHOLD="${YOLO_EVAL_SCORE_THRESHOLD:-0.25}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

to_abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$ROOT_DIR" "$1" ;;
  esac
}

DATA_ROOT="$(to_abs_path "$DATA_ROOT")"
OUT_ROOT="$(to_abs_path "$OUT_ROOT")"
YOLO_DATASET="$(to_abs_path "$YOLO_DATASET")"
YOLO_PROJECT="$(to_abs_path "$YOLO_PROJECT")"

models_to_run=()
for model_path in $YOLO_MODELS; do
  name="$(basename "$model_path" .pt)"
  final_metrics="$YOLO_PROJECT/${name}_eval/final_metrics.json"
  if [[ "$SKIP_COMPLETED" == "1" && -s "$final_metrics" ]]; then
    echo "Skipping $name: found $final_metrics"
    continue
  fi
  models_to_run+=("$model_path")
done

if [[ "${#models_to_run[@]}" -eq 0 ]]; then
  echo "All YOLO baselines already have final metrics."
  exit 0
fi

"$PYTHON" scripts/export_yolo_dataset.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$YOLO_DATASET" \
  --overwrite

for model_path in "${models_to_run[@]}"; do
  name="$(basename "$model_path" .pt)"

  if [[ ! -f "$model_path" ]]; then
    echo "Missing YOLO weight: $model_path" >&2
    exit 1
  fi

  batch="$YOLO_BATCH_S"
  if [[ "$name" == *m ]]; then
    batch="$YOLO_BATCH_M"
  fi

  "$YOLO" detect train \
    model="$model_path" \
    data="$YOLO_DATASET/dataset.yaml" \
    imgsz="$YOLO_IMGSZ" \
    epochs="$YOLO_EPOCHS" \
    batch="$batch" \
    workers="$YOLO_WORKERS" \
    seed="$SEED" \
    project="$YOLO_PROJECT" \
    name="$name" \
    exist_ok=True

  "$YOLO" detect val \
    model="$YOLO_PROJECT/$name/weights/best.pt" \
    data="$YOLO_DATASET/dataset.yaml" \
    split=test \
    imgsz="$YOLO_IMGSZ" \
    batch="$batch" \
    workers="$YOLO_WORKERS" \
    project="$YOLO_PROJECT" \
    name="${name}_test" \
    exist_ok=True

  "$PYTHON" scripts/evaluate_yolo_baseline.py \
    --weights "$YOLO_PROJECT/$name/weights/best.pt" \
    --data-root "$DATA_ROOT" \
    --output-dir "$YOLO_PROJECT/${name}_eval" \
    --splits $YOLO_EVAL_SPLITS \
    --imgsz "$YOLO_IMGSZ" \
    --batch "$batch" \
    --workers "$YOLO_WORKERS" \
    --conf "$YOLO_EVAL_CONF" \
    --score-threshold "$YOLO_EVAL_SCORE_THRESHOLD"
done
