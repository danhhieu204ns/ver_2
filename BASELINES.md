# Baseline Setup

Use the repo virtual environment for all commands:

```bash
cd /home/jovyan/ver_2
.venv/bin/python --version
```

## Dataset

The scripts read the existing annotation files:

```text
data/positive/annotations.json
data/negative_in_domain/annotations.json
data/negative_out_domain/annotations.json
```

The current split is `70/15/15`:

```text
train: 5141 images, 1837 objects
val:   1102 images, 402 objects
test:  1101 images, 397 objects
```

## Canonical Experiment Protocol

Baseline comparison, dataset ablation, and method ablation source the same defaults from `scripts/experiment_defaults.sh`:

| Setting | Canonical value |
|---|---:|
| protocol | `canonical_v2` |
| epochs | 50 |
| train/eval batch size | 8 / 8 |
| seed | 42 |
| evaluation splits | val, test |
| resize (Faster R-CNN/DETR) | shortest 800, longest 1333 |
| YOLO square input | 960 |
| horizontal flip | 0.5 |
| brightness / saturation / hue jitter | 0.2 / 0.2 / 0.015 |
| retained evaluation confidence | 0.001 |
| fixed FPPI threshold | 0.25 |

YOLO mosaic, mixup, copy-paste, scale, translate, rotation, shear, perspective, and vertical flip are explicitly disabled. Optimizer and learning rate remain architecture-specific and are saved in each run config; forcing one optimizer onto CNN and transformer detectors is not a controlled comparison.

All three experiments use the same train/val/test records. Dataset ablation changes training groups only; its val/test records remain identical. Method ablation changes only the HN-SARD components listed by the variant.

`SKIP_COMPLETED` defaults to `0` so metrics produced under an older protocol are not silently reused. Set `SKIP_COMPLETED=1` only when the existing `config.json` has the same `protocol_name`. Table builders reject missing or mixed protocols by default.

## Baseline Matrix

| Group | Baseline | Script | Output |
|---|---|---|---|
| Paper-faithful | Faster R-CNN R50 no-KD | `scripts/train_faster_rcnn_baseline.py --model fasterrcnn_r50` | `results/baselines/fasterrcnn_r50` |
| Capacity check | Faster R-CNN R101 no-KD | `scripts/train_faster_rcnn_baseline.py --model fasterrcnn_r101` | `results/baselines/fasterrcnn_r101` |
| Real-time | YOLOv8s, YOLOv8m | `scripts/run_yolo_baselines.sh` | `results/baselines/yolo/yolov8s`, `results/baselines/yolo/yolov8m` |
| Modern YOLO | YOLOv9s/m, YOLO11s/m | `scripts/run_yolo_baselines.sh` | `results/baselines/yolo/*` |
| Transformer | DETR-R50 | `scripts/train_detr_baseline.py` | `results/baselines/detr_r50` |
| Lightweight | SSD/VGG16, YOLO-nano | `scripts/train_faster_rcnn_baseline.py --model ssd300_vgg16`, `scripts/run_yolo_baselines.sh` | `results/baselines/ssd300_vgg16`, `results/baselines/yolo/yolo26n` |

Notes:

- `fasterrcnn_r50` loads torchvision COCO detection weights, then replaces the final class head.
- `fasterrcnn_r101` uses an ImageNet-pretrained ResNet-101 FPN backbone because torchvision does not ship a COCO Faster R-CNN R101 checkpoint.
- `ssd300_vgg16` loads torchvision COCO detector weights, replaces its classification head with a 2-class head, and uses gradient clipping for early-step stability.
- `detr_r50` loads `facebook/detr-resnet-50` and replaces the class head for one foreground class.
- `yolo26n.pt` is used as the YOLO-nano/lightweight baseline because that nano-sized weight is present in the repo.

## Run Everything

This runs all baselines sequentially:

```bash
bash scripts/run_all_baselines.sh
```

Expected full set:

```text
Faster R-CNN R50
Faster R-CNN R101
SSD300 VGG16
YOLOv8s
YOLOv8m
YOLOv9s
YOLOv9m
YOLO11s
YOLO11m
YOLO26n
DETR-R50
```

Useful overrides:

```bash
# Smoke/short run for code checking.
EPOCHS_TORCHVISION=1 YOLO_EPOCHS=1 DETR_EPOCHS=1 bash scripts/run_all_baselines.sh

# Run only YOLO baselines.
RUN_TORCHVISION=0 RUN_DETR=0 bash scripts/run_all_baselines.sh

# Run only torchvision baselines.
RUN_YOLO=0 RUN_DETR=0 bash scripts/run_all_baselines.sh

# Run only DETR.
RUN_TORCHVISION=0 RUN_YOLO=0 bash scripts/run_all_baselines.sh
```

## Run By Group

Torchvision baselines:

```bash
bash scripts/run_torchvision_baselines.sh
```

YOLO baselines:

```bash
bash scripts/run_yolo_baselines.sh
```

DETR baseline:

```bash
bash scripts/run_detr_baseline.sh
```

## Individual Commands

Paper-faithful Faster R-CNN R50:

```bash
.venv/bin/python scripts/train_faster_rcnn_baseline.py \
  --model fasterrcnn_r50 \
  --data-root data \
  --output-dir results/baselines/fasterrcnn_r50 \
  --epochs 50 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --lr 0.005 \
  --lr-step-size 5 \
  --workers 8 \
  --hflip-prob 0.5 --aug-brightness 0.2 --aug-saturation 0.2 --aug-hue 0.015 \
  --seed 42
```

Capacity check Faster R-CNN R101:

```bash
.venv/bin/python scripts/train_faster_rcnn_baseline.py \
  --model fasterrcnn_r101 \
  --data-root data \
  --output-dir results/baselines/fasterrcnn_r101 \
  --epochs 50 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --lr 0.005 \
  --lr-step-size 5 \
  --workers 8 \
  --hflip-prob 0.5 --aug-brightness 0.2 --aug-saturation 0.2 --aug-hue 0.015 \
  --seed 42
```

Lightweight SSD/VGG16:

```bash
.venv/bin/python scripts/train_faster_rcnn_baseline.py \
  --model ssd300_vgg16 \
  --data-root data \
  --output-dir results/baselines/ssd300_vgg16 \
  --epochs 50 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --lr 0.002 \
  --clip-grad-norm 10.0 \
  --lr-step-size 5 \
  --workers 8 \
  --hflip-prob 0.5 --aug-brightness 0.2 --aug-saturation 0.2 --aug-hue 0.015 \
  --seed 42
```

Transformer DETR-R50:

```bash
.venv/bin/python scripts/train_detr_baseline.py \
  --data-root data \
  --output-dir results/baselines/detr_r50 \
  --epochs 50 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --lr 0.0001 \
  --backbone-lr 0.00001 \
  --workers 8 \
  --hflip-prob 0.5 --aug-brightness 0.2 --aug-saturation 0.2 --aug-hue 0.015 \
  --seed 42
```

YOLO dataset export:

```bash
.venv/bin/python scripts/export_yolo_dataset.py \
  --data-root data \
  --output-dir results/baselines/yolo_dataset \
  --overwrite
```

Single YOLO example:

```bash
.venv/bin/yolo detect train \
  model=yolo11s.pt \
  data=results/baselines/yolo_dataset/dataset.yaml \
  imgsz=960 \
  epochs=50 \
  batch=8 \
  seed=42 \
  project=results/baselines/yolo \
  name=yolo11s
```

## Outputs And Metrics

Torchvision and DETR outputs:

```text
results/baselines/<name>/checkpoints/best.pt
results/baselines/<name>/final_metrics.json
results/baselines/<name>/final_metrics.csv
results/baselines/<name>/final/predictions_test.json
results/baselines/<name>/final/image_scores_test.csv
```

YOLO outputs:

```text
results/baselines/yolo/<name>/weights/best.pt
results/baselines/yolo/<name>_test
results/baselines/yolo/<name>_eval/final_metrics.json
```

Report these metrics:

- `mAP`: COCO mAP at IoU `0.50:0.95`.
- `mAP50`, `mAP75`.
- `mAP_tiny`, `mAP_small_rel`, `mAP_medium_rel`, `mAP_large_rel`: AP by relative bbox area bucket (`tiny < 1%`, `small < 5%`, `medium < 25%`, otherwise large).
- `AR100`: closest to the paper's AR style.
- `image_AUROC`, `image_AP`: image-level moderation metrics using max detection score per image.
- `FPR@95TPR`, `Recall@FPR=1%`, `Recall@FPR=5%`: high-recall moderation operating metrics.
- `in_domain_FPR@95TPR`, `out_domain_FPR@95TPR`: domain-specific image-level false-positive rates at the 95% TPR operating point.
- `false_positives_on_negatives.fppi`: false positives per negative image at score threshold `0.25`.
- `invalid_predictions_dropped`: malformed, non-finite, or out-of-range predictions excluded before evaluation; this should be zero for a valid run.

Metrics from Python scripts are stored as fractions. For paper-style percentages, multiply by `100`.

Metric regression checks:

```bash
.venv/bin/python scripts/test_baseline_eval_utils.py
.venv/bin/python scripts/check_experiment_protocol.py
```

The metric checks cover tied-score image AP, zero-threshold behavior for images without detections, invalid prediction filtering, and relative-scale isolation. The protocol audit scans completed configs and exits nonzero when any common budget, augmentation, seed, or evaluation threshold differs from `canonical_v2`. COCO metrics return `null`, rather than a misleading `0`, when a ground-truth/area bucket is unavailable.

## Standalone Evaluation

Recompute the clean baseline metrics for any COCO-style prediction file:

```bash
.venv/bin/python scripts/evaluate_predictions.py \
  --data-root data \
  --split test \
  --predictions results/baselines/fasterrcnn_r50/final/predictions_test.json \
  --output-dir results/baselines/fasterrcnn_r50/recomputed_test
```

Evaluate a trained YOLO checkpoint with the same metric code used by the torchvision and DETR baselines:

```bash
.venv/bin/python scripts/evaluate_yolo_baseline.py \
  --weights results/baselines/yolo/yolo11s/weights/best.pt \
  --data-root data \
  --output-dir results/baselines/yolo/yolo11s_eval \
  --splits val test \
  --imgsz 960
```

Build the main baseline table:

```bash
.venv/bin/python scripts/summarize_baseline_results.py \
  --result-root results/baselines \
  --output-dir results/baselines/tables \
  --split test
```

## Stage 3 HN-SARD Loss Ablation

Smoke test without downloading DINOv2:

```bash
VARIANTS="baseline l_pos_scale_l_con" \
TEACHER_BACKEND=dummy \
EPOCHS_LOSS_ABLATION=1 \
MAX_TRAIN_IMAGES=8 MAX_VAL_IMAGES=4 MAX_TEST_IMAGES=4 \
BATCH_SIZE=1 EVAL_BATCH_SIZE=1 WORKERS=0 \
NO_PRETRAINED=1 MIN_SIZE=128 MAX_SIZE=256 \
SKIP_FINAL_EVAL=1 RUN_SUMMARY=0 COLLECT_ERRORS=0 \
ABLATION_ROOT=results/hnsard_smoke/loss_ablation \
  bash scripts/run_loss_ablation.sh
```

Full Experiment 3 loss ablation:

```bash
bash scripts/run_loss_ablation.sh
```

The first real run may download `facebook/dinov2-small` through Hugging Face. Set `TEACHER_LOCAL_FILES_ONLY=1` to force cached/offline loading.

Default variants:

```text
baseline
l_pos
l_pos_scale
l_pos_scale_l_con
full_hnsard
```

Outputs:

```text
results/hnsard_loss_ablation/runs/<variant>/checkpoints/best.pt
results/hnsard_loss_ablation/runs/<variant>/final_metrics.json
results/hnsard_loss_ablation/runs/<variant>/final/predictions_test.json
results/hnsard_loss_ablation/tables/loss_ablation_test.csv
results/hnsard_loss_ablation/tables/loss_ablation_test.md
```

Rebuild the loss-ablation table:

```bash
.venv/bin/python scripts/summarize_loss_ablation.py \
  --result-root results/hnsard_loss_ablation/runs \
  --output-dir results/hnsard_loss_ablation/tables \
  --split test
```

Collect qualitative error cases and overlay images:

```bash
.venv/bin/python scripts/collect_baseline_errors.py \
  --data-root data \
  --split test \
  --predictions results/baselines/fasterrcnn_r50/final/predictions_test.json \
  --output-dir results/baselines/fasterrcnn_r50/errors_test
```
