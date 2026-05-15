import argparse
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TORCHVISION_BASELINES = (
    "faster_rcnn_r50_fpn",
    "faster_rcnn_r101_fpn",
    "retinanet_r50_fpn",
    "fcos_r50_fpn",
)
MMDET_BASELINES = ("cascade_rcnn_r50_fpn",)
ULTRALYTICS_BASELINES = {
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov9s": "yolov9s.pt",
    "yolov9m": "yolov9m.pt",
    "yolov11s": "yolo11s.pt",
    "yolov11m": "yolo11m.pt",
    "rtdetr_l": "rtdetr-l.pt",
}
HF_TRANSFORMER_BASELINES = (
    "rtdetr_r18",
    "rtdetr_r50",
    "deformable_detr",
    "detr_r50",
)
OTHER_BASELINES = ("old_dinov2_kd",)
MODEL_ALIASES = {
    "yolo11s": "yolov11s",
    "yolo11m": "yolov11m",
}
DEFAULT_BASELINES = (
    "faster_rcnn_r50_fpn",
    "faster_rcnn_r101_fpn",
    "cascade_rcnn_r50_fpn",
    "yolov8s",
    "yolov8m",
    "yolov9s",
    "yolov9m",
    "yolov11s",
    "yolov11m",
    "retinanet_r50_fpn",
    "fcos_r50_fpn",
    "rtdetr_r18",
    "rtdetr_r50",
    "deformable_detr",
    "detr_r50",
    "old_dinov2_kd",
)
BASELINES = TORCHVISION_BASELINES + MMDET_BASELINES + tuple(ULTRALYTICS_BASELINES) + HF_TRANSFORMER_BASELINES + OTHER_BASELINES
MODEL_CHOICES = tuple(dict.fromkeys(BASELINES + tuple(MODEL_ALIASES)))


def py_cmd():
    for venv_python in (
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
    ):
        if venv_python.is_file():
            return [str(venv_python)]
    return [sys.executable]


def tool_cmd(name):
    python_path = Path(py_cmd()[0])
    candidate = python_path.parent / name
    if candidate.is_file():
        return [str(candidate)]
    return [name]


def quote_cmd(cmd):
    return " ".join(shlex.quote(str(part)) for part in cmd)


def backend_available(module_name):
    return importlib.util.find_spec(module_name) is not None


def normalize_models(models):
    return [MODEL_ALIASES.get(model, model) for model in models]


def has_ultralytics_model(models):
    return bool(set(models) & set(ULTRALYTICS_BASELINES))


def backend_status():
    return {
        "torchvision": backend_available("torchvision"),
        "transformers": backend_available("transformers"),
        "timm": backend_available("timm"),
        "ultralytics": backend_available("ultralytics"),
        "mmdet": backend_available("mmdet"),
        "mmengine": backend_available("mmengine"),
    }


def model_backend_missing(model_name, status):
    if model_name in ULTRALYTICS_BASELINES and not status["ultralytics"]:
        return "ultralytics"
    if model_name in HF_TRANSFORMER_BASELINES and not (status["transformers"] and status["timm"]):
        return "transformers+timm"
    if model_name in MMDET_BASELINES and not (status["mmdet"] and status["mmengine"]):
        return "mmdet+mmengine"
    return None


def unavailable_models(models, status):
    missing = {}
    for model_name in models:
        backend = model_backend_missing(model_name, status)
        if backend:
            missing[model_name] = backend
    return missing


def filter_available_models(models, status):
    return [model_name for model_name in models if model_backend_missing(model_name, status) is None]


def build_torchvision_command(args, model_name, seed):
    return py_cmd() + [
        "code/train_torchvision_detector.py",
        "--architecture",
        model_name,
        "--image-root",
        args.image_root,
        "--train-ann",
        args.train_ann,
        "--val-ann",
        args.val_ann,
        "--test-ann",
        args.test_ann,
        "--test-robustness-ann",
        args.test_robustness_ann,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size_torchvision),
        "--num-workers",
        str(args.workers),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--lr-schedule",
        args.lr_schedule,
        "--min-size",
        str(args.imgsz),
        "--max-size",
        str(args.imgsz),
        "--seed",
        str(seed),
        "--target-val-fpr",
        str(args.target_val_fpr),
        "--run-name",
        f"baseline_{model_name}_seed{seed}",
        "--output-dir",
        "results/stage2_baselines/torchvision",
    ] + (["--export-preds"] if args.export_preds else []) + (["--deterministic"] if args.deterministic else [])


def build_hf_transformer_command(args, model_name, seed):
    return py_cmd() + [
        "code/train_hf_transformer_detector.py",
        "--architecture",
        model_name,
        "--image-root",
        args.image_root,
        "--train-ann",
        args.train_ann,
        "--val-ann",
        args.val_ann,
        "--test-ann",
        args.test_ann,
        "--test-robustness-ann",
        args.test_robustness_ann,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size_hf),
        "--num-workers",
        str(args.workers),
        "--lr",
        str(args.lr_hf),
        "--weight-decay",
        str(args.weight_decay),
        "--lr-schedule",
        args.lr_schedule,
        "--imgsz",
        str(args.imgsz),
        "--pred-conf",
        str(args.pred_conf),
        "--seed",
        str(seed),
        "--target-val-fpr",
        str(args.target_val_fpr),
        "--run-name",
        f"baseline_{model_name}_seed{seed}",
        "--output-dir",
        "results/stage2_baselines/hf_transformers",
    ] + (["--export-preds"] if args.export_preds else []) + (["--deterministic"] if args.deterministic else [])


def build_mmdet_command(args, model_name, seed):
    cmd = py_cmd() + [
        "code/train_mmdetection_baseline.py",
        "--architecture",
        model_name,
        "--image-root",
        args.image_root,
        "--train-ann",
        args.train_ann,
        "--val-ann",
        args.val_ann,
        "--test-ann",
        args.test_ann,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size_mmdet),
        "--num-workers",
        str(args.workers),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(seed),
        "--run-name",
        f"baseline_{model_name}_seed{seed}",
        "--output-dir",
        "results/stage2_baselines/mmdetection",
    ]
    if args.mmdet_cascade_config:
        cmd += ["--config", args.mmdet_cascade_config]
    return cmd


def build_old_kd_command(args, seed):
    return py_cmd() + [
        "code/train_old_dinov2_kd_from_notebook.py",
        "--image-root",
        args.image_root,
        "--train-ann",
        args.train_ann,
        "--val-ann",
        args.val_ann,
        "--test-ann",
        args.test_ann,
        "--test-robustness-ann",
        args.test_robustness_ann,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size_torchvision),
        "--num-workers",
        str(args.workers),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--min-size",
        str(args.imgsz),
        "--max-size",
        str(args.imgsz),
        "--seed",
        str(seed),
        "--target-val-fpr",
        str(args.target_val_fpr),
        "--beta1",
        str(args.old_kd_beta1),
        "--beta2",
        str(args.old_kd_beta2),
        "--kd-weight",
        str(args.old_kd_weight),
        "--iou-threshold",
        str(args.old_kd_iou_threshold),
        "--run-name",
        "baseline_old_dinov2_l1_irm",
        "--output-dir",
        "results/stage2_baselines/old_dinov2_kd",
    ] + (["--export-preds"] if args.export_preds else []) + (["--deterministic"] if args.deterministic else [])


def build_ultralytics_commands(args, model_name, seed):
    weights = ULTRALYTICS_BASELINES[model_name]
    run_name = f"baseline_{model_name}_seed{seed}"
    project = Path("results/stage2_baselines/ultralytics")
    yolo_yaml = Path(args.ultralytics_root) / "nine_dash_line.yaml"
    best_weights = project / run_name / "weights" / "best.pt"
    pred_dir = Path("results/stage2_baselines/ultralytics_predictions") / run_name

    commands = [
        tool_cmd("yolo")
        + [
            "detect",
            "train",
            f"model={weights}",
            f"data={yolo_yaml}",
            f"epochs={args.epochs}",
            f"imgsz={args.imgsz}",
            f"batch={args.batch_size_ultralytics}",
            f"workers={args.workers}",
            f"seed={seed}",
            "optimizer=AdamW",
            f"lr0={args.lr}",
            f"weight_decay={args.weight_decay}",
            f"cos_lr={str(args.lr_schedule == 'cosine')}",
            f"project={project}",
            f"name={run_name}",
            "exist_ok=True",
            f"deterministic={str(args.deterministic)}",
        ]
    ]
    if not args.export_preds:
        return commands

    split_to_ann = {
        "val": args.val_ann,
        "test_standard": args.test_ann,
        "test_robustness": args.test_robustness_ann,
    }
    for split, ann in split_to_ann.items():
        commands.append(
            py_cmd()
            + [
                "code/export_ultralytics_predictions.py",
                "--weights",
                str(best_weights),
                "--coco-ann",
                ann,
                "--image-root",
                args.image_root,
                "--output",
                str(pred_dir / f"{split}_predictions.json"),
                "--imgsz",
                str(args.imgsz),
                "--conf",
                str(args.pred_conf),
                "--iou",
                str(args.pred_iou),
                "--batch",
                str(args.batch_size_ultralytics),
            ]
        )

    for split, ann in (("test_standard", args.test_ann), ("test_robustness", args.test_robustness_ann)):
        commands.append(
            py_cmd()
            + [
                "code/evaluate_image_level_fpr.py",
                "--val-coco",
                args.val_ann,
                "--val-preds",
                str(pred_dir / "val_predictions.json"),
                "--test-coco",
                ann,
                "--test-preds",
                str(pred_dir / f"{split}_predictions.json"),
                "--target-val-fpr",
                str(args.target_val_fpr),
                "--output",
                str(pred_dir / f"image_level_metrics_{split}.json"),
            ]
        )
    return commands


def build_commands(args):
    commands = []
    requested = list(dict.fromkeys(normalize_models(args.models)))
    status = backend_status()
    missing = unavailable_models(requested, status)
    if missing and args.skip_unavailable:
        for model_name, backend in missing.items():
            print(f"Skipping {model_name}: missing backend {backend}")
        requested = filter_available_models(requested, status)
    elif missing and args.run:
        details = ", ".join(f"{model}:{backend}" for model, backend in missing.items())
        raise SystemExit(
            "Missing backend(s) for requested model(s): " + details + 
            ". Install dependencies, choose --models explicitly, or add --skip-unavailable."
        )

    if has_ultralytics_model(requested):
        commands.append(
            py_cmd()
            + [
                "scripts/prepare_ultralytics_dataset.py",
                "--data-root",
                args.image_root,
                "--coco-dir",
                str(Path(args.train_ann).parent),
                "--output-root",
                args.ultralytics_root,
            ]
        )

    for seed in args.seeds:
        for model_name in requested:
            if model_name in TORCHVISION_BASELINES:
                commands.append(build_torchvision_command(args, model_name, seed))
            elif model_name in HF_TRANSFORMER_BASELINES:
                commands.append(build_hf_transformer_command(args, model_name, seed))
            elif model_name in MMDET_BASELINES:
                commands.append(build_mmdet_command(args, model_name, seed))
            elif model_name in ULTRALYTICS_BASELINES:
                commands.extend(build_ultralytics_commands(args, model_name, seed))
            elif model_name == "old_dinov2_kd":
                commands.append(build_old_kd_command(args, seed))
            else:
                raise ValueError(f"Unsupported model requested: {model_name}")
    return commands


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full Stage-2 baseline benchmark protocol.")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(DEFAULT_BASELINES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch-size-torchvision", type=int, default=8)
    parser.add_argument("--batch-size-ultralytics", type=int, default=16)
    parser.add_argument("--batch-size-hf", type=int, default=4)
    parser.add_argument("--batch-size-mmdet", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-hf", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=("none", "cosine", "step"), default="cosine")
    parser.add_argument("--target-val-fpr", type=float, default=0.01)
    parser.add_argument("--pred-conf", type=float, default=0.001)
    parser.add_argument("--pred-iou", type=float, default=0.7)
    parser.add_argument("--old-kd-beta1", type=float, default=1.0)
    parser.add_argument("--old-kd-beta2", type=float, default=0.1)
    parser.add_argument("--old-kd-weight", type=float, default=2.0)
    parser.add_argument("--old-kd-iou-threshold", type=float, default=0.2)
    parser.add_argument("--mmdet-cascade-config", default=None)
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--train-ann", default="data/audit/coco/train.json")
    parser.add_argument("--val-ann", default="data/audit/coco/val.json")
    parser.add_argument("--test-ann", default="data/audit/coco/test_standard.json")
    parser.add_argument("--test-robustness-ann", default="data/audit/coco/test_robustness.json")
    parser.add_argument("--ultralytics-root", default="data/ultralytics_stage2")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-export-preds", dest="export_preds", action="store_false")
    parser.set_defaults(export_preds=True)
    parser.add_argument("--skip-unavailable", action="store_true", help="Skip models whose backend packages are not installed.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute commands. Without this flag the script prints commands only.",
    )
    return parser.parse_args()


def print_backend_status():
    status = backend_status()
    print("Backend availability:", status)
    if not status["ultralytics"]:
        print("  - YOLO/Ultralytics commands are generated, but require `ultralytics` to be installed before --run.")
    if not (status["mmdet"] and status["mmengine"]):
        print("  - Cascade R-CNN command is generated, but requires MMDetection before --run.")


def main():
    args = parse_args()
    commands = build_commands(args)
    print("Default full Stage-2 baselines:", ", ".join(DEFAULT_BASELINES))
    print_backend_status()

    for idx, cmd in enumerate(commands, start=1):
        print(f"\n[{idx}/{len(commands)}] {quote_cmd(cmd)}")
        if args.run:
            subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    if not args.run:
        print("\nDry run only. Add --run to execute.")


if __name__ == "__main__":
    main()
