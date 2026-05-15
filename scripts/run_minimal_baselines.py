import argparse
import shlex
import subprocess
import sys
from pathlib import Path


BASELINES = ("faster_rcnn_r50_fpn", "yolov8", "yolov11", "rtdetr", "old_dinov2_kd")


def py_cmd():
    venv_python = Path(".venv") / "Scripts" / "python.exe"
    if venv_python.is_file():
        return [str(venv_python)]
    return [sys.executable]


def quote_cmd(cmd):
    return " ".join(shlex.quote(str(part)) for part in cmd)


def build_commands(args):
    commands = []
    train_ann = Path("data/audit/coco/train.json")
    val_ann = Path("data/audit/coco/val.json")
    yolo_yaml = Path("data/ultralytics_minimal/nine_dash_line.yaml")

    requested = set(args.models)

    if {"yolov8", "yolov11", "rtdetr"} & requested:
        commands.append(
            py_cmd()
            + [
                "scripts/prepare_ultralytics_dataset.py",
                "--data-root",
                "data",
                "--coco-dir",
                "data/audit/coco",
                "--output-root",
                str(yolo_yaml.parent),
            ]
        )

    if "faster_rcnn_r50_fpn" in requested:
        commands.append(
            py_cmd()
            + [
                "code/train_faster_rcnn_r50_fpn.py",
                "--image-root",
                "data",
                "--train-ann",
                str(train_ann),
                "--val-ann",
                str(val_ann),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size_frcnn),
                "--num-workers",
                str(args.workers),
                "--run-name",
                "baseline_faster_rcnn_r50_fpn",
                "--output-dir",
                "results/minimal_baselines/faster_rcnn",
            ]
        )

    if "old_dinov2_kd" in requested:
        commands.append(
            py_cmd()
            + [
                "code/train_old_dinov2_kd_from_notebook.py",
                "--image-root",
                "data",
                "--train-ann",
                str(train_ann),
                "--val-ann",
                str(val_ann),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size_frcnn),
                "--num-workers",
                str(args.workers),
                "--beta1",
                "1.0",
                "--beta2",
                "0.1",
                "--kd-weight",
                "2.0",
                "--iou-threshold",
                "0.2",
                "--run-name",
                "baseline_old_dinov2_l1_irm_notebook",
                "--output-dir",
                "results/minimal_baselines/old_dinov2_kd",
            ]
        )

    ultralytics_specs = {
        "yolov8": "yolov8s.pt",
        "yolov11": "yolo11s.pt",
        "rtdetr": "rtdetr-l.pt",
    }
    for name, weights in ultralytics_specs.items():
        if name not in requested:
            continue
        commands.append(
            [
                "yolo",
                "detect",
                "train",
                f"model={weights}",
                f"data={yolo_yaml}",
                f"epochs={args.epochs}",
                f"imgsz={args.imgsz}",
                f"batch={args.batch_size_yolo}",
                "project=results/minimal_baselines/ultralytics",
                f"name=baseline_{name}",
                "exist_ok=True",
            ]
        )

    return commands


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run only the minimal baseline set: Faster R-CNN, YOLOv8/YOLOv11, RT-DETR, old DINOv2 KD."
    )
    parser.add_argument("--models", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch-size-frcnn", type=int, default=4)
    parser.add_argument("--batch-size-yolo", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute commands. Without this flag the script prints the exact commands only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    commands = build_commands(args)
    for idx, cmd in enumerate(commands, start=1):
        print(f"\n[{idx}/{len(commands)}] {quote_cmd(cmd)}")
        if args.run:
            subprocess.run(cmd, check=True)

    if not args.run:
        print("\nDry run only. Add --run to execute.")


if __name__ == "__main__":
    main()
