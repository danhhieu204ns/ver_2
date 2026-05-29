#!/usr/bin/env python3
"""Train and evaluate a DETR-R50 baseline on the local dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_faster_rcnn_baseline import (
    CLASS_ID,
    CLASS_NAME,
    SPLITS,
    bbox_xywh_to_xyxy,
    limit_records,
    load_records,
    set_seed,
)
from baseline_eval_utils import build_detection_metrics, write_image_scores_csv


DETR_CLASS_ID = 0


def record_to_coco_annotations(record: dict[str, Any], image_id: int, category_id: int) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for obj in record["objects"]:
        if not isinstance(obj, dict):
            continue
        box = bbox_xywh_to_xyxy(obj.get("bbox"), record["width"], record["height"])
        if box is None:
            continue
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        annotations.append(
            {
                "bbox": [x1, y1, width, height],
                "category_id": category_id,
                "area": width * height,
                "iscrowd": 0,
                "image_id": image_id,
            }
        )
    return annotations


class DetrNineDashDataset(Dataset):
    def __init__(self, data_root: Path, split: str, max_images: int | None = None) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        self.data_root = data_root
        self.split = split
        self.records = limit_records(load_records(data_root, split), max_images)
        self.coco_ids = list(range(1, len(self.records) + 1))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, dict[str, Any], dict[str, Any]]:
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        coco_id = self.coco_ids[index]
        annotations = {
            "image_id": coco_id,
            "annotations": record_to_coco_annotations(record, coco_id, DETR_CLASS_ID),
        }
        meta = {
            "coco_id": coco_id,
            "height": int(record["height"]),
            "width": int(record["width"]),
            "group": record["group"],
            "file_name": record["file_name"],
            "has_gt": bool(record["objects"]),
        }
        return image, annotations, meta

    def record_for_coco_id(self, coco_id: int) -> dict[str, Any]:
        return self.records[self.coco_ids.index(coco_id)]

    def to_coco(self) -> COCO:
        dataset: dict[str, Any] = {
            "info": {"description": "nine-dash-line DETR baseline dataset"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": CLASS_ID, "name": CLASS_NAME}],
        }
        annotation_id = 1
        for index, record in enumerate(self.records):
            coco_id = self.coco_ids[index]
            dataset["images"].append(
                {
                    "id": coco_id,
                    "file_name": f"{record['group']}/{record['file_name']}",
                    "width": record["width"],
                    "height": record["height"],
                }
            )
            for annotation in record_to_coco_annotations(record, coco_id, CLASS_ID):
                annotation["id"] = annotation_id
                dataset["annotations"].append(annotation)
                annotation_id += 1

        coco = COCO()
        coco.dataset = dataset
        coco.createIndex()
        return coco


def make_collate_fn(processor: DetrImageProcessor):
    def collate_fn(batch: list[tuple[Image.Image, dict[str, Any], dict[str, Any]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        images, annotations, metas = zip(*batch)
        encoding = processor(images=list(images), annotations=list(annotations), return_tensors="pt")
        for image in images:
            image.close()
        return encoding, list(metas)

    return collate_fn


def make_loader(
    dataset: Dataset,
    processor: DetrImageProcessor,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=make_collate_fn(processor),
    )


def create_processor(args: argparse.Namespace) -> DetrImageProcessor:
    size = {"shortest_edge": args.shortest_edge, "longest_edge": args.longest_edge}
    if args.no_pretrained:
        return DetrImageProcessor(format="coco_detection", size=size)
    return DetrImageProcessor.from_pretrained(args.model_name, format="coco_detection", size=size)


def create_model(args: argparse.Namespace) -> DetrForObjectDetection:
    id2label = {DETR_CLASS_ID: CLASS_NAME}
    label2id = {CLASS_NAME: DETR_CLASS_ID}
    if args.no_pretrained:
        config = DetrConfig(
            num_labels=1,
            id2label=id2label,
            label2id=label2id,
            use_pretrained_backbone=False,
            num_queries=args.num_queries,
        )
        return DetrForObjectDetection(config)

    config = DetrConfig.from_pretrained(args.model_name)
    config.num_labels = 1
    config.id2label = id2label
    config.label2id = label2id
    return DetrForObjectDetection.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )


def move_labels_to_device(labels: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in label.items()} for label in labels]


def optimizer_parameter_groups(model: torch.nn.Module, lr: float, backbone_lr: float) -> list[dict[str, Any]]:
    backbone_names = ("backbone", "model.backbone")
    backbone_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(part in name for part in backbone_names):
            backbone_params.append(parameter)
        else:
            other_params.append(parameter)
    return [{"params": other_params, "lr": lr}, {"params": backbone_params, "lr": backbone_lr}]


def train_one_epoch(
    model: DetrForObjectDetection,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int,
) -> dict[str, float]:
    model.train()
    loss_totals: Counter[str] = Counter()
    start = time.time()

    for step, (encoding, _) in enumerate(loader, start=1):
        pixel_values = encoding["pixel_values"].to(device)
        pixel_mask = encoding["pixel_mask"].to(device)
        labels = move_labels_to_device(encoding["labels"], device)

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_totals["loss"] += float(loss.detach().cpu())
        for name, value in outputs.loss_dict.items():
            loss_totals[name] += float(value.detach().cpu())

        if print_freq > 0 and (step == 1 or step % print_freq == 0 or step == len(loader)):
            elapsed = time.time() - start
            avg_loss = loss_totals["loss"] / step
            print(f"epoch={epoch} step={step}/{len(loader)} loss={avg_loss:.4f} elapsed={elapsed:.1f}s", flush=True)

    return {name: value / max(len(loader), 1) for name, value in loss_totals.items()}


@torch.inference_mode()
def evaluate(
    model: DetrForObjectDetection,
    processor: DetrImageProcessor,
    loader: DataLoader,
    dataset: DetrNineDashDataset,
    device: torch.device,
    output_dir: Path,
    split: str,
    fppi_threshold: float,
) -> dict[str, Any]:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []

    for encoding, metas in loader:
        pixel_values = encoding["pixel_values"].to(device)
        pixel_mask = encoding["pixel_mask"].to(device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        target_sizes = torch.tensor([[meta["height"], meta["width"]] for meta in metas], device=device)
        processed = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)

        for meta, output in zip(metas, processed):
            coco_id = int(meta["coco_id"])

            for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
                if int(label.item()) != DETR_CLASS_ID:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                if width <= 0 or height <= 0:
                    continue
                score_value = float(score.detach().cpu().item())
                predictions.append(
                    {
                        "image_id": coco_id,
                        "category_id": CLASS_ID,
                        "bbox": [x1, y1, width, height],
                        "score": score_value,
                    }
                )

    metrics = build_detection_metrics(dataset.records, dataset.coco_ids, predictions, split, fppi_threshold)

    (output_dir / f"predictions_{split}.json").write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    write_image_scores_csv(output_dir / f"image_scores_{split}.csv", dataset.records, dataset.coco_ids, predictions)
    (output_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def dataset_summary(datasets: dict[str, DetrNineDashDataset]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split, dataset in datasets.items():
        group_counter = Counter(record["group"] for record in dataset.records)
        object_count = sum(len(record["objects"]) for record in dataset.records)
        positives = sum(1 for record in dataset.records if record["objects"])
        summary[split] = {
            "images": len(dataset),
            "positive_images": positives,
            "negative_images": len(dataset) - positives,
            "objects": object_count,
            "groups": dict(sorted(group_counter.items())),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=Path("data"), type=Path, help="Dataset root containing group folders.")
    parser.add_argument(
        "--output-dir",
        default=Path("results/baselines/detr_r50"),
        type=Path,
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument("--model-name", default="facebook/detr-resnet-50", help="Hugging Face DETR checkpoint name.")
    parser.add_argument("--epochs", default=50, type=int, help="Training epochs.")
    parser.add_argument("--batch-size", default=2, type=int, help="Training batch size.")
    parser.add_argument("--eval-batch-size", default=2, type=int, help="Evaluation batch size.")
    parser.add_argument("--workers", default=4, type=int, help="DataLoader workers.")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate for non-backbone parameters.")
    parser.add_argument("--backbone-lr", default=1e-5, type=float, help="Learning rate for DETR backbone parameters.")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="AdamW weight decay.")
    parser.add_argument("--lr-step-size", default=30, type=int, help="StepLR step size.")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="StepLR gamma.")
    parser.add_argument("--shortest-edge", default=800, type=int, help="DETR image processor shortest edge.")
    parser.add_argument("--longest-edge", default=1333, type=int, help="DETR image processor longest edge.")
    parser.add_argument("--num-queries", default=100, type=int, help="Number of object queries for random-init DETR.")
    parser.add_argument("--fppi-threshold", default=0.25, type=float, help="Score threshold for false-positive-per-image reporting.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--print-freq", default=50, type=int, help="Training log frequency in steps.")
    parser.add_argument("--eval-every", default=1, type=int, help="Run validation every N epochs.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable Hugging Face pretrained DETR weights.")
    parser.add_argument("--resume", default=None, type=Path, help="Resume checkpoint.")
    parser.add_argument("--eval-only", default=None, type=Path, help="Only evaluate a checkpoint.")
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=SPLITS, help="Splits used for final/eval-only evaluation.")
    parser.add_argument("--max-train-images", default=None, type=int, help="Limit training images for smoke tests.")
    parser.add_argument("--max-val-images", default=None, type=int, help="Limit validation images for smoke tests.")
    parser.add_argument("--max-test-images", default=None, type=int, help="Limit test images for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Load datasets and one processed batch, then exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = create_processor(args)
    datasets = {
        "train": DetrNineDashDataset(args.data_root, "train", args.max_train_images),
        "val": DetrNineDashDataset(args.data_root, "val", args.max_val_images),
        "test": DetrNineDashDataset(args.data_root, "test", args.max_test_images),
    }
    summary = dataset_summary(datasets)
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"device": str(device), "model": "detr_r50", "dataset": summary}, ensure_ascii=False, indent=2), flush=True)

    train_loader = make_loader(datasets["train"], processor, args.batch_size, args.workers, shuffle=True)
    val_loader = make_loader(datasets["val"], processor, args.eval_batch_size, args.workers, shuffle=False)
    test_loader = make_loader(datasets["test"], processor, args.eval_batch_size, args.workers, shuffle=False)
    eval_loaders = {"val": val_loader, "test": test_loader}

    if args.dry_run:
        encoding, metas = next(iter(train_loader))
        dry_run_summary = {
            "batch_images": len(metas),
            "pixel_values_shape": list(encoding["pixel_values"].shape),
            "first_target_boxes": int(len(encoding["labels"][0]["boxes"])),
        }
        print(json.dumps({"dry_run": dry_run_summary}, indent=2), flush=True)
        return

    model = create_model(args).to(device)
    optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, args.lr, args.backbone_lr), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    start_epoch = 1

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, device)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    if args.eval_only:
        load_checkpoint(args.eval_only, model, device)
        for split in args.eval_splits:
            metrics = evaluate(model, processor, eval_loaders[split], datasets[split], device, args.output_dir / "eval_only", split, args.fppi_threshold)
            print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        return

    best_map = -1.0
    best_checkpoint = args.output_dir / "checkpoints" / "best.pt"
    history_path = args.output_dir / "metrics_history.jsonl"
    if history_path.exists() and not args.resume:
        history_path.unlink()

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch, args.print_freq)
        scheduler.step()

        val_metrics: dict[str, Any] = {}
        if args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            val_metrics = evaluate(
                model,
                processor,
                val_loader,
                datasets["val"],
                device,
                args.output_dir / "eval" / f"epoch_{epoch:03d}",
                "val",
                args.fppi_threshold,
            )

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
        append_jsonl(history_path, row)
        save_checkpoint(args.output_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, epoch, val_metrics, args)
        current_map = float(val_metrics.get("mAP", -1.0))
        if current_map > best_map:
            best_map = current_map
            save_checkpoint(best_checkpoint, model, optimizer, scheduler, epoch, val_metrics, args)

    if best_checkpoint.exists():
        load_checkpoint(best_checkpoint, model, device)

    final_metrics: dict[str, Any] = {}
    for split in args.eval_splits:
        final_metrics[split] = evaluate(model, processor, eval_loaders[split], datasets[split], device, args.output_dir / "final", split, args.fppi_threshold)
    (args.output_dir / "final_metrics.json").write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = args.output_dir / "final_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "mAP",
            "mAP50",
            "mAP75",
            "mAP_tiny",
            "mAP_small_rel",
            "mAP_medium_rel",
            "mAP_large_rel",
            "AR100",
            "image_AUROC",
            "image_AP",
            "FPR@95TPR",
            "in_domain_FPR@95TPR",
            "out_domain_FPR@95TPR",
            "detections",
            "negative_fppi",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split, metrics in final_metrics.items():
            writer.writerow(
                {
                    "split": split,
                    "mAP": metrics.get("mAP"),
                    "mAP50": metrics.get("mAP50"),
                    "mAP75": metrics.get("mAP75"),
                    "mAP_tiny": metrics.get("mAP_tiny"),
                    "mAP_small_rel": metrics.get("mAP_small_rel"),
                    "mAP_medium_rel": metrics.get("mAP_medium_rel"),
                    "mAP_large_rel": metrics.get("mAP_large_rel"),
                    "AR100": metrics.get("AR100"),
                    "image_AUROC": metrics.get("image_AUROC"),
                    "image_AP": metrics.get("image_AP"),
                    "FPR@95TPR": metrics.get("FPR@95TPR"),
                    "in_domain_FPR@95TPR": metrics.get("in_domain_FPR@95TPR"),
                    "out_domain_FPR@95TPR": metrics.get("out_domain_FPR@95TPR"),
                    "detections": metrics.get("detections"),
                    "negative_fppi": metrics.get("false_positives_on_negatives", {}).get("fppi"),
                }
            )


if __name__ == "__main__":
    main()
