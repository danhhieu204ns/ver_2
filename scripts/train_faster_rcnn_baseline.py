#!/usr/bin/env python3
"""Train and evaluate torchvision detection baselines on the local dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn,
    ssd300_vgg16,
)
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.ops import misc as misc_nn_ops
from torchvision.transforms import functional as F

from baseline_eval_utils import build_detection_metrics, write_image_scores_csv
from detection_augmentation import DetectionAugmentation, augment_pil_detection


CLASS_ID = 1
CLASS_NAME = "nine_dash_line"
SPLITS = ("train", "val", "test")
MODEL_CHOICES = ("fasterrcnn_r50", "fasterrcnn_r101", "ssd300_vgg16")


def load_annotation(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalized_record(record: dict[str, Any], annotation_path: Path, group: str) -> dict[str, Any]:
    file_name = str(record.get("file_name", ""))
    return {
        "group": group,
        "split": str(record.get("split", "")),
        "image_id": str(record.get("image_id", "")),
        "file_name": file_name,
        "image_path": annotation_path.parent / file_name,
        "width": int(record.get("width") or 0),
        "height": int(record.get("height") or 0),
        "objects": record.get("objects") or [],
    }


def load_records(data_root: Path, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for annotation_path in sorted(data_root.glob("*/annotations.json")):
        group = annotation_path.parent.name
        for record in load_annotation(annotation_path):
            if str(record.get("split", "")) != split:
                continue
            item = normalized_record(record, annotation_path, group)
            if not item["image_path"].is_file():
                raise FileNotFoundError(f"Missing image referenced by annotation: {item['image_path']}")
            records.append(item)
    records.sort(key=lambda item: (item["group"], item["file_name"]))
    return records


def limit_records(records: list[dict[str, Any]], max_images: int | None) -> list[dict[str, Any]]:
    if max_images is None or len(records) <= max_images:
        return records

    positives = [record for record in records if record["objects"]]
    negatives = [record for record in records if not record["objects"]]
    selected: list[dict[str, Any]] = []

    if positives and negatives and max_images >= 2:
        positive_target = min(len(positives), max(1, max_images // 2))
        negative_target = min(len(negatives), max_images - positive_target)
        selected.extend(positives[:positive_target])
        selected.extend(negatives[:negative_target])
    else:
        selected.extend(records[:max_images])

    if len(selected) < max_images:
        selected_ids = {id(record) for record in selected}
        for record in records:
            if id(record) in selected_ids:
                continue
            selected.append(record)
            if len(selected) == max_images:
                break

    selected.sort(key=lambda item: (item["group"], item["file_name"]))
    return selected


def bbox_xywh_to_xyxy(bbox: Any, image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x, y, width, height = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    x1 = clip(x, 0.0, float(image_width))
    y1 = clip(y, 0.0, float(image_height))
    x2 = clip(x + width, 0.0, float(image_width))
    y2 = clip(y + height, 0.0, float(image_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


class NineDashDetectionDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        max_images: int | None = None,
        augmentation: DetectionAugmentation | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        self.data_root = data_root
        self.split = split
        self.augmentation = augmentation or DetectionAugmentation()
        self.records = limit_records(load_records(data_root, split), max_images)
        self.coco_ids = list(range(1, len(self.records) + 1))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")

        image_width, image_height = image.size
        boxes: list[tuple[float, float, float, float]] = []
        labels: list[int] = []
        areas: list[float] = []
        for obj in record["objects"]:
            if not isinstance(obj, dict):
                continue
            box = bbox_xywh_to_xyxy(obj.get("bbox"), image_width, image_height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            boxes.append(box)
            labels.append(CLASS_ID)
            areas.append((x2 - x1) * (y2 - y1))

        if self.split == "train":
            image, boxes = augment_pil_detection(image, boxes, self.augmentation)

        image_tensor = F.to_tensor(image)
        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.int64)
            area_tensor = torch.tensor(areas, dtype=torch.float32)
        else:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            label_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor([self.coco_ids[index]], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": torch.zeros((len(box_tensor),), dtype=torch.int64),
        }
        return image_tensor, target

    def record_for_coco_id(self, coco_id: int) -> dict[str, Any]:
        return self.records[self.coco_ids.index(coco_id)]

def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_fasterrcnn_r50(pretrained: bool, min_size: int, max_size: int, score_threshold: float) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.roi_heads.score_thresh = score_threshold
    return model


def create_fasterrcnn_r101(pretrained: bool, min_size: int, max_size: int, score_threshold: float) -> torch.nn.Module:
    # torchvision does not ship a COCO Faster R-CNN R101 checkpoint. This uses
    # ImageNet-pretrained ResNet-101 as the FPN backbone when pretrained=True.
    weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if pretrained else torch.nn.BatchNorm2d
    backbone = resnet101(weights=weights, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_layers=3)
    model = FasterRCNN(backbone, num_classes=2, min_size=min_size, max_size=max_size)
    model.roi_heads.score_thresh = score_threshold
    return model


def create_ssd300_vgg16(pretrained: bool, score_threshold: float) -> torch.nn.Module:
    if not pretrained:
        return ssd300_vgg16(
            weights=None,
            weights_backbone=None,
            num_classes=2,
            score_thresh=score_threshold,
        )

    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT, score_thresh=score_threshold)
    old_head = model.head.classification_head
    in_channels = [module.in_channels for module in old_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes=2)
    return model


def create_model(model_name: str, pretrained: bool, min_size: int, max_size: int, score_threshold: float) -> torch.nn.Module:
    if model_name == "fasterrcnn_r50":
        return create_fasterrcnn_r50(pretrained, min_size, max_size, score_threshold)
    if model_name == "fasterrcnn_r101":
        return create_fasterrcnn_r101(pretrained, min_size, max_size, score_threshold)
    if model_name == "ssd300_vgg16":
        return create_ssd300_vgg16(pretrained, score_threshold)
    raise ValueError(f"Unsupported model: {model_name}")


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int,
    clip_grad_norm: float | None,
) -> dict[str, float]:
    model.train()
    loss_totals: Counter[str] = Counter()
    start = time.time()

    for step, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        if not torch.isfinite(losses):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}: {losses.item()}")

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        if clip_grad_norm is not None and clip_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at epoch {epoch}, step {step}: {grad_norm.item()}")
        optimizer.step()

        loss_totals["loss"] += float(losses.detach().cpu())
        for name, loss_value in loss_dict.items():
            loss_totals[name] += float(loss_value.detach().cpu())

        if print_freq > 0 and (step == 1 or step % print_freq == 0 or step == len(loader)):
            elapsed = time.time() - start
            avg_loss = loss_totals["loss"] / step
            print(f"epoch={epoch} step={step}/{len(loader)} loss={avg_loss:.4f} elapsed={elapsed:.1f}s", flush=True)

    return {name: value / max(len(loader), 1) for name, value in loss_totals.items()}


def coco_metrics_from_stats(stats: np.ndarray) -> dict[str, float]:
    names = [
        "mAP",
        "mAP50",
        "mAP75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    return {name: float(value) for name, value in zip(names, stats.tolist())}


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: NineDashDetectionDataset,
    device: torch.device,
    output_dir: Path,
    split: str,
    fppi_threshold: float,
) -> dict[str, Any]:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []

    for images, targets in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)
        for target, output in zip(targets, outputs):
            coco_id = int(target["image_id"].item())
            boxes = output["boxes"].detach().cpu()
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()

            for box, score, label in zip(boxes, scores, labels):
                if int(label.item()) != CLASS_ID:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box.tolist()]
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                if width <= 0 or height <= 0:
                    continue
                score_value = float(score.item())
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


def make_loader(dataset: Dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )


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


def dataset_summary(datasets: dict[str, NineDashDetectionDataset]) -> dict[str, Any]:
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
    parser.add_argument(
        "--model",
        default="fasterrcnn_r50",
        choices=MODEL_CHOICES,
        help="torchvision detector baseline to train.",
    )
    parser.add_argument("--data-root", default=Path("data"), type=Path, help="Dataset root containing group folders.")
    parser.add_argument(
        "--output-dir",
        default=Path("results/baselines/faster_rcnn_r50"),
        type=Path,
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument("--epochs", default=50, type=int, help="Training epochs.")
    parser.add_argument("--batch-size", default=8, type=int, help="Training batch size.")
    parser.add_argument("--eval-batch-size", default=8, type=int, help="Evaluation batch size.")
    parser.add_argument("--workers", default=8, type=int, help="DataLoader workers.")
    parser.add_argument("--lr", default=0.005, type=float, help="Initial learning rate.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum.")
    parser.add_argument("--weight-decay", default=0.0005, type=float, help="SGD weight decay.")
    parser.add_argument(
        "--clip-grad-norm",
        default=None,
        type=float,
        help="Clip gradient norm after backward. Defaults to 10.0 for SSD300; use 0 to disable.",
    )
    parser.add_argument("--lr-step-size", default=5, type=int, help="StepLR step size.")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="StepLR gamma.")
    parser.add_argument("--min-size", default=800, type=int, help="Faster R-CNN transform min_size. Ignored by SSD.")
    parser.add_argument("--max-size", default=1333, type=int, help="Faster R-CNN transform max_size. Ignored by SSD.")
    parser.add_argument("--model-score-threshold", default=0.001, type=float, help="Low inference threshold retained for COCO/operating-point metrics.")
    parser.add_argument("--fppi-threshold", default=0.25, type=float, help="Score threshold for false-positive-per-image reporting.")
    parser.add_argument("--hflip-prob", default=0.5, type=float, help="Training horizontal-flip probability.")
    parser.add_argument("--aug-brightness", default=0.2, type=float, help="Symmetric training brightness jitter.")
    parser.add_argument("--aug-saturation", default=0.2, type=float, help="Symmetric training saturation jitter.")
    parser.add_argument("--aug-hue", default=0.015, type=float, help="Symmetric training hue jitter.")
    parser.add_argument("--protocol-name", default="canonical_v2", help="Experiment protocol recorded in config.json.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--print-freq", default=50, type=int, help="Training log frequency in steps.")
    parser.add_argument("--eval-every", default=1, type=int, help="Run validation every N epochs.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained weights. R50 and SSD use COCO detector weights; R101 uses an ImageNet backbone.",
    )
    parser.add_argument("--resume", default=None, type=Path, help="Resume checkpoint.")
    parser.add_argument("--eval-only", default=None, type=Path, help="Only evaluate a checkpoint.")
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=SPLITS, help="Splits used for final/eval-only evaluation.")
    parser.add_argument("--max-train-images", default=None, type=int, help="Limit training images for smoke tests.")
    parser.add_argument("--max-val-images", default=None, type=int, help="Limit validation images for smoke tests.")
    parser.add_argument("--max-test-images", default=None, type=int, help="Limit test images for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Load datasets and one batch, then exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clip_grad_norm is not None and args.clip_grad_norm < 0:
        raise ValueError("--clip-grad-norm must be non-negative")
    if args.clip_grad_norm is None and args.model == "ssd300_vgg16":
        args.clip_grad_norm = 10.0
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_augmentation = DetectionAugmentation(args.hflip_prob, args.aug_brightness, args.aug_saturation, args.aug_hue)
    datasets = {
        "train": NineDashDetectionDataset(args.data_root, "train", args.max_train_images, augmentation=train_augmentation),
        "val": NineDashDetectionDataset(args.data_root, "val", args.max_val_images),
        "test": NineDashDetectionDataset(args.data_root, "test", args.max_test_images),
    }
    summary = dataset_summary(datasets)
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"device": str(device), "model": args.model, "dataset": summary}, ensure_ascii=False, indent=2), flush=True)

    train_loader = make_loader(datasets["train"], args.batch_size, args.workers, shuffle=True)
    val_loader = make_loader(datasets["val"], args.eval_batch_size, args.workers, shuffle=False)
    test_loader = make_loader(datasets["test"], args.eval_batch_size, args.workers, shuffle=False)
    eval_loaders = {"val": val_loader, "test": test_loader}

    if args.dry_run:
        images, targets = next(iter(train_loader))
        dry_run_summary = {
            "batch_images": len(images),
            "first_image_shape": list(images[0].shape),
            "first_target_boxes": int(len(targets[0]["boxes"])),
        }
        print(json.dumps({"dry_run": dry_run_summary}, indent=2), flush=True)
        return

    model = create_model(
        model_name=args.model,
        pretrained=not args.no_pretrained,
        min_size=args.min_size,
        max_size=args.max_size,
        score_threshold=args.model_score_threshold,
    ).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
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
            metrics = evaluate(
                model,
                eval_loaders[split],
                datasets[split],
                device,
                args.output_dir / "eval_only",
                split,
                args.fppi_threshold,
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        return

    best_map = -1.0
    best_checkpoint = args.output_dir / "checkpoints" / "best.pt"
    history_path = args.output_dir / "metrics_history.jsonl"
    if history_path.exists() and not args.resume:
        history_path.unlink()
    if not args.resume:
        # Model/head construction consumes RNG; reset so equivalent fresh runs
        # share DataLoader, augmentation, RPN, and RoI sampling streams.
        set_seed(args.seed)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            args.print_freq,
            args.clip_grad_norm,
        )
        scheduler.step()

        val_metrics: dict[str, Any] = {}
        if args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            val_metrics = evaluate(
                model,
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
        map_value = val_metrics.get("mAP")
        current_map = float(map_value) if map_value is not None else -1.0
        if current_map > best_map:
            best_map = current_map
            save_checkpoint(best_checkpoint, model, optimizer, scheduler, epoch, val_metrics, args)

    if best_checkpoint.exists():
        load_checkpoint(best_checkpoint, model, device)

    final_metrics: dict[str, Any] = {}
    for split in args.eval_splits:
        final_metrics[split] = evaluate(
            model,
            eval_loaders[split],
            datasets[split],
            device,
            args.output_dir / "final",
            split,
            args.fppi_threshold,
        )
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
