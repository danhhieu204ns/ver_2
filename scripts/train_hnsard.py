#!/usr/bin/env python3
"""Train HN-SARD and its loss ablations on the nine-dash-line dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as nnF
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from train_faster_rcnn_baseline import (
    CLASS_ID,
    SPLITS,
    NineDashDetectionDataset,
    append_jsonl,
    create_model,
    dataset_summary,
    evaluate,
    make_loader,
    save_checkpoint,
    set_seed,
)
from detection_augmentation import DetectionAugmentation


FASTER_RCNN_CHOICES = ("fasterrcnn_r50", "fasterrcnn_r101")


def expand_box(
    box: torch.Tensor,
    image_width: int,
    image_height: int,
    context_scale: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    new_width = width * context_scale
    new_height = height * context_scale
    nx1 = max(0, int(math.floor(cx - new_width * 0.5)))
    ny1 = max(0, int(math.floor(cy - new_height * 0.5)))
    nx2 = min(image_width, int(math.ceil(cx + new_width * 0.5)))
    ny2 = min(image_height, int(math.ceil(cy + new_height * 0.5)))
    if nx2 <= nx1:
        nx2 = min(image_width, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(image_height, ny1 + 1)
    return nx1, ny1, nx2, ny2


def build_context_crops(
    images: list[torch.Tensor],
    image_indices: list[int],
    boxes: torch.Tensor,
    context_scale: float,
    crop_size: int,
) -> torch.Tensor:
    crops: list[torch.Tensor] = []
    for image_index, box in zip(image_indices, boxes):
        image = images[image_index]
        _, image_height, image_width = image.shape
        x1, y1, x2, y2 = expand_box(box, image_width, image_height, context_scale)
        crop = image[:, y1:y2, x1:x2]
        if crop.numel() == 0:
            crop = image.new_zeros((3, crop_size, crop_size))
        else:
            crop = nnF.interpolate(
                crop.unsqueeze(0),
                size=(crop_size, crop_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        crops.append(crop.clamp(0.0, 1.0))

    if not crops:
        device = images[0].device if images else boxes.device
        return torch.empty((0, 3, crop_size, crop_size), device=device)
    return torch.stack(crops, dim=0)


class DummyTeacher(torch.nn.Module):
    """Small deterministic teacher for smoke tests without downloading DINOv2."""

    def __init__(self, dim: int, crop_size: int, context_scale: float) -> None:
        super().__init__()
        self.dim = dim
        self.crop_size = crop_size
        self.context_scale = context_scale
        generator = torch.Generator().manual_seed(13)
        projection = torch.randn(3 * 8 * 8, dim, generator=generator) / math.sqrt(3 * 8 * 8)
        self.register_buffer("projection", projection)

    @torch.no_grad()
    def encode_regions(
        self,
        images: list[torch.Tensor],
        image_indices: list[int],
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        crops = build_context_crops(images, image_indices, boxes, self.context_scale, self.crop_size)
        if crops.numel() == 0:
            return crops.new_empty((0, self.dim))
        pooled = nnF.adaptive_avg_pool2d(crops, (8, 8)).flatten(1)
        return pooled @ self.projection.to(device=pooled.device, dtype=pooled.dtype)


class TransformersDinoTeacher(torch.nn.Module):
    """Frozen DINOv2 teacher loaded through Hugging Face transformers."""

    def __init__(
        self,
        model_name: str,
        crop_size: int,
        context_scale: float,
        crop_batch_size: int,
        local_files_only: bool,
        fp16: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.dim = int(getattr(self.model.config, "hidden_size"))
        self.crop_size = crop_size
        self.context_scale = context_scale
        self.crop_batch_size = crop_batch_size
        self.fp16 = fp16 and device.type == "cuda"

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        try:
            processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=local_files_only)
            image_mean = list(getattr(processor, "image_mean", image_mean))
            image_std = list(getattr(processor, "image_std", image_std))
        except Exception as exc:  # pragma: no cover - fallback is deterministic and safe.
            print(f"Warning: using ImageNet normalization because processor loading failed: {exc}", flush=True)

        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1))

    @torch.no_grad()
    def encode_regions(
        self,
        images: list[torch.Tensor],
        image_indices: list[int],
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        crops = build_context_crops(images, image_indices, boxes, self.context_scale, self.crop_size)
        if crops.numel() == 0:
            return crops.new_empty((0, self.dim))

        embeddings: list[torch.Tensor] = []
        for chunk in crops.split(self.crop_batch_size):
            pixel_values = (chunk - self.image_mean.to(chunk.device)) / self.image_std.to(chunk.device)
            with torch.autocast(device_type=pixel_values.device.type, dtype=torch.float16, enabled=self.fp16):
                outputs = self.model(pixel_values=pixel_values)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            embeddings.append(pooled.float())
        return torch.cat(embeddings, dim=0)


class EmbeddingQueue:
    def __init__(self, dim: int, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.embeddings = torch.empty((0, dim), device=device)

    def enqueue(self, values: torch.Tensor) -> None:
        if self.capacity <= 0 or values.numel() == 0:
            return
        values = values.detach()
        self.embeddings = torch.cat([self.embeddings, values], dim=0)[-self.capacity :]

    def get(self) -> torch.Tensor:
        return self.embeddings.detach()


def build_teacher(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module | None, int]:
    if args.teacher_backend == "none":
        return None, args.teacher_dim
    if args.teacher_backend == "dummy":
        teacher = DummyTeacher(args.teacher_dim, args.teacher_crop_size, args.teacher_context_scale).to(device)
        return teacher, teacher.dim
    if args.teacher_backend == "transformers":
        teacher = TransformersDinoTeacher(
            model_name=args.teacher_model,
            crop_size=args.teacher_crop_size,
            context_scale=args.teacher_context_scale,
            crop_batch_size=args.teacher_crop_batch_size,
            local_files_only=args.teacher_local_files_only,
            fp16=args.teacher_fp16,
            device=device,
        )
        return teacher, teacher.dim
    raise ValueError(f"Unsupported teacher backend: {args.teacher_backend}")


def add_projection_head(model: torch.nn.Module, teacher_dim: int, device: torch.device) -> None:
    roi_dim = int(model.roi_heads.box_predictor.cls_score.in_features)
    model.hnsard_projection = torch.nn.Linear(roi_dim, teacher_dim).to(device)


def roi_box_features(
    model: torch.nn.Module,
    features: OrderedDict[str, torch.Tensor],
    boxes_by_image: list[torch.Tensor],
    image_sizes: list[tuple[int, int]],
) -> torch.Tensor:
    total_boxes = sum(int(len(boxes)) for boxes in boxes_by_image)
    in_features = int(model.hnsard_projection.in_features)
    device = next(iter(features.values())).device
    if total_boxes == 0:
        return torch.empty((0, in_features), device=device)
    pooled = model.roi_heads.box_roi_pool(features, boxes_by_image, image_sizes)
    box_features = model.roi_heads.box_head(pooled)
    if box_features.shape[1] != in_features:
        raise RuntimeError(
            f"roi_box_features shape mismatch: box_head output dim {box_features.shape[1]} "
            f"!= hnsard_projection.in_features {in_features}. "
            "Ensure add_projection_head() is called after the detector is fully constructed."
        )
    return box_features


def scale_weights_for_boxes(
    boxes: torch.Tensor,
    image: torch.Tensor,
    use_scale_aware: bool,
    args: argparse.Namespace,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_empty((0,))
    if not use_scale_aware:
        return boxes.new_ones((len(boxes),))

    _, image_height, image_width = image.shape
    image_area = max(float(image_height * image_width), 1.0)
    areas = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    ratios = areas / image_area
    weights = boxes.new_full((len(boxes),), float(args.large_weight))
    weights = torch.where(ratios < 0.25, boxes.new_full(weights.shape, float(args.medium_weight)), weights)
    weights = torch.where(ratios < 0.05, boxes.new_full(weights.shape, float(args.small_weight)), weights)
    weights = torch.where(ratios < 0.01, boxes.new_full(weights.shape, float(args.tiny_weight)), weights)
    return weights


def collect_positive_regions(
    original_images: list[torch.Tensor],
    original_targets: list[dict[str, torch.Tensor]],
    transformed_targets: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[list[int], torch.Tensor, list[torch.Tensor], torch.Tensor]:
    image_indices: list[int] = []
    original_boxes: list[torch.Tensor] = []
    transformed_boxes_by_image: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []

    for image_index, (image, original_target, transformed_target) in enumerate(
        zip(original_images, original_targets, transformed_targets)
    ):
        boxes = original_target["boxes"]
        transformed_boxes = transformed_target["boxes"]
        if len(boxes) == 0:
            transformed_boxes_by_image.append(transformed_boxes.new_empty((0, 4)))
            continue

        if args.max_positive_regions_per_image > 0 and len(boxes) > args.max_positive_regions_per_image:
            selected = torch.randperm(len(boxes), device=boxes.device)[: args.max_positive_regions_per_image]
        else:
            selected = torch.arange(len(boxes), device=boxes.device)

        selected_original = boxes[selected]
        selected_transformed = transformed_boxes[selected]
        image_indices.extend([image_index] * len(selected))
        original_boxes.append(selected_original)
        transformed_boxes_by_image.append(selected_transformed)
        weights.append(scale_weights_for_boxes(selected_original, image, args.scale_aware, args))

    if original_boxes:
        flat_original_boxes = torch.cat(original_boxes, dim=0)
        flat_weights = torch.cat(weights, dim=0)
    else:
        device = original_images[0].device
        flat_original_boxes = torch.empty((0, 4), device=device)
        flat_weights = torch.empty((0,), device=device)
    return image_indices, flat_original_boxes, transformed_boxes_by_image, flat_weights


def positive_distillation_loss(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    original_images: list[torch.Tensor],
    original_targets: list[dict[str, torch.Tensor]],
    transformed_targets: list[dict[str, torch.Tensor]],
    features: OrderedDict[str, torch.Tensor],
    image_sizes: list[tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    image_indices, original_boxes, transformed_boxes_by_image, weights = collect_positive_regions(
        original_images,
        original_targets,
        transformed_targets,
        args,
    )
    device = next(iter(features.values())).device
    if original_boxes.numel() == 0:
        dim = int(model.hnsard_projection.out_features)
        return device_zero(device), torch.empty((0, dim), device=device), 0

    student_features = roi_box_features(model, features, transformed_boxes_by_image, image_sizes)
    student_embeddings = model.hnsard_projection(student_features)
    teacher_embeddings = teacher.encode_regions(original_images, image_indices, original_boxes)

    student_embeddings = nnF.normalize(student_embeddings, dim=1)
    teacher_embeddings = nnF.normalize(teacher_embeddings, dim=1)
    distances = 1.0 - (student_embeddings * teacher_embeddings).sum(dim=1)
    loss = (distances * weights).sum() / weights.sum().clamp_min(1.0)
    return loss, teacher_embeddings.detach(), int(len(original_boxes))


def device_zero(device: torch.device) -> torch.Tensor:
    return torch.tensor(0.0, device=device)


def hard_negative_features(
    model: torch.nn.Module,
    features: OrderedDict[str, torch.Tensor],
    proposals: list[torch.Tensor],
    transformed_targets: list[dict[str, torch.Tensor]],
    image_sizes: list[tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, int]:
    candidate_boxes: list[torch.Tensor] = []
    candidate_counts: list[int] = []
    for proposal, target in zip(proposals, transformed_targets):
        proposal = proposal[: args.hard_negative_preselect]
        gt_boxes = target["boxes"]
        if len(proposal) > 0 and len(gt_boxes) > 0:
            max_iou = box_iou(proposal, gt_boxes).max(dim=1).values
            proposal = proposal[max_iou < args.negative_iou_threshold]
        candidate_boxes.append(proposal)
        candidate_counts.append(int(len(proposal)))

    if sum(candidate_counts) == 0:
        device = next(iter(features.values())).device
        return torch.empty((0, int(model.hnsard_projection.in_features)), device=device), 0

    candidate_features = roi_box_features(model, features, candidate_boxes, image_sizes)
    class_logits, _ = model.roi_heads.box_predictor(candidate_features)
    foreground_scores = class_logits.softmax(dim=1)[:, CLASS_ID].detach()

    selected_indices: list[torch.Tensor] = []
    offset = 0
    for count in candidate_counts:
        if count == 0:
            continue
        local_scores = foreground_scores[offset : offset + count]
        keep = min(args.hard_negatives_per_image, count)
        local_indices = torch.topk(local_scores, k=keep).indices + offset
        selected_indices.append(local_indices)
        offset += count

    if not selected_indices:
        device = next(iter(features.values())).device
        return torch.empty((0, int(model.hnsard_projection.in_features)), device=device), 0

    flat_indices = torch.cat(selected_indices, dim=0)
    return candidate_features[flat_indices], int(len(flat_indices))


def contrastive_loss(
    model: torch.nn.Module,
    negative_features: torch.Tensor,
    teacher_anchors: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if negative_features.numel() == 0 or teacher_anchors.numel() == 0:
        return device_zero(negative_features.device)

    negative_embeddings = nnF.normalize(model.hnsard_projection(negative_features), dim=1)
    teacher_anchors = nnF.normalize(teacher_anchors.detach(), dim=1)
    similarities = negative_embeddings @ teacher_anchors.t()
    hardest_similarity = similarities.max(dim=1).values
    if args.contrastive_loss == "hinge":
        return torch.relu(hardest_similarity - args.contrastive_margin).mean()
    scaled = (hardest_similarity - args.contrastive_margin) / max(args.contrastive_temperature, 1e-6)
    return nnF.softplus(scaled).mean() * args.contrastive_temperature


def hnsard_forward_train(
    model: torch.nn.Module,
    teacher: torch.nn.Module | None,
    teacher_bank: EmbeddingQueue,
    images: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    epoch: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    transformed_images, transformed_targets = model.transform(images, targets)
    features = model.backbone(transformed_images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    proposals, proposal_losses = model.rpn(transformed_images, features, transformed_targets)
    _, detector_losses = model.roi_heads(features, proposals, transformed_images.image_sizes, transformed_targets)
    loss_dict: dict[str, torch.Tensor] = {}
    loss_dict.update(detector_losses)
    loss_dict.update(proposal_losses)

    detection_loss = sum(loss for loss in loss_dict.values())
    total_loss = detection_loss
    device = detection_loss.device

    raw_pos = device_zero(device)
    raw_con = device_zero(device)
    weighted_pos = device_zero(device)
    weighted_con = device_zero(device)
    positive_regions = 0
    hard_negatives = 0

    needs_teacher = args.lambda_pos > 0 or args.lambda_con > 0
    if needs_teacher and teacher is None:
        raise RuntimeError("HN-SARD losses require a teacher. Use --teacher-backend dummy or transformers.")

    if needs_teacher and teacher is not None:
        # Auxiliary sampling must not advance the detector RNG stream; otherwise
        # later RPN/RoI samples differ across ablation variants despite one seed.
        fork_devices = []
        if device.type == "cuda":
            fork_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=fork_devices):
            raw_pos, positive_teacher_embeddings, positive_regions = positive_distillation_loss(
                model,
                teacher,
                images,
                targets,
                transformed_targets,
                features,
                transformed_images.image_sizes,
                args,
            )
            teacher_bank.enqueue(positive_teacher_embeddings)
            if args.lambda_pos > 0:
                weighted_pos = raw_pos * args.lambda_pos
                total_loss = total_loss + weighted_pos

            contrastive_active = args.lambda_con > 0 and epoch > args.contrastive_warmup_epochs
            if contrastive_active:
                negative_features, hard_negatives = hard_negative_features(
                    model,
                    features,
                    proposals,
                    transformed_targets,
                    transformed_images.image_sizes,
                    args,
                )
                raw_con = contrastive_loss(model, negative_features, teacher_bank.get(), args)
                weighted_con = raw_con * args.lambda_con
                total_loss = total_loss + weighted_con

    loss_dict["loss_hnsard_pos"] = weighted_pos
    loss_dict["loss_hnsard_con"] = weighted_con
    loss_dict["raw_hnsard_pos"] = raw_pos.detach()
    loss_dict["raw_hnsard_con"] = raw_con.detach()
    loss_dict["hnsard_positive_regions"] = torch.tensor(float(positive_regions), device=device)
    loss_dict["hnsard_hard_negatives"] = torch.tensor(float(hard_negatives), device=device)
    return total_loss, loss_dict


def train_one_epoch_hnsard(
    model: torch.nn.Module,
    teacher: torch.nn.Module | None,
    teacher_bank: EmbeddingQueue,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int,
    clip_grad_norm: float | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    if teacher is not None:
        teacher.eval()
    loss_totals: Counter[str] = Counter()
    start = time.time()

    for step, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        losses, loss_dict = hnsard_forward_train(model, teacher, teacher_bank, images, targets, epoch, args)
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
            avg_pos = loss_totals["loss_hnsard_pos"] / step
            avg_con = loss_totals["loss_hnsard_con"] / step
            print(
                f"epoch={epoch} step={step}/{len(loader)} loss={avg_loss:.4f} "
                f"pos={avg_pos:.4f} con={avg_con:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    return {name: value / max(len(loader), 1) for name, value in loss_totals.items()}


def load_checkpoint_compatible(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
    strict_shapes: bool,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    if strict_shapes:
        model.load_state_dict(state_dict)
        return checkpoint

    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    skipped = sorted(set(state_dict) - set(compatible))
    if skipped or missing or unexpected:
        print(
            json.dumps(
                {
                    "checkpoint_load": str(path),
                    "skipped_shape_mismatch_or_unknown": skipped,
                    "missing": list(missing),
                    "unexpected": list(unexpected),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="fasterrcnn_r50", choices=FASTER_RCNN_CHOICES)
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--output-dir", default=Path("results/hnsard/full_hnsard"), type=Path)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--eval-batch-size", default=8, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=0.0005, type=float)
    parser.add_argument("--clip-grad-norm", default=None, type=float)
    parser.add_argument("--lr-step-size", default=5, type=int)
    parser.add_argument("--lr-gamma", default=0.1, type=float)
    parser.add_argument("--min-size", default=800, type=int)
    parser.add_argument("--max-size", default=1333, type=int)
    parser.add_argument("--model-score-threshold", default=0.001, type=float)
    parser.add_argument("--fppi-threshold", default=0.25, type=float)
    parser.add_argument("--hflip-prob", default=0.5, type=float)
    parser.add_argument("--aug-brightness", default=0.2, type=float)
    parser.add_argument("--aug-saturation", default=0.2, type=float)
    parser.add_argument("--aug-hue", default=0.015, type=float)
    parser.add_argument("--protocol-name", default="canonical_v2")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--print-freq", default=50, type=int)
    parser.add_argument("--eval-every", default=1, type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", default=None, type=Path)
    parser.add_argument("--eval-only", default=None, type=Path)
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=SPLITS)
    parser.add_argument("--max-train-images", default=None, type=int)
    parser.add_argument("--max-val-images", default=None, type=int)
    parser.add_argument("--max-test-images", default=None, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument(
        "--patience",
        default=15,
        type=int,
        help="Early-stopping patience in epochs (0 = disabled).",
    )

    parser.add_argument("--teacher-backend", default="transformers", choices=("none", "dummy", "transformers"))
    parser.add_argument("--teacher-model", default="facebook/dinov2-small")
    parser.add_argument("--teacher-dim", default=384, type=int, help="Projection dim for teacher-backend none/dummy.")
    parser.add_argument("--teacher-crop-size", default=224, type=int)
    parser.add_argument("--teacher-context-scale", default=1.5, type=float)
    parser.add_argument("--teacher-crop-batch-size", default=32, type=int)
    parser.add_argument("--teacher-local-files-only", action="store_true")
    parser.add_argument("--teacher-fp16", action="store_true")

    parser.add_argument("--lambda-pos", default=0.5, type=float)
    parser.add_argument("--lambda-con", default=0.1, type=float)
    parser.add_argument("--scale-aware", dest="scale_aware", action="store_true", default=True)
    parser.add_argument("--no-scale-aware", dest="scale_aware", action="store_false")
    parser.add_argument("--tiny-weight", default=2.0, type=float)
    parser.add_argument("--small-weight", default=1.5, type=float)
    parser.add_argument("--medium-weight", default=1.2, type=float)
    parser.add_argument("--large-weight", default=1.0, type=float)
    parser.add_argument("--max-positive-regions-per-image", default=16, type=int)
    parser.add_argument("--contrastive-warmup-epochs", default=3, type=int)
    parser.add_argument("--hard-negative-preselect", default=128, type=int)
    parser.add_argument("--hard-negatives-per-image", default=16, type=int)
    parser.add_argument("--negative-iou-threshold", default=0.1, type=float)
    parser.add_argument("--teacher-bank-size", default=512, type=int)
    parser.add_argument("--contrastive-loss", default="softplus", choices=("softplus", "hinge"))
    parser.add_argument("--contrastive-margin", default=0.2, type=float)
    parser.add_argument("--contrastive-temperature", default=0.1, type=float)
    return parser.parse_args()


def write_final_metrics_csv(path: Path, final_metrics: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
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


def main() -> None:
    args = parse_args()
    if args.clip_grad_norm is not None and args.clip_grad_norm < 0:
        raise ValueError("--clip-grad-norm must be non-negative")
    if args.lambda_pos < 0 or args.lambda_con < 0:
        raise ValueError("--lambda-pos and --lambda-con must be non-negative")
    if not args.eval_only and args.teacher_backend == "none" and (args.lambda_pos > 0 or args.lambda_con > 0):
        raise ValueError("--teacher-backend none is only valid when both HN-SARD loss weights are zero")

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
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "model": args.model,
                "teacher_backend": args.teacher_backend,
                "lambda_pos": args.lambda_pos,
                "lambda_con": args.lambda_con,
                "scale_aware": args.scale_aware,
                "dataset": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

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

    needs_training_teacher = not args.eval_only and (args.lambda_pos > 0 or args.lambda_con > 0)
    teacher_dim = args.teacher_dim
    model = create_model(
        model_name=args.model,
        pretrained=not args.no_pretrained,
        min_size=args.min_size,
        max_size=args.max_size,
        score_threshold=args.model_score_threshold,
    ).to(device)
    add_projection_head(model, teacher_dim, device)
    teacher, loaded_teacher_dim = build_teacher(args, device) if needs_training_teacher else (None, teacher_dim)
    if loaded_teacher_dim != teacher_dim:
        # Recreate only the auxiliary head; the detector was initialized before
        # loading the teacher so every ablation starts from identical weights.
        teacher_dim = loaded_teacher_dim
        add_projection_head(model, teacher_dim, device)

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    teacher_bank = EmbeddingQueue(teacher_dim, args.teacher_bank_size, device)
    start_epoch = 1

    if args.resume:
        checkpoint = load_checkpoint_compatible(args.resume, model, device, strict_shapes=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    if args.eval_only:
        load_checkpoint_compatible(args.eval_only, model, device, strict_shapes=False)
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
    no_improve = 0  # consecutive eval epochs without val mAP improvement
    best_checkpoint = args.output_dir / "checkpoints" / "best.pt"
    history_path = args.output_dir / "metrics_history.jsonl"
    if history_path.exists() and not args.resume:
        history_path.unlink()
    if not args.resume:
        set_seed(args.seed)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch_hnsard(
            model,
            teacher,
            teacher_bank,
            optimizer,
            train_loader,
            device,
            epoch,
            args.print_freq,
            args.clip_grad_norm,
            args,
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
        if val_metrics:
            if current_map > best_map:
                best_map = current_map
                no_improve = 0
                save_checkpoint(best_checkpoint, model, optimizer, scheduler, epoch, val_metrics, args)
            else:
                no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}: val mAP did not improve "
                    f"for {no_improve} consecutive eval epoch(s) (patience={args.patience}).",
                    flush=True,
                )
                break

    if not best_checkpoint.exists():
        save_checkpoint(best_checkpoint, model, optimizer, scheduler, args.epochs, {}, args)
    load_checkpoint_compatible(best_checkpoint, model, device, strict_shapes=True)

    if args.skip_final_eval:
        return

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
    (args.output_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_final_metrics_csv(args.output_dir / "final_metrics.csv", final_metrics)


if __name__ == "__main__":
    main()
