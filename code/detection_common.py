import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm


class AuditCocoDetection(CocoDetection):
    def __init__(self, root, ann_file):
        super().__init__(root, ann_file)
        cats = self.coco.loadCats(self.coco.getCatIds())
        cat_ids = sorted(cat["id"] for cat in cats)
        self.cat_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}

    def __getitem__(self, idx):
        img, annotations = super().__getitem__(idx)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")
        image = torch.as_tensor(np.asarray(img), dtype=torch.float32).permute(2, 0, 1) / 255.0

        image_id = self.ids[idx]
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for obj in annotations:
            x, y, w, h = obj["bbox"]
            if w <= 0.5 or h <= 0.5:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[int(obj["category_id"])])
            areas.append(float(obj.get("area", w * h)))
            iscrowd.append(int(obj.get("iscrowd", 0)))

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            areas = torch.empty((0,), dtype=torch.float32)
            iscrowd = torch.empty((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
            "image_size": torch.tensor([image.shape[-2], image.shape[-1]], dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def move_targets_to_device(targets, device):
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def get_faster_rcnn_r50_fpn(num_classes=2, pretrained=True, min_size=800, max_size=1333):
    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights, min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_detector_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    loss_sums = {}
    loop = tqdm(data_loader, desc=f"Epoch {epoch} train")
    for images, targets in loop:
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        for name, value in loss_dict.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.item())
        loop.set_postfix(loss=f"{loss.item():.4f}")

    denom = max(1, len(data_loader))
    result = {"train_loss": total_loss / denom}
    for name, value in loss_sums.items():
        result[f"train_{name}"] = value / denom
    return result


@torch.no_grad()
def evaluate_map(model, data_loader, device, desc="eval"):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    for images, targets in tqdm(data_loader, desc=desc):
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)
        predictions = model(images)
        predictions = [{k: v.detach().cpu() for k, v in pred.items()} for pred in predictions]
        eval_targets = []
        for target in targets:
            eval_targets.append(
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                    "area": target["area"].detach().cpu(),
                    "iscrowd": target["iscrowd"].detach().cpu(),
                }
            )
        metric.update(predictions, eval_targets)
    return metric.compute()


@torch.no_grad()
def export_coco_predictions(model, data_loader, device, output_path):
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for images, targets in tqdm(data_loader, desc=f"predict {output_path.name}"):
        images = [image.to(device) for image in images]
        predictions = model(images)
        for pred, target in zip(predictions, targets):
            image_id = int(target["image_id"].item())
            boxes = pred["boxes"].detach().cpu().tolist()
            labels = pred["labels"].detach().cpu().tolist()
            scores = pred["scores"].detach().cpu().tolist()
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box
                records.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    }
                )
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return output_path


def metric_value(results, key, default=0.0):
    value = results.get(key, default)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().cpu().tolist()
    return value


def json_safe(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def create_run_dir(base_dir, run_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name)
    run_dir = Path(base_dir) / f"{safe_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_json_log(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def append_csv_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_datasets(train_ann, val_ann, image_root):
    return (
        AuditCocoDetection(root=image_root, ann_file=train_ann),
        AuditCocoDetection(root=image_root, ann_file=val_ann),
    )


def build_optimizer(model, lr, weight_decay):
    return optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
