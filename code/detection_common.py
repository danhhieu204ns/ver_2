import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models import ResNet101_Weights
from torchvision.models.detection import (
    FasterRCNN,
    fasterrcnn_resnet50_fpn,
    fcos_resnet50_fpn,
    retinanet_resnet50_fpn,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from tqdm import tqdm


NEGATIVE_LABEL_TYPES = {"in_domain_neg", "out_domain_neg", "hard_neg"}
POSITIVE_LABEL_TYPES = {"positive_real", "synthetic_pos", "positive_synthetic"}


class AuditCocoDetection(Dataset):
    def __init__(self, root, ann_file):
        self.root = Path(root)
        self.ann_file = Path(ann_file)
        with self.ann_file.open("r", encoding="utf-8") as f:
            self.coco = json.load(f)

        self.images = list(self.coco.get("images", []))
        self.ids = [int(image["id"]) for image in self.images]
        self.image_by_id = {int(image["id"]): image for image in self.images}
        self.anns_by_image = {image_id: [] for image_id in self.ids}
        for ann in self.coco.get("annotations", []):
            self.anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        cats = sorted(self.coco.get("categories", []), key=lambda item: int(item["id"]))
        cat_ids = [int(cat["id"]) for cat in cats]
        self.cat_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}
        self.label_to_cat_id = {label: cat_id for cat_id, label in self.cat_id_to_label.items()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = int(image_info["id"])
        image_path = self.root / image_info["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image referenced by COCO: {image_path}")

        img = Image.open(image_path).convert("RGB")
        image = torch.as_tensor(np.array(img, copy=True), dtype=torch.float32).permute(2, 0, 1) / 255.0

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        width = float(image_info.get("width", image.shape[-1]))
        height = float(image_info.get("height", image.shape[-2]))
        for obj in self.anns_by_image.get(image_id, []):
            x, y, w, h = [float(v) for v in obj["bbox"]]
            x1 = max(0.0, min(width, x))
            y1 = max(0.0, min(height, y))
            x2 = max(0.0, min(width, x + w))
            y2 = max(0.0, min(height, y + h))
            if x2 <= x1 + 0.5 or y2 <= y1 + 0.5:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_label[int(obj["category_id"])])
            areas.append(float(obj.get("area", (x2 - x1) * (y2 - y1))))
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


def set_random_seed(seed, deterministic=False):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, batch_size, shuffle, num_workers, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if seed is not None else None,
        generator=generator,
    )


def move_targets_to_device(targets, device):
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def get_faster_rcnn_r50_fpn(num_classes=2, pretrained=True, min_size=800, max_size=1333):
    return get_torchvision_detector(
        "faster_rcnn_r50_fpn",
        num_classes=num_classes,
        pretrained=pretrained,
        min_size=min_size,
        max_size=max_size,
    )


def get_torchvision_detector(architecture, num_classes=2, pretrained=True, min_size=800, max_size=1333):
    architecture = architecture.lower()
    weights = "DEFAULT" if pretrained else None

    if architecture == "faster_rcnn_r50_fpn":
        kwargs = {"weights": weights, "min_size": min_size, "max_size": max_size}
        if not pretrained:
            kwargs["weights_backbone"] = None
        model = fasterrcnn_resnet50_fpn(**kwargs)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    if architecture == "faster_rcnn_r101_fpn":
        backbone_weights = ResNet101_Weights.DEFAULT if pretrained else None
        backbone = resnet_fpn_backbone(
            backbone_name="resnet101",
            weights=backbone_weights,
            trainable_layers=3,
        )
        return FasterRCNN(
            backbone,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
        )

    if architecture == "retinanet_r50_fpn":
        model = retinanet_resnet50_fpn(weights=weights, weights_backbone=None, min_size=min_size, max_size=max_size)
        in_channels = model.backbone.out_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        return model

    if architecture == "fcos_r50_fpn":
        model = fcos_resnet50_fpn(weights=weights, weights_backbone=None, min_size=min_size, max_size=max_size)
        in_channels = model.backbone.out_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = FCOSClassificationHead(in_channels, num_anchors, num_classes)
        return model

    raise ValueError(f"Unsupported torchvision detector architecture: {architecture}")


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
    metrics, _ = evaluate_map_and_predictions(model, data_loader, device, desc=desc, keep_predictions=False)
    return metrics


def label_to_category_id(dataset, label):
    current = dataset
    while current is not None:
        mapping = getattr(current, "label_to_cat_id", None)
        if mapping is not None:
            return int(mapping.get(int(label), int(label)))
        current = getattr(current, "dataset", None)
    return int(label)


def predictions_to_coco_records(predictions, targets, dataset):
    records = []
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
                    "category_id": label_to_category_id(dataset, label),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                }
            )
    return records


@torch.no_grad()
def evaluate_map_and_predictions(model, data_loader, device, desc="eval", keep_predictions=True):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    records = []
    for images, targets in tqdm(data_loader, desc=desc):
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)
        predictions = model(images)
        predictions = [{k: v.detach().cpu() for k, v in pred.items()} for pred in predictions]
        if keep_predictions:
            records.extend(predictions_to_coco_records(predictions, targets, data_loader.dataset))
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
    return metric.compute(), records


@torch.no_grad()
def collect_coco_predictions(model, data_loader, device, desc="predict"):
    model.eval()
    records = []
    for images, targets in tqdm(data_loader, desc=desc):
        images = [image.to(device) for image in images]
        predictions = model(images)
        predictions = [{k: v.detach().cpu() for k, v in pred.items()} for pred in predictions]
        records.extend(predictions_to_coco_records(predictions, targets, data_loader.dataset))
    return records


@torch.no_grad()
def export_coco_predictions(model, data_loader, device, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = collect_coco_predictions(model, data_loader, device, desc=f"predict {output_path.name}")
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


def load_coco_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def max_score_by_image(predictions):
    scores = {}
    for pred in predictions:
        image_id = str(pred.get("image_id"))
        score = float(pred.get("score", 0.0))
        scores[image_id] = max(scores.get(image_id, 0.0), score)
    return scores


def summarize_image_level(coco, predictions, threshold):
    scores = max_score_by_image(predictions)
    positive_image_ids = {ann["image_id"] for ann in coco.get("annotations", [])}
    rows = {}
    totals = {
        "images": 0,
        "positive_images": 0,
        "negative_images": 0,
        "flagged": 0,
        "false_positive": 0,
        "true_positive": 0,
    }

    for image in coco.get("images", []):
        label_type = image.get("label_type", "")
        has_gt = image["id"] in positive_image_ids
        is_positive = has_gt or label_type in POSITIVE_LABEL_TYPES
        group = "positive" if is_positive else label_type or "negative"
        row = rows.setdefault(
            group,
            {"images": 0, "flagged": 0, "false_positive": 0, "true_positive": 0},
        )
        row["images"] += 1
        totals["images"] += 1
        totals["positive_images" if is_positive else "negative_images"] += 1

        flagged = scores.get(str(image["id"]), 0.0) >= threshold
        if flagged:
            row["flagged"] += 1
            totals["flagged"] += 1
            if is_positive:
                row["true_positive"] += 1
                totals["true_positive"] += 1
            else:
                row["false_positive"] += 1
                totals["false_positive"] += 1

    for row in rows.values():
        denom = max(1, row["images"])
        row["flag_rate"] = row["flagged"] / denom
        row["fpr"] = row["false_positive"] / denom
        row["recall_image"] = row["true_positive"] / denom

    precision = totals["true_positive"] / max(1, totals["true_positive"] + totals["false_positive"])
    recall = totals["true_positive"] / max(1, totals["positive_images"])
    fpr = totals["false_positive"] / max(1, totals["negative_images"])
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    rows["overall"] = {
        **totals,
        "precision_image": precision,
        "recall_image": recall,
        "f1_image": f1,
        "fpr": fpr,
    }
    return rows


def choose_threshold_for_target_fpr(coco, predictions, target_fpr):
    scores = max_score_by_image(predictions)
    candidates = {0.0, 1.0}
    for score in scores.values():
        candidates.add(score)
        candidates.add(score + 1e-12)
    candidates.add(max(scores.values(), default=1.0) + 1e-6)

    best_threshold = 1.0
    best_recall = -1.0
    best_fpr = float("inf")
    for threshold in sorted(candidates):
        summary = summarize_image_level(coco, predictions, threshold)
        overall = summary["overall"]
        fpr = overall["fpr"]
        recall = overall["recall_image"]
        if fpr <= target_fpr and (recall > best_recall or (recall == best_recall and fpr < best_fpr)):
            best_threshold = threshold
            best_recall = recall
            best_fpr = fpr
    return best_threshold
