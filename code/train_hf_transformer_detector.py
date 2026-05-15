import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    DeformableDetrForObjectDetection,
    DetrForObjectDetection,
    RTDetrForObjectDetection,
)

from detection_common import (
    append_csv_row,
    append_json_log,
    choose_threshold_for_target_fpr,
    create_run_dir,
    json_safe,
    load_coco_dict,
    metric_value,
    set_random_seed,
    summarize_image_level,
)


MODEL_SPECS = {
    "detr_r50": {
        "family": "detr",
        "checkpoint": "facebook/detr-resnet-50",
        "model_class": DetrForObjectDetection,
    },
    "deformable_detr": {
        "family": "deformable_detr",
        "checkpoint": "SenseTime/deformable-detr",
        "model_class": DeformableDetrForObjectDetection,
    },
    "rtdetr_r18": {
        "family": "rtdetr",
        "checkpoint": "PekingU/rtdetr_r18vd",
        "model_class": RTDetrForObjectDetection,
    },
    "rtdetr_r50": {
        "family": "rtdetr",
        "checkpoint": "PekingU/rtdetr_r50vd",
        "model_class": RTDetrForObjectDetection,
    },
}


class HFCocoDetection(Dataset):
    def __init__(self, image_root, ann_file, processor, imgsz=None):
        self.image_root = Path(image_root)
        self.ann_file = Path(ann_file)
        self.processor = processor
        self.imgsz = imgsz
        with self.ann_file.open("r", encoding="utf-8") as f:
            self.coco = json.load(f)
        self.images = list(self.coco.get("images", []))
        self.ids = [int(image["id"]) for image in self.images]
        self.anns_by_image = {image_id: [] for image_id in self.ids}
        for ann in self.coco.get("annotations", []):
            self.anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
        categories = sorted(self.coco.get("categories", []), key=lambda item: int(item["id"]))
        self.cat_id_to_label = {int(category["id"]): idx for idx, category in enumerate(categories)}
        self.label_to_cat_id = {label: cat_id for cat_id, label in self.cat_id_to_label.items()}

    def __len__(self):
        return len(self.images)

    def _processor_kwargs(self):
        kwargs = {"return_tensors": "pt"}
        if self.imgsz:
            processor_name = self.processor.__class__.__name__.lower()
            if "rtdetr" in processor_name:
                kwargs["size"] = {"height": self.imgsz, "width": self.imgsz}
            else:
                kwargs["size"] = {"shortest_edge": self.imgsz, "longest_edge": self.imgsz}
        return kwargs

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = int(image_info["id"])
        image_path = self.image_root / image_info["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image referenced by COCO: {image_path}")
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        hf_annotations = []
        boxes_xyxy = []
        labels_eval = []
        areas = []
        iscrowd = []
        for ann in self.anns_by_image.get(image_id, []):
            x, y, w, h = [float(v) for v in ann["bbox"]]
            x1 = max(0.0, min(float(width), x))
            y1 = max(0.0, min(float(height), y))
            x2 = max(0.0, min(float(width), x + w))
            y2 = max(0.0, min(float(height), y + h))
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0.5 or bh <= 0.5:
                continue
            category_id = int(ann["category_id"])
            label = self.cat_id_to_label[category_id]
            area = float(ann.get("area", bw * bh))
            crowd = int(ann.get("iscrowd", 0))
            hf_annotations.append(
                {
                    "id": int(ann.get("id", len(hf_annotations))),
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": [x1, y1, bw, bh],
                    "area": area,
                    "iscrowd": crowd,
                }
            )
            boxes_xyxy.append([x1, y1, x2, y2])
            labels_eval.append(label + 1)
            areas.append(area)
            iscrowd.append(crowd)

        encoded = self.processor(
            images=image,
            annotations={"image_id": image_id, "annotations": hf_annotations},
            **self._processor_kwargs(),
        )
        item = {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"][0],
            "image_id": image_id,
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "target": {
                "boxes": torch.as_tensor(boxes_xyxy, dtype=torch.float32)
                if boxes_xyxy
                else torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.as_tensor(labels_eval, dtype=torch.int64)
                if labels_eval
                else torch.empty((0,), dtype=torch.int64),
                "area": torch.as_tensor(areas, dtype=torch.float32)
                if areas
                else torch.empty((0,), dtype=torch.float32),
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64)
                if iscrowd
                else torch.empty((0,), dtype=torch.int64),
                "image_id": torch.tensor([image_id], dtype=torch.int64),
            },
        }
        if "pixel_mask" in encoded:
            item["pixel_mask"] = encoded["pixel_mask"].squeeze(0)
        else:
            item["pixel_mask"] = None
        return item


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_masks = [item["pixel_mask"] for item in batch]
    if all(mask is not None for mask in pixel_masks):
        pixel_mask = torch.stack(pixel_masks)
    else:
        pixel_mask = None
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": [item["labels"] for item in batch],
        "targets": [item["target"] for item in batch],
        "orig_sizes": torch.stack([item["orig_size"] for item in batch]),
    }


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
        generator=generator,
    )


def move_labels_to_device(labels, device):
    moved = []
    for label in labels:
        moved.append({k: v.to(device) if torch.is_tensor(v) else v for k, v in label.items()})
    return moved


def move_targets_to_device(targets, device):
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()} for target in targets]


def forward_model(model, pixel_values, pixel_mask=None, labels=None):
    kwargs = {"pixel_values": pixel_values}
    if pixel_mask is not None:
        kwargs["pixel_mask"] = pixel_mask
    if labels is not None:
        kwargs["labels"] = labels
    try:
        return model(**kwargs)
    except TypeError:
        kwargs.pop("pixel_mask", None)
        return model(**kwargs)


def build_model_and_processor(args):
    spec = MODEL_SPECS[args.architecture]
    checkpoint = args.checkpoint or spec["checkpoint"]
    id2label = {0: "nine_dash_line"}
    label2id = {"nine_dash_line": 0}
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model_class = spec["model_class"]
    if args.no_pretrained:
        config = AutoConfig.from_pretrained(
            checkpoint,
            num_labels=args.num_classes,
            id2label=id2label,
            label2id=label2id,
        )
        model = model_class(config)
    else:
        model = model_class.from_pretrained(
            checkpoint,
            num_labels=args.num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    return model, processor, checkpoint


def build_scheduler(args, optimizer):
    if args.lr_schedule == "none":
        return None
    if args.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    if args.lr_schedule == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    raise ValueError(f"Unsupported lr schedule: {args.lr_schedule}")


def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc=f"Epoch {epoch} train")
    for batch in loop:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device) if batch["pixel_mask"] is not None else None
        labels = move_labels_to_device(batch["labels"], device)
        outputs = forward_model(model, pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        loop.set_postfix(loss=f"{loss.item():.4f}")
    denom = max(1, len(loader))
    return {"train_loss": total_loss / denom}


@torch.no_grad()
def evaluate_and_predict(model, processor, loader, device, pred_conf):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    records = []
    for batch in tqdm(loader, desc="eval"):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device) if batch["pixel_mask"] is not None else None
        targets = move_targets_to_device(batch["targets"], device)
        target_sizes = batch["orig_sizes"].to(device)
        outputs = forward_model(model, pixel_values, pixel_mask=pixel_mask)
        processed = processor.post_process_object_detection(outputs, threshold=pred_conf, target_sizes=target_sizes)

        metric_predictions = []
        metric_targets = []
        for result, target in zip(processed, targets):
            boxes = result["boxes"].detach().cpu()
            scores = result["scores"].detach().cpu()
            labels = result["labels"].detach().cpu().to(torch.int64) + 1
            metric_predictions.append({"boxes": boxes, "scores": scores, "labels": labels})
            metric_targets.append(
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                    "area": target["area"].detach().cpu(),
                    "iscrowd": target["iscrowd"].detach().cpu(),
                }
            )
            image_id = int(target["image_id"].item())
            for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                x1, y1, x2, y2 = box
                records.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    }
                )
        metric.update(metric_predictions, metric_targets)
    return metric.compute(), records


def flat_image_metrics(prefix, summary):
    overall = summary.get("overall", {})
    positive = summary.get("positive", {})
    in_domain = summary.get("in_domain_neg", {})
    out_domain = summary.get("out_domain_neg", {})
    hard = summary.get("hard_neg", {})
    return {
        f"{prefix}_image_precision": overall.get("precision_image", 0.0),
        f"{prefix}_image_recall": overall.get("recall_image", 0.0),
        f"{prefix}_image_f1": overall.get("f1_image", 0.0),
        f"{prefix}_image_fpr": overall.get("fpr", 0.0),
        f"{prefix}_positive_recall": positive.get("recall_image", 0.0),
        f"{prefix}_fpr_in_domain": in_domain.get("fpr", 0.0),
        f"{prefix}_fpr_out_domain": out_domain.get("fpr", 0.0),
        f"{prefix}_fpr_hard": hard.get("fpr", 0.0),
    }


def save_checkpoint(path, model, optimizer, scheduler, args, epoch, checkpoint_name, val_metrics, val_image_summary):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "epoch": epoch,
        "checkpoint_name": checkpoint_name,
        "val_metrics": {k: json_safe(v) for k, v in val_metrics.items()},
        "val_image_summary": val_image_summary,
    }
    torch.save(payload, path)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def write_prediction_file(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return path


def export_best_predictions(model, processor, device, args, run_dir, best_map_ckpt):
    state = load_checkpoint(best_map_ckpt, device)
    model.load_state_dict(state["model"])
    split_specs = [
        ("val", Path(args.val_ann)),
        ("test_standard", Path(args.test_ann)),
        ("test_robustness", Path(args.test_robustness_ann)),
    ]
    predictions = {}
    cocos = {}
    for split, ann_path in split_specs:
        dataset = HFCocoDetection(args.image_root, ann_path, processor, imgsz=args.imgsz)
        loader = make_loader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)
        _, records = evaluate_and_predict(model, processor, loader, device, args.pred_conf)
        pred_path = write_prediction_file(run_dir / f"{split}_predictions.json", records)
        predictions[split] = {"path": str(pred_path), "records": records}
        cocos[split] = load_coco_dict(ann_path)

    threshold = choose_threshold_for_target_fpr(cocos["val"], predictions["val"]["records"], args.target_val_fpr)
    image_metrics = {
        "threshold_source": "validation",
        "target_val_fpr": args.target_val_fpr,
        "selected_threshold": threshold,
        "checkpoint": str(best_map_ckpt),
        "splits": {},
    }
    for split in predictions:
        image_metrics["splits"][split] = summarize_image_level(cocos[split], predictions[split]["records"], threshold)
    metrics_path = run_dir / "image_level_metrics.json"
    metrics_path.write_text(json.dumps(image_metrics, indent=2), encoding="utf-8")
    return {"prediction_files": {k: v["path"] for k, v in predictions.items()}, "image_metrics": str(metrics_path)}


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2 HuggingFace transformer detector trainer.")
    parser.add_argument("--architecture", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--checkpoint", default=None, help="Override the default HuggingFace checkpoint.")
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--train-ann", default="data/audit/coco/train.json")
    parser.add_argument("--val-ann", default="data/audit/coco/val.json")
    parser.add_argument("--test-ann", default="data/audit/coco/test_standard.json")
    parser.add_argument("--test-robustness-ann", default="data/audit/coco/test_robustness.json")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=("none", "cosine", "step"), default="cosine")
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--pred-conf", type=float, default=0.001)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--target-val-fpr", type=float, default=0.01)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="results/stage2_baselines/hf_transformers")
    parser.add_argument("--export-preds", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"baseline_{args.architecture}_seed{args.seed}"
    run_dir = create_run_dir(args.output_dir, run_name)

    model, processor, checkpoint_name = build_model_and_processor(args)
    model.to(device)
    train_set = HFCocoDetection(args.image_root, args.train_ann, processor, imgsz=args.imgsz)
    val_set = HFCocoDetection(args.image_root, args.val_ann, processor, imgsz=args.imgsz)
    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, seed=args.seed)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)
    val_coco = load_coco_dict(args.val_ann)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer)
    log_json = run_dir / "training_log.json"
    log_csv = run_dir / "training_log.csv"
    best_map_ckpt = run_dir / "best_map.pth"
    best_recall_ckpt = run_dir / "best_recall_at_fpr.pth"
    records = []
    best_map = -1.0
    best_recall_at_fpr = -1.0
    best_recall_tiebreak_map = -1.0

    for epoch in range(1, args.epochs + 1):
        train_row = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_metrics, val_predictions = evaluate_and_predict(model, processor, val_loader, device, args.pred_conf)
        threshold = choose_threshold_for_target_fpr(val_coco, val_predictions, args.target_val_fpr)
        val_image_summary = summarize_image_level(val_coco, val_predictions, threshold)
        map_score = metric_value(val_metrics, "map")
        recall_at_fpr = val_image_summary["overall"]["recall_image"]

        saved_best_map = False
        saved_best_recall = False
        if map_score > best_map:
            best_map = map_score
            save_checkpoint(best_map_ckpt, model, optimizer, scheduler, args, epoch, checkpoint_name, val_metrics, val_image_summary)
            saved_best_map = True
        if recall_at_fpr > best_recall_at_fpr or (
            recall_at_fpr == best_recall_at_fpr and map_score > best_recall_tiebreak_map
        ):
            best_recall_at_fpr = recall_at_fpr
            best_recall_tiebreak_map = map_score
            save_checkpoint(best_recall_ckpt, model, optimizer, scheduler, args, epoch, checkpoint_name, val_metrics, val_image_summary)
            saved_best_recall = True

        row = {
            "epoch": epoch,
            "architecture": args.architecture,
            "checkpoint": checkpoint_name,
            "seed": args.seed,
            "lr": optimizer.param_groups[0]["lr"],
            **train_row,
            **{f"val_{k}": json_safe(v) for k, v in val_metrics.items()},
            "val_threshold_at_target_fpr": threshold,
            **flat_image_metrics("val", val_image_summary),
            "saved_best_map": saved_best_map,
            "saved_best_recall_at_fpr": saved_best_recall,
            "best_map_checkpoint": str(best_map_ckpt),
            "best_recall_at_fpr_checkpoint": str(best_recall_ckpt),
        }
        records.append(row)
        append_json_log(log_json, records)
        append_csv_row(log_csv, row)
        if scheduler is not None:
            scheduler.step()
        print(
            f"Epoch {epoch}: val mAP={map_score:.4f} best={best_map:.4f} "
            f"val recall@FPR<={args.target_val_fpr:.3f}={recall_at_fpr:.4f}"
        )

    summary = {
        "run_dir": str(run_dir),
        "architecture": args.architecture,
        "checkpoint": checkpoint_name,
        "seed": args.seed,
        "best_map": best_map,
        "best_recall_at_fpr": best_recall_at_fpr,
        "best_map_checkpoint": str(best_map_ckpt),
        "best_recall_at_fpr_checkpoint": str(best_recall_ckpt),
        "protocol": vars(args),
    }
    if args.export_preds and best_map_ckpt.is_file():
        summary["best_map_exports"] = export_best_predictions(model, processor, device, args, run_dir, best_map_ckpt)
    elif args.export_preds:
        summary["best_map_exports"] = "skipped_no_checkpoint"
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Run dir: {run_dir}")
    print(f"Best mAP checkpoint: {best_map_ckpt}")
    print(f"Best recall-at-FPR checkpoint: {best_recall_ckpt}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
