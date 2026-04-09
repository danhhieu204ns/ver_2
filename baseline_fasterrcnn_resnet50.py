import csv
import os
import sys
from contextlib import contextmanager
from datetime import datetime

import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from multiprocessing import freeze_support

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


class NineDashLineDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile)
        self._transform = transform

    def __getitem__(self, idx):
        img, annotations = super().__getitem__(idx)

        img = np.array(img)
        image_id = self.ids[idx]

        boxes = [obj["bbox"] for obj in annotations]
        labels = [obj["category_id"] for obj in annotations]

        if len(boxes) == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = []
        else:
            boxes = np.array(boxes, dtype=np.float32)
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]

            valid_boxes = []
            valid_labels = []
            for box, label in zip(boxes, labels):
                if box[2] > box[0] + 0.5 and box[3] > box[1] + 0.5:
                    valid_boxes.append(box)
                    valid_labels.append(label)

            boxes = np.array(valid_boxes, dtype=np.float32) if valid_boxes else np.empty((0, 4), dtype=np.float32)
            labels = valid_labels

        if self._transform is not None:
            transformed = self._transform(
                image=img,
                bboxes=boxes.tolist(),
                class_labels=labels,
            )
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "image_size": torch.tensor([img.shape[-2], img.shape[-1]]),
        }

        return img, target


def get_train_transform():
    return A.Compose(
        [
            A.Resize(height=600, width=600),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_area=10,
            min_visibility=0.1,
            label_fields=["class_labels"],
            clip=True,
        ),
    )


def get_valid_transform():
    return A.Compose(
        [
            A.Resize(height=600, width=600),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_area=10,
            min_visibility=0.1,
            label_fields=["class_labels"],
            clip=True,
        ),
    )


def collate_fn(batch):
    images, targets = zip(*batch)
    valid_images, valid_targets = [], []

    for img, tgt in zip(images, targets):
        boxes = tgt["boxes"]
        labels = tgt["labels"]

        # Keep negative samples (no boxes), drop malformed targets.
        if boxes.ndim == 2 and boxes.shape[-1] == 4 and labels.ndim == 1 and boxes.shape[0] == labels.shape[0]:
            valid_images.append(img)
            valid_targets.append(tgt)

    if not valid_images:
        return [], []

    return valid_images, valid_targets


def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        min_size=600,
        max_size=600,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self):
        if hasattr(self.streams[0], "fileno"):
            return self.streams[0].fileno()
        raise OSError("fileno is not available")


@contextmanager
def redirect_output_to_log(log_path):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def metric_to_float(metric_value):
    if isinstance(metric_value, torch.Tensor):
        if metric_value.numel() == 0:
            return 0.0
        if metric_value.numel() == 1:
            return float(metric_value.item())
        return float(metric_value.mean().item())
    if metric_value is None:
        return 0.0
    return float(metric_value)


def extract_map_mar(results):
    map_score = metric_to_float(results.get("map", 0.0))
    mar_score = 0.0
    for mar_key in ("mar_100", "mar_10", "mar_1"):
        if mar_key in results:
            mar_score = metric_to_float(results[mar_key])
            break
    return map_score, mar_score


def save_metric_history_csv(history, csv_path):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "map", "mar"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved metric history: {csv_path}")


def plot_map_mar(history, plot_path, title):
    if not history:
        print("No metric history to plot")
        return
    if plt is None:
        print("matplotlib is not available, skipping plot")
        return

    epochs = [row["epoch"] for row in history]
    map_values = [row["map"] for row in history]
    mar_values = [row["mar"] for row in history]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, map_values, marker="o", linewidth=2, label="mAP")
    plt.plot(epochs, mar_values, marker="s", linewidth=2, label="mAR")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()
    print(f"Saved metric plot: {plot_path}")


def train_one_epoch(model, optimizer, data_loader, device, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    num_steps = 0

    loop = tqdm(data_loader, desc=f"Epoch {epoch + 1} Training")
    for images, targets in loop:
        if not images:
            continue

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_value for loss_value in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_steps += 1
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    print(f"Epoch {epoch + 1} - avg train loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    metric.to(device)

    print("\nValidation")
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        if not images:
            continue

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)
        metric.update(predictions, targets)

    results = metric.compute()

    print("Validation metrics")
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                print(f"  {k}: {v.item():.4f}")
            elif v.numel() > 1:
                print(f"  {k}: {v.tolist()}")
            else:
                print(f"  {k}: 0.0000")
        else:
            print(f"  {k}: {v}")

    return results


def main():
    train_img_dir = r"D:\ver2\nine-dash-line-coco-2-5\train"
    train_ann_file = r"D:\ver2\nine-dash-line-coco-2-5\train\_annotations.coco.json"
    valid_img_dir = r"D:\ver2\nine-dash-line-coco-2-5\valid"
    valid_ann_file = r"D:\ver2\nine-dash-line-coco-2-5\valid\_annotations.coco.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    num_classes = 2
    num_epochs = 30
    batch_size = 16
    num_workers = 4
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("Preparing data...")
    dataset_train = NineDashLineDataset(
        root=train_img_dir,
        annFile=train_ann_file,
        transform=get_train_transform(),
    )
    dataset_valid = NineDashLineDataset(
        root=valid_img_dir,
        annFile=valid_ann_file,
        transform=get_valid_transform(),
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )
    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )

    print(f"Loaded {len(dataset_train)} train images and {len(dataset_valid)} valid images")

    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        steps_per_epoch=len(data_loader_train),
        epochs=num_epochs,
        pct_start=0.2,
    )

    best_map = 0.0
    checkpoint_path = os.path.join(output_dir, "baseline_fasterrcnn_resnet50_best.pth")
    metric_history = []

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, scheduler=scheduler)

        results = evaluate(model, data_loader_valid, device)
        map_score, mar_score = extract_map_mar(results)
        metric_history.append({"epoch": epoch + 1, "map": map_score, "mar": mar_score})
        print(f"Epoch {epoch + 1} - mAP: {map_score:.4f} - mAR: {mar_score:.4f}")

        if map_score > best_map:
            best_map = map_score
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")

    metrics_csv_path = os.path.join(output_dir, "baseline_metrics.csv")
    metrics_plot_path = os.path.join(output_dir, "baseline_map_mar.png")
    save_metric_history_csv(metric_history, metrics_csv_path)
    plot_map_mar(metric_history, metrics_plot_path, "Baseline Faster R-CNN ResNet50")

    print("\nTraining finished")
    print(f"Best mAP: {best_map:.4f}")
    best_mar = max((row["mar"] for row in metric_history), default=0.0)
    print(f"Best mAR: {best_mar:.4f}")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_outputs")
    log_path = os.path.join(root_output_dir, f"baseline_train_{timestamp}.log")

    with redirect_output_to_log(log_path):
        print(f"Logging to: {log_path}")
        freeze_support()
        main()
