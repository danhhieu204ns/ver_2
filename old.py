# !pip -q install torch torchvision
# !pip -q install torchmetrics albumentations numpy tqdm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import box_iou, batched_nms, clip_boxes_to_image, remove_small_boxes
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import os
import csv
import sys
import torchvision.transforms as T
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from typing import List, Dict, Tuple
from contextlib import contextmanager
from datetime import datetime
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
        super(NineDashLineDataset, self).__init__(root, annFile)
        self._transform = transform

    def __getitem__(self, idx):
        img, target = super(NineDashLineDataset, self).__getitem__(idx)
        
        img = np.array(img)
        image_id = self.ids[idx]
        annotations = target
        
        boxes = [obj['bbox'] for obj in annotations]
        labels = [obj['category_id'] for obj in annotations]
        
        if len(boxes) == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = []
        else:
            boxes = np.array(boxes, dtype=np.float32)
            if boxes.ndim == 2 and boxes.shape[1] == 4:
                boxes[:, 2] += boxes[:, 0]  # x_max = x_min + width
                boxes[:, 3] += boxes[:, 1]  # y_max = y_min + height
                valid_boxes = []
                valid_labels = []
                for box, label in zip(boxes, labels):
                    if box[2] > box[0] + 0.5 and box[3] > box[1] + 0.5:
                        valid_boxes.append(box)
                        valid_labels.append(label)
                    else:
                        pass
                boxes = np.array(valid_boxes, dtype=np.float32) if valid_boxes else np.empty((0, 4), dtype=np.float32)
                labels = valid_labels
            else:
                boxes = np.empty((0, 4), dtype=np.float32)
                labels = []

        aug_data = {
            'image': img,
            'bboxes': boxes.tolist(),
            'class_labels': labels
        }
        
        if self._transform is not None:
            try:
                aug_data = self._transform(**aug_data)
            except Exception as e:
                aug_data = {'image': img, 'bboxes': boxes.tolist(), 'class_labels': labels}
        
        img = aug_data['image']
        boxes = aug_data['bboxes']
        labels = aug_data['class_labels']
        
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = []
        
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        
        final_target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'image_size': torch.tensor([img.shape[-2], img.shape[-1]])
        }
        
        return img, final_target

def get_train_transform():
    return A.Compose(
        [
            A.Resize(height=600, width=600),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(format='pascal_voc', min_area=10, min_visibility=0.1, label_fields=['class_labels'], clip=True)
    )

def get_valid_transform():
    return A.Compose(
        [
            A.Resize(height=600, width=600),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(format='pascal_voc', min_area=10, min_visibility=0.1, label_fields=['class_labels'], clip=True)
    )


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    @property
    def encoding(self):
        return getattr(self.streams[0], 'encoding', 'utf-8')

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)

    def fileno(self):
        if hasattr(self.streams[0], 'fileno'):
            return self.streams[0].fileno()
        raise OSError('fileno is not available')


@contextmanager
def redirect_output_to_log(log_path):
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(log_path, 'w', encoding='utf-8') as log_file:
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
    map_score = metric_to_float(results.get('map', 0.0))
    mar_score = 0.0
    for mar_key in ('mar_100', 'mar_10', 'mar_1'):
        if mar_key in results:
            mar_score = metric_to_float(results[mar_key])
            break
    return map_score, mar_score


def save_metric_history_csv(history, csv_path):
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['epoch', 'map', 'mar'])
        writer.writeheader()
        writer.writerows(history)
    print(f"Đã lưu lịch sử metric: {csv_path}")


def plot_map_mar(history, plot_path, title):
    if not history:
        print('Không có metric để vẽ biểu đồ')
        return
    if plt is None:
        print('Chưa có matplotlib, bỏ qua bước vẽ biểu đồ')
        return

    epochs = [row['epoch'] for row in history]
    map_values = [row['map'] for row in history]
    mar_values = [row['mar'] for row in history]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, map_values, marker='o', linewidth=2, label='mAP')
    plt.plot(epochs, mar_values, marker='s', linewidth=2, label='mAR')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()
    print(f"Đã lưu biểu đồ metric: {plot_path}")



class CustomRoIHeads(RoIHeads):
    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = clip_boxes_to_image(boxes, image_shape)

            if boxes.dim() == 2:
                boxes = boxes.reshape(scores.shape[0], -1, 4)

            labels = torch.arange(num_classes, device=device).view(1, -1).expand_as(scores)

            # Drop background class (label 0) before thresholding and NMS.
            if scores.shape[1] <= 1:
                all_boxes.append(torch.empty((0, 4), device=device))
                all_scores.append(torch.empty((0,), device=device))
                all_labels.append(torch.empty((0,), dtype=torch.int64, device=device))
                continue

            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            scores, labels = scores.reshape(-1), labels.reshape(-1)
            boxes = boxes.reshape(-1, 4)
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            keep = remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            keep = batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        detections = []
        for boxes, scores, labels in zip(all_boxes, all_scores, all_labels):
            detections.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels
            })

        return detections

    def forward(self, features: Dict[str, torch.Tensor], proposals: List[torch.Tensor], image_shapes: List[Tuple[int, int]], targets: List[Dict[str, torch.Tensor]] = None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        detections: List[Dict[str, torch.Tensor]] = []
        losses = {}
        refined_boxes_for_kd_list = None

        if self.training:
            if labels is None or regression_targets is None:
                raise ValueError("labels and regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
            
            refined_boxes = self.box_coder.decode(box_regression, proposals)
            refined_scores = F.softmax(class_logits, -1)
            split_sizes = [len(p) for p in proposals]

            if refined_boxes.dim() == 2:
                refined_boxes = refined_boxes.reshape(refined_scores.shape[0], -1, 4)

            if refined_boxes.dim() != 3 or refined_boxes.shape[-1] != 4:
                refined_boxes_for_kd_list = [torch.empty((0, 4), device=class_logits.device) for _ in proposals]
            else:
                boxes_per_image = refined_boxes.split(split_sizes, dim=0)
                scores_per_image = refined_scores.split(split_sizes, dim=0)
                refined_boxes_for_kd_list = []

                for boxes_img, scores_img, image_shape in zip(boxes_per_image, scores_per_image, image_shapes):
                    if boxes_img.numel() == 0 or scores_img.shape[1] <= 1:
                        refined_boxes_for_kd_list.append(torch.empty((0, 4), device=class_logits.device))
                        continue

                    fg_scores, fg_labels = scores_img[:, 1:].max(dim=1)
                    fg_labels = fg_labels + 1
                    proposal_indices = torch.arange(boxes_img.shape[0], device=boxes_img.device)
                    selected_boxes = boxes_img[proposal_indices, fg_labels]
                    selected_boxes = clip_boxes_to_image(selected_boxes, image_shape)

                    keep = torch.where(
                        (fg_scores >= self.score_thresh)
                        & (selected_boxes[:, 2] > selected_boxes[:, 0] + 0.5)
                        & (selected_boxes[:, 3] > selected_boxes[:, 1] + 0.5)
                    )[0]

                    if keep.numel() == 0:
                        refined_boxes_for_kd_list.append(torch.empty((0, 4), device=boxes_img.device))
                    else:
                        refined_boxes_for_kd_list.append(selected_boxes[keep])

        else:
            detections = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)

        return detections, losses, refined_boxes_for_kd_list

class EnhancedFasterRCNN(FasterRCNN):
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)

        detections, detector_losses, refined_boxes_for_kd = self.roi_heads(features, proposals, images.image_sizes, targets)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses, features, detections, refined_boxes_for_kd, proposals
        
        return detections

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(
        weights='DEFAULT',
        rpn_pre_nms_top_n_train=10000,
        rpn_post_nms_top_n_train=6000, 
        rpn_nms_thresh=0.1, 
        rpn_score_thresh=0.0,
        min_size=600,
        max_size=600
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    new_box_predictor = FastRCNNPredictor(in_features, num_classes)
    box_roi_pool = model.roi_heads.box_roi_pool
    box_head = model.roi_heads.box_head
    
    custom_heads = CustomRoIHeads(
        box_roi_pool=box_roi_pool,
        box_head=box_head,
        box_predictor=new_box_predictor,
        fg_iou_thresh=model.roi_heads.proposal_matcher.high_threshold,
        bg_iou_thresh=model.roi_heads.proposal_matcher.low_threshold,
        batch_size_per_image=128,  
        positive_fraction=0.25,
        bbox_reg_weights=model.roi_heads.box_coder.weights,
        score_thresh=0.05,
        nms_thresh=0.3,
        detections_per_img=100
    )

    model.roi_heads = custom_heads
    model.__class__ = EnhancedFasterRCNN
    return model



class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, beta1=1.0, beta2=0.5, dino_patch_size=14, iou_threshold=0.2): 
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.iou_threshold = iou_threshold
        
        print("Đang tải mô hình DINOv2 cho KD Loss...")
        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        for param in self.dino_encoder.parameters():
            param.requires_grad = False
            
        self.dino_input_size = 224
        self.dino_patch_size = dino_patch_size

    def _match_predictions_to_ground_truth(self, pred_boxes, gt_boxes, iou_threshold):
        if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0 or pred_boxes.shape[1] != 4 or gt_boxes.shape[1] != 4:
            return [], []
            
        iou_matrix = box_iou(pred_boxes, gt_boxes)
        max_iou_per_gt, max_iou_indices_per_gt = iou_matrix.max(dim=0)
        
        matched_pred_indices = []
        matched_gt_indices = []
        
        for i, iou in enumerate(max_iou_per_gt):
            if iou >= iou_threshold:
                pred_idx = max_iou_indices_per_gt[i].item()
                if pred_idx not in matched_pred_indices:
                    matched_pred_indices.append(pred_idx)
                    matched_gt_indices.append(i)
        
        if not matched_pred_indices:
            pass
        
        return matched_pred_indices, matched_gt_indices

    def _extract_embeddings(self, images, boxes_list, image_size):
        cropped_images = []
        img = images[0]
        for box in boxes_list[0]:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_size[1], x2), min(image_size[0], y2)
            if x2 > x1 + 0.5 and y2 > y1 + 0.5:
                crop = img[:, y1:y2, x1:x2]
                try:
                    resized_crop = T.functional.resize(crop, [self.dino_input_size, self.dino_input_size])
                    cropped_images.append(resized_crop)
                except Exception as e:
                    continue
            else:
                pass
        if not cropped_images:
            return torch.tensor([], device=images.device)
            
        batch = torch.stack(cropped_images)
        
        with torch.no_grad():
            embeddings = self.dino_encoder(batch)
        
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, original_images, predictions, targets, batch_idx=0):
        self.dino_encoder.eval()
        
        all_gt_embeddings = []
        all_pred_embeddings = []

        for img_idx, (img, pred, target) in enumerate(zip(original_images, predictions, targets)):
            pred_boxes = pred['boxes']
            gt_boxes = target['boxes']
            image_size = target['image_size']
            
            if pred_boxes.dim() != 2 or pred_boxes.shape[1] != 4 or gt_boxes.dim() != 2 or gt_boxes.shape[1] != 4:
                continue
            
            pred_boxes = clip_boxes_to_image(pred_boxes, image_size)
            gt_boxes = clip_boxes_to_image(gt_boxes, image_size)
            
            matched_pred_indices, matched_gt_indices = self._match_predictions_to_ground_truth(
                pred_boxes, gt_boxes, iou_threshold=self.iou_threshold
            )
            
            if not matched_pred_indices:
                continue

            matched_pred_boxes = pred_boxes[matched_pred_indices]
            matched_gt_boxes = gt_boxes[matched_gt_indices]

            img_batch = img.unsqueeze(0)
            
            gt_embeddings = self._extract_embeddings(img_batch, [matched_gt_boxes], image_size)
            pred_embeddings = self._extract_embeddings(img_batch, [matched_pred_boxes], image_size)
            
            if gt_embeddings.numel() == 0 or pred_embeddings.numel() == 0:
                continue
            
            all_gt_embeddings.append(gt_embeddings)
            all_pred_embeddings.append(pred_embeddings)

        if not all_gt_embeddings or not all_pred_embeddings:
            return torch.tensor(0.0, device=original_images[0].device)

        final_gt_embeddings = torch.cat(all_gt_embeddings, dim=0)
        final_pred_embeddings = torch.cat(all_pred_embeddings, dim=0)

        if final_gt_embeddings.shape[0] != final_pred_embeddings.shape[0]:
            return torch.tensor(0.0, device=original_images[0].device)

        loss_l1 = F.l1_loss(final_pred_embeddings, final_gt_embeddings)
        SR = torch.matmul(final_pred_embeddings, final_pred_embeddings.T)
        SI = torch.matmul(final_gt_embeddings, final_gt_embeddings.T)
        loss_irm = torch.norm(SR - SI, p='fro')

        total_kd_loss = self.beta1 * loss_l1 + self.beta2 * loss_irm
        return total_kd_loss
    


def collate_fn(batch):
    images, targets = zip(*batch)
    valid_images, valid_targets = [], []
    for img, tgt in zip(images, targets):
        boxes = tgt['boxes']
        labels = tgt['labels']

        # Keep negative samples (no boxes) but drop malformed targets.
        if boxes.ndim == 2 and boxes.shape[-1] == 4 and labels.ndim == 1 and boxes.shape[0] == labels.shape[0]:
            valid_images.append(img)
            valid_targets.append(tgt)
        else:
            pass
    if not valid_images:
        return [], []
    return valid_images, valid_targets

def train_one_epoch(model, optimizer, data_loader, device, epoch, kd_loss_fn, scheduler=None):
    model.train()
    kd_loss_fn.to(device)
    
    total_main_loss = 0
    total_kd_loss = 0
    
    loop = tqdm(data_loader, desc=f"Epoch {epoch+1} Training")
    for i, (images, targets) in enumerate(loop):
        if not images:
            continue
            
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict, features, final_predictions, refined_boxes_for_kd, proposals = model(images, targets)
        
        main_losses = sum(loss for loss in loss_dict.values())
        
        if refined_boxes_for_kd is not None:

            pass
        
        predictions_for_kd = []
        if refined_boxes_for_kd is not None:
            for refined_boxes_tensor in refined_boxes_for_kd:
                if refined_boxes_tensor.numel() > 0 and refined_boxes_tensor.shape[-1] == 4:
                    valid_boxes = refined_boxes_tensor[
                        (refined_boxes_tensor[:, 2] > refined_boxes_tensor[:, 0] + 0.5) & 
                        (refined_boxes_tensor[:, 3] > refined_boxes_tensor[:, 1] + 0.5)
                    ]
                    predictions_for_kd.append({'boxes': valid_boxes if valid_boxes.shape[0] > 0 else torch.empty((0, 4), device=device)})
                else:
                    predictions_for_kd.append({'boxes': torch.empty((0, 4), device=device)})

        kd_loss = kd_loss_fn(images, predictions_for_kd, targets, batch_idx=i)
        
        total_loss = main_losses + kd_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        loop.set_postfix(main_loss=main_losses.item(), kd_loss=kd_loss.item(), total_loss=total_loss.item())
                
        total_main_loss += main_losses.item()
        if isinstance(kd_loss, torch.Tensor):
            total_kd_loss += kd_loss.item()
        
    avg_main_loss = total_main_loss / len(data_loader)
    avg_kd_loss = total_kd_loss / len(data_loader)
    print(f"--- Kết thúc Epoch {epoch+1} Training ---")
    print(f"  Loss chính trung bình: {avg_main_loss:.4f}")
    print(f"  KD Loss trung bình: {avg_kd_loss:.4f}")
    print(f"  Tổng Loss trung bình: {avg_main_loss + avg_kd_loss:.4f}")

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy')
    metric.to(device)

    print("\n--- Bắt đầu Đánh giá (Validation) ---")
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        if not images:
            continue
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        try:
            predictions = model(images)
            metric.update(predictions, targets)
        except Exception as e:
            continue

    results = metric.compute()
    print(f"--- Kết quả Đánh giá ---")
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                print(f"  {k}: {v.item():.4f}")
            elif v.numel() > 1:
                print(f"  {k}: {v.tolist()}")
            else:
                print(f"  {k}: 0.0000 (empty tensor)")
        else:
            print(f"  {k}: {v}")
    
    return results



def main():
    TRAIN_IMG_DIR = r"D:\ver2\nine-dash-line-coco-2-5\train"
    TRAIN_ANN_FILE = r"D:\ver2\nine-dash-line-coco-2-5\train\_annotations.coco.json"
    VALID_IMG_DIR = r"D:\ver2\nine-dash-line-coco-2-5\valid"
    VALID_ANN_FILE = r"D:\ver2\nine-dash-line-coco-2-5\valid\_annotations.coco.json"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    num_classes = 2
    num_epochs = 30
    batch_size = 16
    num_workers = 4
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_outputs')
    os.makedirs(output_dir, exist_ok=True)

    beta_pairs = [(1.0, 0.1)]

    print("Đang chuẩn bị dữ liệu...")
    dataset_train = NineDashLineDataset(
        root=TRAIN_IMG_DIR,
        annFile=TRAIN_ANN_FILE,
        transform=get_train_transform()
    )
    dataset_valid = NineDashLineDataset(
        root=VALID_IMG_DIR,
        annFile=VALID_ANN_FILE,
        transform=get_valid_transform()
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0)
    )
    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0)
    )

    print(f"Đã tải {len(dataset_train)} ảnh huấn luyện và {len(dataset_valid)} ảnh validation.")

    results_summary = []

    for beta1, beta2 in beta_pairs:
        print("\n" + "="*20 + f" HUẤN LUYỆN VỚI beta1={beta1}, beta2={beta2} " + "="*20)
        run_tag = f"beta1_{str(beta1).replace('.', '_')}_beta2_{str(beta2).replace('.', '_')}"
        metric_history = []
        
        print("Đang khởi tạo mô hình...")
        model = get_model(num_classes)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=0.0001, weight_decay=0.0001)
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.0001,
            steps_per_epoch=len(data_loader_train),
            epochs=num_epochs,
            pct_start=0.2  # Tăng warmup
        )
        
        kd_loss_fn = KnowledgeDistillationLoss(beta1=beta1, beta2=beta2, iou_threshold=0.2)
        kd_loss_fn.to(device)
        
        best_map = 0.0
        best_mar = 0.0
        checkpoint_path = ""
        for epoch in range(num_epochs):
            train_one_epoch(model, optimizer, data_loader_train, device, epoch, kd_loss_fn, scheduler=scheduler)
            
            results = evaluate(model, data_loader_valid, device)
            map_score, mar_score = extract_map_mar(results)
            metric_history.append({'epoch': epoch + 1, 'map': map_score, 'mar': mar_score})
            print(f"Epoch {epoch+1} - mAP: {map_score:.4f} - mAR: {mar_score:.4f}")
            best_mar = max(best_mar, mar_score)
            
            if map_score > best_map:
                best_map = map_score
                checkpoint_path = os.path.join(output_dir, f'model_{run_tag}_best.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Đã lưu checkpoint tốt nhất: {checkpoint_path}")

        metrics_csv_path = os.path.join(output_dir, f'metrics_{run_tag}.csv')
        metrics_plot_path = os.path.join(output_dir, f'metrics_{run_tag}_map_mar.png')
        save_metric_history_csv(metric_history, metrics_csv_path)
        plot_map_mar(metric_history, metrics_plot_path, f'KD Faster R-CNN ({run_tag})')
        
        results_summary.append({
            'beta1': beta1,
            'beta2': beta2,
            'map': best_map,
            'mar': best_mar,
            'checkpoint': checkpoint_path if checkpoint_path else 'no-checkpoint-saved'
        })

    print("\n" + "="*20 + " TÓM TẮT KẾT QUẢ " + "="*20)
    for res in results_summary:
        print(
            f"beta1={res['beta1']}, beta2={res['beta2']}: "
            f"mAP={res['map']:.4f}, mAR={res['mar']:.4f}, Checkpoint={res['checkpoint']}"
        )

    print("\n" + "="*20 + " HOÀN THÀNH HUẤN LUYỆN " + "="*20)


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    root_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_outputs')
    log_path = os.path.join(root_output_dir, f'kd_train_{timestamp}.log')

    with redirect_output_to_log(log_path):
        print(f"Đang ghi log vào: {log_path}")
        freeze_support()
        main()