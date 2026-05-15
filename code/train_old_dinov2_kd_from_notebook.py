import argparse
import multiprocessing as mp
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import box_iou, clip_boxes_to_image
from tqdm import tqdm

from detection_common import (
    append_csv_row,
    append_json_log,
    build_datasets,
    create_run_dir,
    evaluate_map,
    export_coco_predictions,
    json_safe,
    make_loader,
    metric_value,
    move_targets_to_device,
)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class NotebookKDRoIHeads(RoIHeads):
    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        if not self.training:
            detections = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            return detections, {}, None

        if labels is None or regression_targets is None:
            raise ValueError("labels and regression_targets cannot be None")
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

        decoded_boxes = self.box_coder.decode(box_regression, proposals)
        boxes_per_image = [len(p) for p in proposals]
        decoded_boxes_list = decoded_boxes.split(boxes_per_image, 0)
        logits_list = class_logits.split(boxes_per_image, 0)
        refined_boxes = []
        for per_image_boxes, per_image_logits, image_shape in zip(decoded_boxes_list, logits_list, image_shapes):
            if per_image_boxes.numel() == 0:
                refined_boxes.append(torch.empty((0, 4), device=class_logits.device))
                continue
            if per_image_boxes.dim() == 3:
                probs = F.softmax(per_image_logits, dim=1)
                if probs.shape[1] <= 1:
                    refined_boxes.append(torch.empty((0, 4), device=class_logits.device))
                    continue
                fg_labels = probs[:, 1:].argmax(dim=1) + 1
                picked = per_image_boxes[torch.arange(per_image_boxes.shape[0], device=class_logits.device), fg_labels]
            else:
                picked = per_image_boxes
            picked = clip_boxes_to_image(picked, image_shape)
            keep = (picked[:, 2] > picked[:, 0] + 0.5) & (picked[:, 3] > picked[:, 1] + 0.5)
            refined_boxes.append(picked[keep])
        return [], losses, refined_boxes


class OldKDFasterRCNN(FasterRCNN):
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("targets should be passed in training mode")

        original_image_sizes = [img.shape[-2:] for img in images]
        transformed_images, transformed_targets = self.transform(images, targets)
        features = self.backbone(transformed_images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        proposals, proposal_losses = self.rpn(transformed_images, features, transformed_targets)
        detections, detector_losses, refined_boxes = self.roi_heads(
            features, proposals, transformed_images.image_sizes, transformed_targets
        )
        detections = self.transform.postprocess(detections, transformed_images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if self.training:
            return losses, transformed_images.tensors, transformed_targets, transformed_images.image_sizes, refined_boxes
        return detections


class NotebookDinoV2L1IRMLoss(nn.Module):
    """Old notebook KD: match DINOv2 embeddings of predicted boxes and GT boxes with L1 + IRM."""

    def __init__(self, beta1=1.0, beta2=0.1, dino_patch_size=14, iou_threshold=0.2):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.iou_threshold = iou_threshold
        self.dino_input_size = 224
        self.dino_patch_size = dino_patch_size
        self.dino_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        for param in self.dino_encoder.parameters():
            param.requires_grad = False
        self.dino_encoder.eval()

    def _match_predictions_to_ground_truth(self, pred_boxes, gt_boxes):
        if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
            return [], []
        iou_matrix = box_iou(pred_boxes, gt_boxes)
        max_iou_per_gt, max_iou_indices_per_gt = iou_matrix.max(dim=0)
        matched_pred_indices = []
        matched_gt_indices = []
        seen = set()
        for gt_idx, iou in enumerate(max_iou_per_gt):
            if iou >= self.iou_threshold:
                pred_idx = int(max_iou_indices_per_gt[gt_idx].item())
                if pred_idx not in seen:
                    seen.add(pred_idx)
                    matched_pred_indices.append(pred_idx)
                    matched_gt_indices.append(gt_idx)
        return matched_pred_indices, matched_gt_indices

    def _extract_embeddings(self, image, boxes, image_size):
        crops = []
        h, w = int(image_size[0]), int(image_size[1])
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box.detach().tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 + 0.5 or y2 <= y1 + 0.5:
                continue
            crop = image[:, y1:y2, x1:x2]
            crops.append(TF.resize(crop, [self.dino_input_size, self.dino_input_size], antialias=True))
        if not crops:
            return torch.empty((0, 768), device=image.device)
        batch = torch.stack(crops, dim=0)
        mean = IMAGENET_MEAN.to(batch.device)
        std = IMAGENET_STD.to(batch.device)
        batch = (batch - mean) / std
        with torch.no_grad():
            embeddings = self.dino_encoder(batch)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, batched_images, refined_predictions, targets, image_shapes):
        self.dino_encoder.eval()
        all_gt_embeddings = []
        all_pred_embeddings = []

        for img_idx, (pred_boxes, target, image_shape) in enumerate(zip(refined_predictions, targets, image_shapes)):
            gt_boxes = target["boxes"]
            if pred_boxes.dim() != 2 or pred_boxes.shape[-1] != 4:
                continue
            if gt_boxes.dim() != 2 or gt_boxes.shape[-1] != 4:
                continue
            image_size = tuple(int(v) for v in image_shape)
            pred_boxes = clip_boxes_to_image(pred_boxes, image_size)
            gt_boxes = clip_boxes_to_image(gt_boxes, image_size)
            matched_pred_indices, matched_gt_indices = self._match_predictions_to_ground_truth(pred_boxes, gt_boxes)
            if not matched_pred_indices:
                continue

            h, w = image_size
            image = batched_images[img_idx, :, :h, :w]
            matched_pred_boxes = pred_boxes[matched_pred_indices]
            matched_gt_boxes = gt_boxes[matched_gt_indices]
            pred_embeddings = self._extract_embeddings(image, matched_pred_boxes, image_size)
            gt_embeddings = self._extract_embeddings(image, matched_gt_boxes, image_size)
            if pred_embeddings.numel() == 0 or gt_embeddings.numel() == 0:
                continue
            if pred_embeddings.shape[0] != gt_embeddings.shape[0]:
                continue
            all_pred_embeddings.append(pred_embeddings)
            all_gt_embeddings.append(gt_embeddings)

        if not all_pred_embeddings or not all_gt_embeddings:
            return torch.tensor(0.0, device=batched_images.device)

        pred_embeddings = torch.cat(all_pred_embeddings, dim=0)
        gt_embeddings = torch.cat(all_gt_embeddings, dim=0)
        loss_l1 = F.l1_loss(pred_embeddings, gt_embeddings)
        sr = torch.matmul(pred_embeddings, pred_embeddings.T)
        si = torch.matmul(gt_embeddings, gt_embeddings.T)
        loss_irm = torch.norm(sr - si, p="fro")
        return self.beta1 * loss_l1 + self.beta2 * loss_irm


def _roi_attr(roi_heads, name, default, alt_names=None):
    for key in [name] + (alt_names or []):
        if hasattr(roi_heads, key):
            return getattr(roi_heads, key)
    return default


def get_old_kd_model(num_classes, pretrained, min_size, max_size):
    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        min_size=min_size,
        max_size=max_size,
        rpn_pre_nms_top_n_train=10000,
        rpn_post_nms_top_n_train=6000,
        rpn_nms_thresh=0.1,
        rpn_score_thresh=0.0,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    predictor = FastRCNNPredictor(in_features, num_classes)
    roi_heads = model.roi_heads
    matcher = getattr(roi_heads, "proposal_matcher", None)
    fg_iou_thresh = getattr(matcher, "high_threshold", 0.5)
    bg_iou_thresh = getattr(matcher, "low_threshold", 0.5)
    model.roi_heads = NotebookKDRoIHeads(
        box_roi_pool=roi_heads.box_roi_pool,
        box_head=roi_heads.box_head,
        box_predictor=predictor,
        fg_iou_thresh=fg_iou_thresh,
        bg_iou_thresh=bg_iou_thresh,
        batch_size_per_image=_roi_attr(roi_heads, "batch_size_per_image", 128, ["box_batch_size_per_image"]),
        positive_fraction=_roi_attr(roi_heads, "positive_fraction", 0.25, ["box_positive_fraction"]),
        bbox_reg_weights=getattr(roi_heads.box_coder, "weights", None),
        score_thresh=_roi_attr(roi_heads, "score_thresh", 0.05, ["box_score_thresh"]),
        nms_thresh=_roi_attr(roi_heads, "nms_thresh", 0.3, ["box_nms_thresh"]),
        detections_per_img=_roi_attr(roi_heads, "detections_per_img", 100, ["box_detections_per_img"]),
    )
    model.__class__ = OldKDFasterRCNN
    return model


def train_one_epoch(model, optimizer, loader, device, epoch, kd_loss_fn, kd_weight):
    model.train()
    total_main = 0.0
    total_kd = 0.0
    loop = tqdm(loader, desc=f"Epoch {epoch} train old-kd")
    for images, targets in loop:
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)
        loss_dict, batched_images, transformed_targets, image_shapes, refined_boxes = model(images, targets)
        main_loss = sum(loss_dict.values())
        kd_loss = kd_weight * kd_loss_fn(batched_images, refined_boxes, transformed_targets, image_shapes)
        total_loss = main_loss + kd_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        total_main += float(main_loss.item())
        total_kd += float(kd_loss.item())
        loop.set_postfix(main=f"{main_loss.item():.4f}", kd=f"{kd_loss.item():.4f}")

    denom = max(1, len(loader))
    return {
        "train_main_loss": total_main / denom,
        "train_kd_loss": total_kd / denom,
        "train_total_loss": (total_main + total_kd) / denom,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Old DINOv2 KD baseline ported from dino-v2-3-real-data.ipynb.")
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--train-ann", default="data/audit/coco/train.json")
    parser.add_argument("--val-ann", default="data/audit/coco/val.json")
    parser.add_argument("--test-ann", default="data/audit/coco/test_standard.json")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-size", type=int, default=600)
    parser.add_argument("--max-size", type=int, default=600)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--beta1", type=float, default=1.0)
    parser.add_argument("--beta2", type=float, default=0.1)
    parser.add_argument("--kd-weight", type=float, default=2.0)
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--run-name", default="baseline_old_dinov2_l1_irm_notebook")
    parser.add_argument("--output-dir", default="results/minimal_baselines/old_dinov2_kd")
    parser.add_argument("--export-preds", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = create_run_dir(args.output_dir, args.run_name)

    train_set, val_set = build_datasets(args.train_ann, args.val_ann, args.image_root)
    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_old_kd_model(args.num_classes, not args.no_pretrained, args.min_size, args.max_size).to(device)
    kd_loss_fn = NotebookDinoV2L1IRMLoss(
        beta1=args.beta1,
        beta2=args.beta2,
        iou_threshold=args.iou_threshold,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    log_json = run_dir / "training_log.json"
    log_csv = run_dir / "training_log.csv"
    best_ckpt = run_dir / "best.pth"
    records = []
    best_map = -1.0
    for epoch in range(1, args.epochs + 1):
        train_row = train_one_epoch(model, optimizer, train_loader, device, epoch, kd_loss_fn, args.kd_weight)
        val_metrics = evaluate_map(model, val_loader, device, desc=f"Epoch {epoch} val")
        map_score = metric_value(val_metrics, "map")
        saved = False
        if map_score > best_map:
            best_map = map_score
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, best_ckpt)
            saved = True

        row = {
            "epoch": epoch,
            **train_row,
            **{f"val_{k}": json_safe(v) for k, v in val_metrics.items()},
            "saved_checkpoint": saved,
            "checkpoint": str(best_ckpt),
        }
        records.append(row)
        append_json_log(log_json, records)
        append_csv_row(log_csv, row)
        print(f"Epoch {epoch}: val mAP={map_score:.4f} best={best_map:.4f}")

    if args.export_preds:
        model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
        val_pred_path = export_coco_predictions(model, val_loader, device, run_dir / "val_predictions.json")
        test_set = type(val_set)(root=args.image_root, ann_file=args.test_ann)
        test_loader = make_loader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_pred_path = export_coco_predictions(model, test_loader, device, run_dir / "test_predictions.json")
        print(f"Prediction files: {val_pred_path}, {test_pred_path}")

    print(f"Run dir: {run_dir}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
