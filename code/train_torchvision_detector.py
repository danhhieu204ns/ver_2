import argparse
import json
import multiprocessing as mp
from pathlib import Path

import torch

from detection_common import (
    append_csv_row,
    append_json_log,
    build_optimizer,
    choose_threshold_for_target_fpr,
    collect_coco_predictions,
    create_run_dir,
    evaluate_map_and_predictions,
    json_safe,
    load_coco_dict,
    make_loader,
    metric_value,
    set_random_seed,
    summarize_image_level,
    train_detector_one_epoch,
    AuditCocoDetection,
    get_torchvision_detector,
)


SUPPORTED_ARCHITECTURES = (
    "faster_rcnn_r50_fpn",
    "faster_rcnn_r101_fpn",
    "retinanet_r50_fpn",
    "fcos_r50_fpn",
)


def build_scheduler(args, optimizer):
    if args.lr_schedule == "none":
        return None
    if args.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    if args.lr_schedule == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    raise ValueError(f"Unsupported lr schedule: {args.lr_schedule}")


def save_checkpoint(path, model, optimizer, scheduler, args, epoch, val_metrics, val_image_summary):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "epoch": epoch,
        "val_metrics": {k: json_safe(v) for k, v in val_metrics.items()},
        "val_image_summary": val_image_summary,
    }
    torch.save(payload, path)


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


def write_prediction_file(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return path


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def export_best_predictions(model, device, args, run_dir, checkpoint_path):
    state = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(state["model"])

    split_specs = [
        ("val", Path(args.val_ann)),
        ("test_standard", Path(args.test_ann)),
        ("test_robustness", Path(args.test_robustness_ann)),
    ]
    predictions = {}
    cocos = {}
    for split, ann_path in split_specs:
        dataset = AuditCocoDetection(root=args.image_root, ann_file=ann_path)
        loader = make_loader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)
        records = collect_coco_predictions(model, loader, device, desc=f"predict {split}")
        pred_path = write_prediction_file(run_dir / f"{split}_predictions.json", records)
        predictions[split] = {"path": str(pred_path), "records": records}
        cocos[split] = load_coco_dict(ann_path)

    threshold = choose_threshold_for_target_fpr(
        cocos["val"],
        predictions["val"]["records"],
        args.target_val_fpr,
    )
    image_metrics = {
        "threshold_source": "validation",
        "target_val_fpr": args.target_val_fpr,
        "selected_threshold": threshold,
        "checkpoint": str(checkpoint_path),
        "splits": {},
    }
    for split in predictions:
        image_metrics["splits"][split] = summarize_image_level(
            cocos[split],
            predictions[split]["records"],
            threshold,
        )
    metrics_path = run_dir / "image_level_metrics.json"
    metrics_path.write_text(json.dumps(image_metrics, indent=2), encoding="utf-8")
    return {"prediction_files": {k: v["path"] for k, v in predictions.items()}, "image_metrics": str(metrics_path)}


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2 TorchVision detector baseline trainer.")
    parser.add_argument("--architecture", choices=SUPPORTED_ARCHITECTURES, required=True)
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--train-ann", default="data/audit/coco/train.json")
    parser.add_argument("--val-ann", default="data/audit/coco/val.json")
    parser.add_argument("--test-ann", default="data/audit/coco/test_standard.json")
    parser.add_argument("--test-robustness-ann", default="data/audit/coco/test_robustness.json")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=("none", "cosine", "step"), default="cosine")
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--min-size", type=int, default=960)
    parser.add_argument("--max-size", type=int, default=960)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--target-val-fpr", type=float, default=0.01)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="results/stage2_baselines/torchvision")
    parser.add_argument("--export-preds", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"baseline_{args.architecture}_seed{args.seed}"
    run_dir = create_run_dir(args.output_dir, run_name)

    train_set = AuditCocoDetection(root=args.image_root, ann_file=args.train_ann)
    val_set = AuditCocoDetection(root=args.image_root, ann_file=args.val_ann)
    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, seed=args.seed)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)
    val_coco = load_coco_dict(args.val_ann)

    model = get_torchvision_detector(
        architecture=args.architecture,
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
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
        train_row = train_detector_one_epoch(model, optimizer, train_loader, device, epoch)
        val_metrics, val_predictions = evaluate_map_and_predictions(
            model,
            val_loader,
            device,
            desc=f"Epoch {epoch} val",
            keep_predictions=True,
        )
        threshold = choose_threshold_for_target_fpr(val_coco, val_predictions, args.target_val_fpr)
        val_image_summary = summarize_image_level(val_coco, val_predictions, threshold)
        val_image_overall = val_image_summary["overall"]
        map_score = metric_value(val_metrics, "map")
        recall_at_fpr = val_image_overall["recall_image"]

        saved_best_map = False
        saved_best_recall = False
        if map_score > best_map:
            best_map = map_score
            save_checkpoint(best_map_ckpt, model, optimizer, scheduler, args, epoch, val_metrics, val_image_summary)
            saved_best_map = True
        if recall_at_fpr > best_recall_at_fpr or (
            recall_at_fpr == best_recall_at_fpr and map_score > best_recall_tiebreak_map
        ):
            best_recall_at_fpr = recall_at_fpr
            best_recall_tiebreak_map = map_score
            save_checkpoint(best_recall_ckpt, model, optimizer, scheduler, args, epoch, val_metrics, val_image_summary)
            saved_best_recall = True

        row = {
            "epoch": epoch,
            "architecture": args.architecture,
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
        "seed": args.seed,
        "best_map": best_map,
        "best_recall_at_fpr": best_recall_at_fpr,
        "best_map_checkpoint": str(best_map_ckpt),
        "best_recall_at_fpr_checkpoint": str(best_recall_ckpt),
        "protocol": vars(args),
    }
    if args.export_preds and best_map_ckpt.is_file():
        summary["best_map_exports"] = export_best_predictions(model, device, args, run_dir, best_map_ckpt)
    elif args.export_preds:
        summary["best_map_exports"] = "skipped_no_checkpoint"
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Run dir: {run_dir}")
    print(f"Best mAP checkpoint: {best_map_ckpt}")
    print(f"Best recall-at-FPR checkpoint: {best_recall_ckpt}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
