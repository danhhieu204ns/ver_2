import argparse
import multiprocessing as mp
from pathlib import Path

import torch

from detection_common import (
    append_csv_row,
    append_json_log,
    build_datasets,
    build_optimizer,
    create_run_dir,
    evaluate_map,
    export_coco_predictions,
    get_faster_rcnn_r50_fpn,
    json_safe,
    make_loader,
    metric_value,
    train_detector_one_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Faster R-CNN R50-FPN baseline for audit COCO splits.")
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
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--run-name", default="baseline_faster_rcnn_r50_fpn")
    parser.add_argument("--output-dir", default="results/minimal_baselines/faster_rcnn")
    parser.add_argument("--export-preds", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = create_run_dir(args.output_dir, args.run_name)

    train_set, val_set = build_datasets(args.train_ann, args.val_ann, args.image_root)
    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_faster_rcnn_r50_fpn(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    log_json = run_dir / "training_log.json"
    log_csv = run_dir / "training_log.csv"
    best_ckpt = run_dir / "best.pth"
    records = []
    best_map = -1.0

    for epoch in range(1, args.epochs + 1):
        train_row = train_detector_one_epoch(model, optimizer, train_loader, device, epoch)
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
