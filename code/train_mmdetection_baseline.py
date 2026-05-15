import argparse
from pathlib import Path


MODEL_CONFIGS = {
    "cascade_rcnn_r50_fpn": "mmdet::cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py",
}


def require_mmdet():
    try:
        from mmengine.config import Config
        from mmengine.hub import get_config
        from mmengine.runner import Runner
        from mmdet.utils import register_all_modules
    except ImportError as exc:
        raise SystemExit(
            "MMDetection backend is required for Cascade R-CNN. Install compatible "
            "mmdet/mmcv/mmengine, then rerun this command. Original error: " + str(exc)
        ) from exc
    return Config, get_config, Runner, register_all_modules


def load_config(config_path, Config, get_config):
    if "::" in str(config_path):
        return get_config(str(config_path))
    return Config.fromfile(config_path)


def set_num_classes(node, num_classes):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "num_classes":
                node[key] = num_classes
            else:
                set_num_classes(value, num_classes)
    elif isinstance(node, list):
        for item in node:
            set_num_classes(item, num_classes)


def configure_dataset(dataset_cfg, image_root, ann_file, test_mode=False):
    if "dataset" in dataset_cfg:
        configure_dataset(dataset_cfg["dataset"], image_root, ann_file, test_mode=test_mode)
        return
    dataset_cfg["type"] = dataset_cfg.get("type", "CocoDataset")
    dataset_cfg["data_root"] = ""
    dataset_cfg["ann_file"] = str(Path(ann_file).resolve())
    dataset_cfg["data_prefix"] = {"img": str(Path(image_root).resolve()) + "/"}
    dataset_cfg["metainfo"] = {"classes": ("nine_dash_line",)}
    dataset_cfg["test_mode"] = test_mode
    if not test_mode:
        dataset_cfg["filter_cfg"] = {"filter_empty_gt": False, "min_size": 0}


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2 MMDetection baseline trainer.")
    parser.add_argument("--architecture", choices=tuple(MODEL_CONFIGS), required=True)
    parser.add_argument("--config", default=None, help="Override MMDetection config path or mmdet:: URI.")
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--train-ann", default="data/audit/coco/train.json")
    parser.add_argument("--val-ann", default="data/audit/coco/val.json")
    parser.add_argument("--test-ann", default="data/audit/coco/test_standard.json")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="results/stage2_baselines/mmdetection")
    return parser.parse_args()


def main():
    args = parse_args()
    Config, get_config, Runner, register_all_modules = require_mmdet()
    register_all_modules(init_default_scope=True)

    config_path = args.config or MODEL_CONFIGS[args.architecture]
    cfg = load_config(config_path, Config, get_config)
    run_name = args.run_name or f"baseline_{args.architecture}_seed{args.seed}"
    cfg.work_dir = str(Path(args.output_dir) / run_name)

    set_num_classes(cfg.model, args.num_classes)
    configure_dataset(cfg.train_dataloader.dataset, args.image_root, args.train_ann, test_mode=False)
    configure_dataset(cfg.val_dataloader.dataset, args.image_root, args.val_ann, test_mode=True)
    configure_dataset(cfg.test_dataloader.dataset, args.image_root, args.test_ann, test_mode=True)
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.train_dataloader.num_workers = args.num_workers
    cfg.val_dataloader.batch_size = 1
    cfg.val_dataloader.num_workers = args.num_workers
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = args.num_workers

    if hasattr(cfg, "val_evaluator"):
        cfg.val_evaluator.ann_file = str(Path(args.val_ann).resolve())
    if hasattr(cfg, "test_evaluator"):
        cfg.test_evaluator.ann_file = str(Path(args.test_ann).resolve())
    if hasattr(cfg, "train_cfg") and hasattr(cfg.train_cfg, "max_epochs"):
        cfg.train_cfg.max_epochs = args.epochs
    if hasattr(cfg, "optim_wrapper"):
        cfg.optim_wrapper.optimizer.lr = args.lr
        cfg.optim_wrapper.optimizer.weight_decay = args.weight_decay
    cfg.randomness = {"seed": args.seed, "deterministic": False}
    if hasattr(cfg, "default_hooks") and "checkpoint" in cfg.default_hooks:
        cfg.default_hooks.checkpoint.save_best = "coco/bbox_mAP"
        cfg.default_hooks.checkpoint.rule = "greater"

    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    cfg.dump(str(Path(cfg.work_dir) / "resolved_config.py"))
    runner = Runner.from_cfg(cfg)
    runner.train()
    print(f"Run dir: {cfg.work_dir}")


if __name__ == "__main__":
    main()
