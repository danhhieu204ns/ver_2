import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def yolo_box_to_coco(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def export_predictions(weights, coco_ann, image_root, output, imgsz, conf, iou, batch, device):
    coco = load_coco(coco_ann)
    images = coco.get("images", [])
    paths = []
    for image in images:
        path = Path(image_root) / image["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing image referenced by COCO: {path}")
        paths.append(str(path.resolve()))

    model = YOLO(str(weights))
    predict_kwargs = {
        "source": paths,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "batch": batch,
        "save": False,
        "verbose": False,
        "stream": False,
    }
    if device:
        predict_kwargs["device"] = device
    results = model.predict(**predict_kwargs)

    records = []
    for image, result in zip(images, results):
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.detach().cpu().tolist()
        scores = result.boxes.conf.detach().cpu().tolist()
        classes = result.boxes.cls.detach().cpu().tolist()
        for box, score, cls in zip(boxes, scores, classes):
            records.append(
                {
                    "image_id": int(image["id"]),
                    "category_id": int(cls) + 1,
                    "bbox": yolo_box_to_coco(box),
                    "score": float(score),
                }
            )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return output, len(images), len(records)


def parse_args():
    parser = argparse.ArgumentParser(description="Export Ultralytics YOLO/RT-DETR predictions as COCO detections.")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--coco-ann", required=True, type=Path)
    parser.add_argument("--image-root", default="data", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output, image_count, pred_count = export_predictions(
        weights=args.weights,
        coco_ann=args.coco_ann,
        image_root=args.image_root,
        output=args.output,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        device=args.device,
    )
    print(f"Wrote {output}: {pred_count} detections for {image_count} images")


if __name__ == "__main__":
    main()
