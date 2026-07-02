"""
Evaluate a trained YOLO model on the test split.

Loads weights from a training run and computes mAP50, mAP50-95,
precision, recall, and IoU. Useful for re-running evaluation without
retraining.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLO model on the test split."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/train/weights/best.pt"),
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YOLO dataset config file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Input image size used during training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/detect"),
        help="Directory for Ultralytics validation outputs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="val",
        help="Name of the validation run.",
    )
    parser.add_argument(
        "--mean-iou",
        action="store_true",
        help="Compute a standalone mean IoU over matched predictions/labels.",
    )
    return parser.parse_args()


def _bbox_iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of xyxy boxes."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_xmin = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_ymin = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_xmax = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_ymax = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    inter = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union


def compute_mean_iou(
    model: YOLO,
    data_config: Path,
    split: str,
    imgsz: int,
    device: str,
    conf: float = 0.25,
    iou_thresh: float = 0.5,
) -> float:
    """
    Compute mean IoU of true-positive detections on the chosen split.

    Each prediction is greedily matched to the highest-IoU ground-truth box of
    the same class. The returned value is the average IoU of all matches above
    `iou_thresh`.
    """
    cfg = yaml.safe_load(data_config.read_text(encoding="utf-8"))
    base_path = Path(cfg["path"]).expanduser().resolve()
    image_dir = base_path / cfg[split]
    label_dir = base_path / cfg[split].replace("images", "labels")

    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    ious: list[float] = []
    for img_path in image_paths:
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        # Ground-truth boxes: YOLO normalized xywh -> xyxy (absolute)
        gt_lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        gt_boxes = []
        gt_classes = []
        for line in gt_lines:
            if not line.strip():
                continue
            cls, cx, cy, w, h = map(float, line.split())
            gt_classes.append(int(cls))
            gt_boxes.append([cx, cy, w, h])

        if not gt_boxes:
            continue

        result = model.predict(
            str(img_path),
            imgsz=imgsz,
            device=device,
            conf=conf,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            continue

        h, w = result.orig_shape[:2]
        gt = torch.tensor(gt_boxes, dtype=torch.float32)
        gt[:, [0, 2]] *= w
        gt[:, [1, 3]] *= h
        gt_xyxy = torch.stack(
            [
                gt[:, 0] - gt[:, 2] / 2,
                gt[:, 1] - gt[:, 3] / 2,
                gt[:, 0] + gt[:, 2] / 2,
                gt[:, 1] + gt[:, 3] / 2,
            ],
            dim=1,
        )
        gt_classes_t = torch.tensor(gt_classes, dtype=torch.long)

        pred_xyxy = result.boxes.xyxy.cpu()
        pred_classes = result.boxes.cls.cpu().long()
        pred_conf = result.boxes.conf.cpu()

        # Greedily match predictions to ground truth of the same class.
        order = torch.argsort(pred_conf, descending=True)
        pred_xyxy = pred_xyxy[order]
        pred_classes = pred_classes[order]
        matched_gt = torch.zeros(len(gt_xyxy), dtype=torch.bool)

        for i in range(len(pred_xyxy)):
            same_class = gt_classes_t == pred_classes[i]
            if not same_class.any():
                continue
            candidate_ious = _bbox_iou_matrix(
                pred_xyxy[i : i + 1], gt_xyxy[same_class]
            )[0]
            best_iou, best_idx = candidate_ious.max(dim=0)
            original_idx = torch.nonzero(same_class, as_tuple=False)[best_idx].item()
            if best_iou >= iou_thresh and not matched_gt[original_idx]:
                matched_gt[original_idx] = True
                ious.append(best_iou.item())

    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("YOLO Model Evaluation")
    print("=" * 60)
    print(f"Weights: {args.weights}")
    print(f"Data:    {args.data}")
    print(f"Split:   {args.split}")
    print(f"Image:   {args.imgsz}px")
    print(f"Device:  {args.device}")
    print()

    if not args.weights.exists():
        print(f"Error: weights not found: {args.weights}")
        print("Train a model first with train.py.")
        return 1

    if not args.data.exists():
        print(f"Error: config file not found: {args.data}")
        return 1

    model = YOLO(str(args.weights))

    # Use an absolute project path so outputs stay inside this repo.
    project_path = Path(args.project).resolve()

    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        project=str(project_path),
        name=args.name,
    )

    # Note: Ultralytics' mean_results() returns [mp, mr, map50, map].
    # Index [0] is mean precision, not IoU. We compute a real mean IoU below.
    results = {
        "mAP50": round(metrics.box.map50, 4),
        "mAP50-95": round(metrics.box.map, 4),
        "mAP75": round(metrics.box.map75, 4),
        "precision": round(metrics.box.mp, 4),
        "recall": round(metrics.box.mr, 4),
    }

    if args.mean_iou:
        print("\nComputing mean IoU over matched predictions...")
        mean_iou = compute_mean_iou(
            model, args.data, args.split, args.imgsz, args.device
        )
        results["mean_iou"] = round(mean_iou, 4)

    print("\nMetrics:")
    for key, value in results.items():
        print(f"  {key:12s}: {value}")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics to {args.save_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
