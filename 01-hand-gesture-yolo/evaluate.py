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
    return parser.parse_args()


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

    results = {
        "mAP50": round(metrics.box.map50, 4),
        "mAP50-95": round(metrics.box.map, 4),
        "mAP75": round(metrics.box.map75, 4),
        "precision": round(metrics.box.mp, 4),
        "recall": round(metrics.box.mr, 4),
        "mean_iou": round(metrics.box.mean_results()[0], 4)
        if hasattr(metrics.box, "mean_results")
        else None,
    }

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
