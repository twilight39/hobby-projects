#!/usr/bin/env python3
"""
Fine-tune YOLOv8n on the HaGRID hand-gesture dataset.

Uses the prepared config.yaml produced by data_setup.py.
Defaults are tuned for CPU training on a MacBook:
- model: yolov8n (nano, smallest / fastest)
- imgsz: 320 (faster CPU inference than 640)
- device: cpu
- epochs: 50 with early stopping patience of 10
"""

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss


# ---------------------------------------------------------------------------
# Work around an MPS bug in Ultralytics' detection-loss preprocessing.
# `unique(return_counts=True)` and `scatter_add_` on MPS can produce
# negative / placeholder errors. We run the small preprocessing step on CPU
# and move the result back to MPS.
# ---------------------------------------------------------------------------
_ORIG_PREPROCESS = v8DetectionLoss.preprocess


def _mps_safe_preprocess(
    self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor
) -> torch.Tensor:
    if self.device.type == "mps":
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)

        batch_idx = targets[:, 0].long().cpu()
        _, counts = batch_idx.unique(return_counts=True)
        max_count = int(counts.max().item())

        # Build the output entirely on CPU
        out = torch.zeros(batch_size, max_count, ne - 1, device="cpu")
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device="cpu")
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(nl, device="cpu") - offsets[batch_idx]
        out[batch_idx, within_idx] = targets[:, 1:].cpu()

        from ultralytics.utils.ops import xywh2xyxy

        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor.cpu()))
        return out.to(self.device)

    return _ORIG_PREPROCESS(self, targets, batch_size, scale_tensor)


v8DetectionLoss.preprocess = _mps_safe_preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n on hand gestures."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YOLO dataset config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base YOLO model to fine-tune.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Input image size (320 or 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu, mps (Apple Silicon), or cuda.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Directory for training outputs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Name of this training run.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to last.pt to resume training from.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("YOLOv8 Hand-Gesture Training")
    print("=" * 60)
    print(f"Model:   {args.model}")
    print(f"Data:    {args.data.resolve()}")
    print(f"Epochs:  {args.epochs}")
    print(f"Image:   {args.imgsz}px")
    print(f"Batch:   {args.batch}")
    print(f"Device:  {args.device}")
    print(f"Seed:    {args.seed}")
    print()

    # MPS validation can crash when a batch contains images with zero targets
    # (e.g. our no_gesture background images). The monkey-patched loss
    # preprocessor above avoids this, so we keep validation on MPS for speed.
    # Final test evaluation is still run on CPU for maximum compatibility.
    val_device = "cpu" if args.device == "mps" else args.device

    # Keep all Ultralytics outputs inside this project directory. Passing an
    # absolute project path prevents the global Ultralytics runs_dir setting
    # from placing artifacts outside the repo.
    project_path = Path(args.project).resolve()

    if not args.data.exists():
        print(f"Error: config file not found: {args.data}")
        print("Run data_setup.py first to generate config.yaml.")
        return 1

    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        # Load pretrained YOLOv8n model. Ultralytics downloads weights automatically.
        model = YOLO(args.model)

        # Train. Ultralytics handles optimizer, scheduler, augmentation, logging, etc.
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            seed=args.seed,
            project=str(project_path),
            name=args.name,
        )

    # Validate on the test split explicitly and print metrics.
    print("\nRunning final validation on test split...")
    metrics = model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        device=val_device,
        project=str(project_path),
        name="test",
    )
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
