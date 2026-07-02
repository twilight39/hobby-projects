#!/usr/bin/env python3
"""
Export a trained YOLO model for edge inference.

Supports ONNX (cross-platform) and CoreML (Apple Silicon / iOS).
Also benchmarks inference latency on the chosen device.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export trained YOLO model to ONNX or CoreML."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/train/weights/best.pt"),
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "coreml"],
        help="Export format.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Input image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Benchmark device: cpu or mps (Mac GPU).",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=100,
        help="Number of inference runs for benchmarking.",
    )
    return parser.parse_args()


def benchmark(model: YOLO, imgsz: int, device: str, runs: int = 100) -> dict:
    """Run a simple latency benchmark on a dummy input and return metadata."""
    print(f"\nBenchmarking {runs} inference runs at {imgsz}px on '{device}'...")

    # Create a dummy RGB image
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warm-up
    for _ in range(10):
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)

    # Timed runs
    start = time.perf_counter()
    for _ in range(runs):
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / runs) * 1000
    fps = runs / elapsed

    print(f"Average latency: {avg_ms:.2f} ms")
    print(f"FPS:             {fps:.2f}")

    return {
        "avg_latency_ms": round(avg_ms, 4),
        "fps": round(fps, 4),
        "runs": runs,
    }


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("YOLO Model Export")
    print("=" * 60)
    print(f"Weights: {args.weights}")
    print(f"Format:  {args.format}")
    print(f"Image:   {args.imgsz}px")
    print()

    if not args.weights.exists():
        print(f"Error: weights not found: {args.weights}")
        print("Train a model first with train.py.")
        return 1

    model = YOLO(str(args.weights))

    # Export
    print(f"Exporting to {args.format.upper()}...")
    export_path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=False,
        simplify=True,
    )
    print(f"Exported to: {export_path}")

    # Benchmark the exported model
    exported_model = YOLO(str(export_path))
    bench = benchmark(
        exported_model, args.imgsz, args.device, args.benchmark_runs
    )

    # Collect and save benchmark metadata
    export_file = Path(export_path)
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights": str(args.weights.resolve()),
        "export_format": args.format,
        "export_path": str(export_file.resolve()),
        "export_size_bytes": export_file.stat().st_size,
        "image_size": args.imgsz,
        "benchmark_device": args.device,
        **bench,
    }

    metadata_path = export_file.with_suffix(".benchmark.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved benchmark metadata to {metadata_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
