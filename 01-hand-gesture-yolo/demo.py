#!/usr/bin/env python3
"""
Record a demo video of live hand-gesture detection.

Runs inference on the webcam feed, draws predictions and FPS,
and writes the output to demos/gesture_demo.mp4 (or a custom path).
Useful for portfolio showcases and remote interviews.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a hand-gesture detection demo video."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/train/weights/best.pt"),
        help="Path to trained model weights or exported model.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Webcam index (e.g., '0') or path to a video file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demos/gesture_demo.mp4"),
        help="Path to save the recorded demo video.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Maximum recording duration in seconds.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Input image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for non-maximum suppression.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu, mps, or cuda.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("Recording Hand Gesture Demo")
    print("=" * 60)
    print(f"Weights:  {args.weights}")
    print(f"Source:   {args.source}")
    print(f"Output:   {args.output}")
    print(f"Duration: {args.duration}s")
    print(f"Device:   {args.device}")
    print("Press 'q' to stop early.")
    print()

    if not args.weights.exists():
        print(f"Error: weights not found: {args.weights}")
        print("Train or export a model first.")
        return 1

    model = YOLO(str(args.weights))

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source: {args.source}")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

        annotated = results[0].plot()

        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0.0
        cv2.putText(
            annotated,
            f"FPS: {current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        writer.write(annotated)
        frame_count += 1

        cv2.imshow("Recording Demo", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if elapsed >= args.duration:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\nSaved {frame_count} frames to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
