#!/usr/bin/env python3
"""
Real-time hand-gesture detection from a webcam or video file.

Loads a trained YOLO model and runs inference on each frame,
drawing bounding boxes, class labels, confidence scores, and FPS.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time hand-gesture inference."
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
    print("Hand Gesture Inference")
    print("=" * 60)
    print(f"Weights: {args.weights}")
    print(f"Source:  {args.source}")
    print(f"Image:   {args.imgsz}px")
    print(f"Conf:    {args.conf}")
    print(f"IoU:     {args.iou}")
    print(f"Device:  {args.device}")
    print("Press 'q' to quit.")
    print()

    if not args.weights.exists():
        print(f"Error: weights not found: {args.weights}")
        print("Train or export a model first.")
        return 1

    model = YOLO(str(args.weights))

    # Convert numeric source to int for webcam index
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source: {args.source}")
        return 1

    # Exponential moving average for smooth FPS display
    fps_smooth = 0.0
    alpha = 0.1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        t1 = time.perf_counter()

        # Smooth FPS
        instant_fps = 1.0 / (t1 - t0)
        fps_smooth = alpha * instant_fps + (1 - alpha) * fps_smooth

        annotated = results[0].plot()
        cv2.putText(
            annotated,
            f"FPS: {fps_smooth:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Hand Gesture Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
