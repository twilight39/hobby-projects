# Hand Gesture YOLO Detector

A YOLOv8-based hand gesture recognition project for real-time edge inference. Built as a portfolio piece for SWE/AI/ML interviews.

## What It Does

Detects and classifies hand gestures from a webcam feed in real time. The pipeline covers the full ML lifecycle: dataset preparation, model fine-tuning, evaluation, export to edge formats, and live inference.

## Gesture Classes

| Class ID | Gesture | Use Case |
|----------|---------|----------|
| 0 | `open_palm` | Play / resume |
| 1 | `closed_fist` | Stop / pause |
| 2 | `thumbs_up` | Confirm / next |
| 3 | `peace_sign` | Select / option 2 |
| 4 | `no_gesture` | Background / idle |

## Results

Fine-tuned `yolov8n.pt` on a 5,000-image HaGRID 512px subset (1,000 images per class) at `imgsz=320`.

| Metric | Target | Actual |
|--------|--------|--------|
| mAP50 | ≥ 0.85 | **0.990** |
| mAP50-95 | ≥ 0.60 | **0.806** |
| mAP75 | — | **0.931** |
| Precision | — | **0.984** |
| Recall | — | **0.976** |

Per-class test performance:

| Class | mAP50 | mAP50-95 |
|-------|-------|----------|
| open_palm | 0.995 | 0.895 |
| closed_fist | 0.995 | 0.795 |
| thumbs_up | 0.995 | 0.817 |
| peace_sign | 0.995 | 0.848 |
| no_gesture | 0.970 | 0.674 |

`no_gesture` is the weakest class, which is typical for a background/idle class: the hand is present but not in a discriminative pose, so it is visually closer to the other classes.

## Quick Start

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync

# Download the HaGRID v2 512px image archive (~128 GB) and annotations (~720 MB)
# from https://github.com/hukenovs/hagrid and place them under:
#   data/hagrid/raw/hagridv2_512.zip
#   data/hagrid/raw/annotations/

# Prepare the dataset (generates config.yaml)
uv run python data_setup.py

# Fine-tune YOLOv8n
uv run python train.py --device mps

# Evaluate on the test split
uv run python evaluate.py

# Export to ONNX (or CoreML for Apple Silicon)
uv run python export.py --format onnx
uv run python export.py --format coreml --device mps

# Run real-time inference
uv run python infer.py --source 0

# Record a demo video
uv run python demo.py --duration 60
```

> **Apple Silicon (MPS):** `train.py` includes a small patch that works around an MPS bug in Ultralytics' loss preprocessing when a batch contains `no_gesture` images with empty targets. Training runs on MPS; final test validation runs on CPU for compatibility.

## Dataset

We use the [HaGRID v2 512px](https://github.com/hukenovs/hagrid) dataset, which contains ~1M images of 33 gestures plus a `no_gesture` class. `data_setup.py`:

- Loads annotations for the 5 target classes.
- Deterministically samples 1,000 images per class with a fixed random seed.
- Splits sampled images into train / val / test by `user_id` to prevent data leakage.
- Extracts only the sampled images from the zip archive.
- Converts HaGRID JSON bounding boxes to YOLO `.txt` format.
- Writes `config.yaml` for Ultralytics.

### Why sample by user?

HaGRID provides a `user_id` per image. Splitting by person ensures the same subject does not appear in both training and testing, giving a more realistic estimate of real-world performance.

## Training

`train.py` fine-tunes a pretrained `yolov8n.pt` model on the prepared dataset.

- **Model:** `yolov8n.pt` (nano, smallest variant)
- **Image size:** 320 px (faster edge inference)
- **Epochs:** 50 with early stopping patience of 10
- **Batch size:** 16
- **Device:** `cpu` by default; pass `--device mps` on Apple Silicon

### Example overrides

```bash
# Train at higher resolution for better accuracy
uv run python train.py --imgsz 640

# Use Apple Silicon GPU
uv run python train.py --device mps
```

## Evaluation

`evaluate.py` loads the trained weights and computes metrics on the test split:

```bash
uv run python evaluate.py --save-json metrics.json
```

Output:

- `mAP50`
- `mAP50-95`
- `mAP75`
- `Precision`
- `Recall`

## Export & Benchmark

`export.py` converts the trained model to an edge format and benchmarks latency/FPS.

Supported formats:

- **ONNX** — cross-platform, runs with ONNX Runtime.
- **CoreML** — optimized for Apple Silicon / iOS, can use the Apple Neural Engine.

Each export produces a `.benchmark.json` file with latency, FPS, file size, and device info.

```bash
# ONNX on CPU
uv run python export.py --format onnx --device cpu

# CoreML on Mac GPU
uv run python export.py --format coreml --device mps
```

## Real-Time Inference

`infer.py` runs the model on a webcam or video file:

```bash
uv run python infer.py --source 0 --device cpu
```

`demo.py` records an annotated demo video:

```bash
uv run python demo.py --duration 60 --output demos/gesture_demo.mp4
```

## Project Structure

```text
.
├── README.md              # This file
├── pyproject.toml         # uv dependencies and tool config
├── uv.lock                # Locked dependency versions
├── .python-version        # Python version for uv
├── data_setup.py          # Prepare HaGRID subset for YOLO
├── train.py               # Fine-tune YOLOv8n
├── evaluate.py            # Evaluate on test split
├── export.py              # Export to ONNX / CoreML + benchmark
├── infer.py               # Real-time webcam inference
├── demo.py                # Record demo video
├── .gitignore             # Ignore data, runs, models, etc.
├── data/                  # Dataset directory (gitignored)
│   └── hagrid/
│       ├── raw/           # Downloaded zip + annotations
│       └── processed/     # Sampled images and labels
├── runs/                  # Training outputs (gitignored)
├── demos/                 # Demo videos (gitignored)
└── models/                # Exported models (gitignored)
```

## Tools

- **Python package manager:** [uv](https://docs.astral.sh/uv/)
- **Linter / formatter:** [ruff](https://docs.astral.sh/ruff/)
- **Type checker:** [basedpyright](https://docs.basedpyright.com/)
- **Framework:** [Ultralytics YOLOv8](https://docs.ultralytics.com/)

## License

MIT
