# Hand Gesture YOLO Detector

A YOLOv8-based hand gesture recognition project for real-time edge inference. Built as a portfolio piece for SWE/AI/ML interviews.

## What It Does

Detects and classifies hand gestures from a webcam feed in real time. The pipeline covers the full ML lifecycle: dataset preparation, model fine-tuning, evaluation, export to edge formats, and live inference.

## Demo

<video src="https://raw.githubusercontent.com/twilight39/hobby-projects/main/01-hand-gesture-yolo/demos/gesture_demo.mp4" controls width="640"></video>

If the video doesn't play above, [download it directly](https://github.com/twilight39/hobby-projects/raw/main/01-hand-gesture-yolo/demos/gesture_demo.mp4).

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
| Mean IoU | — | **0.889** |

Per-class test performance:

| Class | mAP50 | mAP50-95 |
|-------|-------|----------|
| open_palm | 0.995 | 0.895 |
| closed_fist | 0.995 | 0.795 |
| thumbs_up | 0.995 | 0.817 |
| peace_sign | 0.995 | 0.848 |
| no_gesture | 0.970 | 0.674 |

`no_gesture` is the weakest class, which is typical for a background/idle class: the hand is present but not in a discriminative pose, so it is visually closer to the other classes.

Detailed evaluation plots, example predictions, and the formal metrics file live under `results/`. See [`METRICS.md`](METRICS.md) for the full metrics report.

## How It Works

A short design overview for reviewers and interviewers.

### Why YOLOv8n?

We chose **YOLOv8n** (the nano variant) because this project targets real-time edge inference:

- **Small:** ~3 M parameters, 5.9 MB in PyTorch.
- **Fast:** >45 FPS on a laptop CPU at 320 px; >80 FPS with ONNX Runtime.
- **Good enough:** Hand gestures are visually simple objects with clear shapes. For five classes, YOLOv8n has plenty of capacity, and the metrics confirm it (mAP50 = 0.99).
- **Easy to deploy:** Ultralytics exports cleanly to ONNX and CoreML.

If we later needed higher mAP50-95 on the harder `no_gesture` class, the next step would be YOLOv8s or 640 px input — but only after confirming the latency budget allows it.

### Why These 5 Classes?

The classes map directly to common UI/control gestures:

| Gesture | Mapped action |
|---------|---------------|
| `open_palm` | Play / resume |
| `closed_fist` | Stop / pause |
| `thumbs_up` | Confirm / next |
| `peace_sign` | Select / option 2 |
| `no_gesture` | Idle / background |

Five classes keep the dataset small (5,000 images), the model fast, and the demo easy to understand. `no_gesture` is included as a first-class class so the system can explicitly report “idle” rather than hallucinating one of the other gestures when no hand is present.

### Data Preparation

We use the [HaGRID v2 512px](https://github.com/hukenovs/hagrid) dataset. `data_setup.py`:

1. Loads JSON annotations for the 5 target classes.
2. Samples 1,000 images per class deterministically (seed = 42).
3. **Splits by `user_id`** before creating train/val/test sets. This is the key anti-leakage step: the same person never appears in both training and testing.
4. Extracts only the sampled images from the 119 GB zip archive.
5. Converts HaGRID `[x_min, y_min, width, height]` boxes to YOLO normalized `[x_center, y_center, width, height]` format.
6. Writes `config.yaml`.

### Augmentation Strategy

We deliberately **do not use a custom augmentation pipeline**. Instead we rely on:

- **YOLOv8’s built-in augmentation:** mosaic, HSV jitter, flips, scale/translate, and random cropping during training.
- **HaGRID’s natural diversity:** tens of thousands of different people, lighting conditions, backgrounds, and camera angles.

For this dataset, hand-crafting augmentations adds complexity without clear benefit and risks distorting hand shapes. The built-in augmentations plus the dataset’s inherent variety are sufficient.

### Training

`train.py` fine-tunes the pretrained COCO weights with:

- `imgsz=320` for speed.
- `batch=16`.
- AdamW optimizer (auto-selected by Ultralytics).
- 50 epochs with early stopping patience of 10.

The model trains on Apple MPS. A small monkey-patch routes the detection-loss preprocessing through CPU for MPS batches that contain empty targets (from `no_gesture` images), avoiding a known MPS bug.

### Deployment Approach

The intended deployment path is:

1. **Export** the trained `.pt` model to ONNX (`export.py --format onnx`) or CoreML (`export.py --format coreml`).
2. **Benchmark** the exported model on the target hardware.
3. **Run inference** with a minimal runtime:
   - ONNX Runtime for cross-platform deployment (Linux, Windows, Raspberry Pi, Jetson).
   - CoreML on Apple Silicon / iOS for Neural Engine acceleration.
4. **Post-process** the raw outputs (decode boxes, NMS, map class IDs to actions) in the application layer.

For concrete edge targets, see the **Edge Deployment** section below.

## Quick Start

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies (uv is preferred; pip works too)
uv sync
# or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

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

Example benchmark on Apple M1 Pro (320 px input, 100 runs):

| Runtime | Device | Latency (ms) | FPS |
|---------|--------|--------------|-----|
| PyTorch | MPS | 11.84 | 84.5 |
| PyTorch | CPU | 22.00 | 45.5 |
| ONNX Runtime | CPU | 14.08 | 71.0 |

> **Note:** ONNX Runtime does not provide an MPS execution provider on macOS, so the exported ONNX model is benchmarked on CPU. Use the CoreML export for Apple GPU/Neural Engine inference.

```bash
# ONNX on CPU
uv run python export.py --format onnx --device cpu

# CoreML on Mac GPU
uv run python export.py --format coreml --device mps
```

## Edge Deployment

The exported model is sized for embedded and edge boards. Suggested deployment targets:

| Platform | Runtime | Notes |
|----------|---------|-------|
| Raspberry Pi 4/5 | ONNX Runtime (CPU) | Use the ONNX export and 320 px input. Expect ~30–50 ms/frame depending on thread count and quantization. |
| NVIDIA Jetson Nano / Orin | TensorRT via ONNX | Convert the ONNX file to a TensorRT engine for GPU/NPU acceleration. |
| Apple Silicon / iOS | CoreML | Export with `--format coreml`. The model can run on the Apple Neural Engine with sub-frame latency. |
| x86 / AMD64 edge PC | ONNX Runtime / OpenVINO | Use OpenVINO for Intel integrated graphics or CPU inference. |

Keep the input resolution at 320 px unless accuracy needs dominate latency needs. If you need higher mAP on `no_gesture`, retrain at 640 px or switch to `yolov8s`.

## Real-Time Inference

`infer.py` runs the model on a webcam or video file:

```bash
uv run python infer.py --source 0 --device mps
```

`demo.py` records an annotated demo video:

```bash
uv run python demo.py --duration 60 --device mps --output demos/gesture_demo.mp4
```

Use `--device mps` on Apple Silicon for much higher FPS; the default is `cpu`.

## Project Structure

```text
.
├── README.md              # This file (includes design rationale and deployment notes)
├── METRICS.md             # Full metrics report and benchmark details
├── requirements.txt       # pip fallback for non-uv users
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
├── results/               # Evaluation plots, metrics, and example predictions
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
