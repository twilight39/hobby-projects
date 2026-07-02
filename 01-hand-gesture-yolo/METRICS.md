# Metrics & Benchmarks

All numbers are from the trained `yolov8n` model on the HaGRID 512px 5-class subset at `imgsz=320`.

## Test-Split Object Detection Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **mAP50** | **0.990** | Mean Average Precision at IoU ≥ 0.50 |
| **mAP50-95** | **0.806** | Mean AP averaged over IoU thresholds 0.50, 0.55, …, 0.95 |
| **mAP75** | **0.931** | Mean AP at IoU ≥ 0.75 |
| **Precision** | **0.984** | Mean precision across classes |
| **Recall** | **0.976** | Mean recall across classes |
| **Mean IoU** | **0.889** | Average IoU of true-positive detections (matched at IoU ≥ 0.50) |
| **IoU threshold (NMS)** | **0.45** | Default non-maximum suppression threshold |
| **Confidence threshold** | **0.50** | Default detection confidence threshold |

### Per-class breakdown

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| open_palm | 99 | 99 | 0.968 | 1.000 | 0.995 | 0.895 |
| closed_fist | 110 | 110 | 1.000 | 0.971 | 0.995 | 0.795 |
| thumbs_up | 99 | 99 | 0.987 | 0.990 | 0.995 | 0.817 |
| peace_sign | 107 | 107 | 1.000 | 0.995 | 0.995 | 0.848 |
| no_gesture | 203 | 235 | 0.964 | 0.922 | 0.970 | 0.674 |
| **all** | **521** | **650** | **0.984** | **0.976** | **0.990** | **0.806** |

`no_gesture` is the weakest class, which is expected for a background/idle class where the hand is present but not in a clear pose.

## Training Details

| Setting | Value |
|---------|-------|
| Base model | `yolov8n.pt` (pretrained on COCO) |
| Image size | 320 px |
| Epochs | 50 (early stopping patience 10) |
| Batch size | 16 |
| Optimizer | AdamW (auto-selected by Ultralytics) |
| Training device | Apple MPS (GPU) |
| Validation device | CPU (MPS-safe fallback) |
| Approx. training time | ~61 minutes |

## Hardware

- **Machine:** MacBook Pro (18,1)
- **Chip:** Apple M1 Pro (10 cores: 8 performance + 2 efficiency)
- **Memory:** 16 GB
- **OS:** macOS Sonoma 14.5
- **Python:** 3.13.5 (project virtualenv)

## Model Sizes

| Format | File | Size |
|--------|------|------|
| PyTorch | `runs/detect/train/weights/best.pt` | 5.9 MB |
| ONNX | `runs/detect/train/weights/best.onnx` | 11.6 MB |

## Inference Benchmarks (single image)

Measured on an Apple M1 Pro with a dummy 320×320 RGB input and 100 timed runs after a 10-run warm-up.

| Runtime | Device | Avg. Latency | FPS | Notes |
|---------|--------|--------------|-----|-------|
| PyTorch (`best.pt`) | MPS | 11.8 ms | 84.5 | Fastest option on Apple Silicon |
| PyTorch (`best.pt`) | CPU | 22.0 ms | 45.5 | Native Ultralytics pipeline |
| ONNX Runtime (`best.onnx`) | CPU | 14.1 ms | 71.0 | ONNX Runtime has no MPS provider on macOS |

On this machine PyTorch on MPS is the fastest configuration. ONNX Runtime on CPU is close and is the best cross-platform option when MPS/Metal is unavailable.

## Dataset

- **Source:** HaGRID v2 512px
- **Classes:** 5 (open_palm, closed_fist, thumbs_up, peace_sign, no_gesture)
- **Total sampled:** 5,000 images (1,000 per class)
- **Split:** 80% train / 10% val / 10% test, split by `user_id` to prevent person-level leakage
