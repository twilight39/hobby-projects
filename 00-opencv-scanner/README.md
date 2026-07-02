# Project 0: OpenCV Document Scanner

A learning project that practices the OpenCV operations that appear in almost every machine vision pipeline.

## What this folder contains

A set of small runnable modules covering image I/O, color spaces, filtering, edge detection, contours, geometric transforms, and camera calibration.

## Motivation

OpenCV is the main gap before moving on to YOLO-based projects. This project forces practice with the core building blocks: image I/O, color spaces, filtering, edge detection, contours, and geometric transforms.

## Topics covered

| Module | Topic | OpenCV functions |
|---|---|---|
| `01_core.py` | Image I/O & windows | `cv2.imread`, `cv2.imshow`, `cv2.VideoCapture`, `cv2.imwrite` |
| `02_imgproc_1.py` | Color spaces | `cv2.cvtColor` |
| `03_imgproc_2.py` | Filtering | `cv2.GaussianBlur`, `cv2.medianBlur` |
| `04_imgproc_3.py` | Edge detection & contours | `cv2.Canny`, `cv2.findContours`, `cv2.approxPolyDP`, `cv2.contourArea` |
| `05_caleb3d.py` | Camera calibration | `cv2.findChessboardCorners`, `cv2.cornerSubPix`, `cv2.calibrateCamera`, `cv2.undistort` |
| ... | Perspective transform | `cv2.getPerspectiveTransform`, `cv2.warpPerspective` |
| ... | Drawing | `cv2.drawContours`, `cv2.polylines`, `cv2.line` |

## Document scanner pipeline

1. Capture a frame from the webcam.
2. Convert to grayscale and blur to reduce noise.
3. Run Canny edge detection.
4. Find contours and select the largest 4-sided polygon.
5. Compute a perspective transform to a rectangle.
6. Save the flattened document image.

## Files

- `01_core.py` — image I/O basics
- `02_imgproc_1.py` — color spaces
- `03_imgproc_2.py` — filtering
- `04_imgproc_3.py` — edge detection and contours
- `05_caleb3d.py` — camera calibration using a chessboard pattern
- `assets/` — sample images and the AlphaPixel calibration dataset
- `README.md` — this file

## Calibration dataset

Module `05_caleb3d.py` uses the real chessboard calibration images from AlphaPixel's OpenCV tutorial:

- Post: https://alphapixeldev.com/opencv-tutorial-part-1-camera-calibration/
- Download: https://alphapixeldev.com/wp-content/uploads/2024/08/Cleaned.zip

The images are extracted to `assets/alphapixel_calibration/`.

## Applications

- Document scanning apps
- Receipt digitization
- Flattening whiteboard or paper captures
- Preprocessing step for OCR pipelines
