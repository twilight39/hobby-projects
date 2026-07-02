# 1. Load a set of real chess board photos.
# 2. Detect the inner corners of the chess board in each image.
# 3. Run cv2.calibrateCamera() to get the camera matrix and distortion coefficients.
# 4. Undistort one of the images using cv2.undistort().
# 5. Print the camera matrix and distortion coefficients.
# 6. Show the original and undistorted images side by side.

from __future__ import annotations

# %% Imports
import glob
from collections.abc import Sequence
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

# %% Calibration target configuration
# The AlphaPixel dataset uses a printed board with 11 x 8 squares,
# which gives 10 x 7 inner corners.
CHESSBOARD_COLS: Final = 10
CHESSBOARD_ROWS: Final = 7
SQUARE_SIZE: Final = 1.0  # units are arbitrary for intrinsics

# 3D object points for one board image. The board is flat, so z = 0.
objp: npt.NDArray[np.float32] = np.zeros(
    (CHESSBOARD_COLS * CHESSBOARD_ROWS, 3), dtype=np.float32
)
objp[:, :2] = (
    SQUARE_SIZE * np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
)

# Termination criteria for sub-pixel corner refinement.
criteria: Final = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints: list[MatLike] = []
imgpoints: list[MatLike] = []
image_size: tuple[int, int] = (0, 0)

# %% 1. Load images and 2. detect chessboard corners
image_paths = sorted(glob.glob("assets/alphapixel_calibration/*.jpg"))
if not image_paths:
    raise RuntimeError("No calibration images found in assets/alphapixel_calibration/")

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    gray: MatLike = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_arr: npt.NDArray[np.uint8] = np.asarray(gray, dtype=np.uint8)

    found, corners = cv2.findChessboardCorners(
        gray_arr,
        (CHESSBOARD_COLS, CHESSBOARD_ROWS),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    if not found:
        continue

    refined: MatLike = cv2.cornerSubPix(
        gray_arr, corners, (11, 11), (-1, -1), criteria
    )

    objpoints.append(objp)
    imgpoints.append(refined)
    image_size = (gray_arr.shape[1], gray_arr.shape[0])

if not objpoints:
    raise RuntimeError("Chessboard corners were not detected in any image.")

print(f"Detected corners in {len(objpoints)} / {len(image_paths)} images")

# %% 3. Calibrate the camera
# OpenCV needs initial cameraMatrix/distCoeffs arrays; zeros mean "compute from scratch".
camera_matrix: MatLike = np.zeros((3, 3), dtype=np.float32)
dist_coeffs: MatLike = np.zeros((5, 1), dtype=np.float32)

rms: float
mtx: MatLike
dist: MatLike
rvecs: Sequence[MatLike]
tvecs: Sequence[MatLike]

rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, camera_matrix, dist_coeffs
)

print("\nReprojection error (RMS):", rms)
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist.ravel())

# %% 4. Undistort the first image
original = cv2.imread(image_paths[0])
if original is None:
    raise RuntimeError(f"Failed to read image: {image_paths[0]}")
original_arr: npt.NDArray[np.uint8] = np.asarray(original, dtype=np.uint8)

undistorted: MatLike = cv2.undistort(original_arr, mtx, dist, None, mtx)

# %% 6. Show original and undistorted side by side
scale = 0.2

original_small: MatLike = cv2.resize(
    original_arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
)
undistorted_small: MatLike = cv2.resize(
    undistorted, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
)
side_by_side: MatLike = np.hstack((original_small, undistorted_small))

cv2.imshow("Original (left) vs Undistorted (right)", side_by_side)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
