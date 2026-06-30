# Pipeline
# 1. Capture a frame from the webcam.
# 2. Convert to grayscale and blur to reduce noise.
# 3. Run Canny edge detection.
# 4. Find contours and select the largest 4-sided polygon.
# 5. Compute a perspective transform to a rectangle.
# 6. Save the flattened document image.

# %% Imports
import cv2

# %%
