# 1. Load chip.png or any image.
# 2. Convert it to grayscale.
# 3. Compute and displays:
#   a Sobel X gradient,
#   b Canny edge map,
#   c 45-degree rotated version of the original color image.
# 4. Save the Canny edge map as chip_edges.png.

# %% Imports
import cv2
import numpy as np

# 1
image = cv2.imread("assets/base.png")
if image is None:
    raise Exception("Image failed to load")

# 2
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3a
edge_64f = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
edge_abs = np.absolute(edge_64f)
edge_8u = np.uint8(edge_abs)

# 3b
canny = cv2.Canny(grey, 100, 255)

# 3c
h, w = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))

# 4
cv2.imwrite("assets/canny.png", canny)

cv2.imshow("Grayscale", grey)
cv2.imshow("Edge", edge_8u)  # pyright: ignore[reportArgumentType, reportCallIssue]
cv2.imshow("Canny", canny)
cv2.imshow("Original", image)
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
