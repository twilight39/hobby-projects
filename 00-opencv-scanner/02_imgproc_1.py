# 1. Create a black image of size 300×300.
# 2. Draw a white filled rectangle somewhere in the middle.
# 3. Add random black-and-white noise speckles across the whole image.
# 4. Use a blur to reduce the noise.
# 5. Convert the result to a binary mask where the bright rectangle is white and the background is black.
# 6. Use a morphological operation to fill any small holes inside the rectangle.
# 7. Show the original noisy image and the final cleaned mask side by side.

# %% Imports

import cv2
import numpy as np

# 1
img = np.zeros((300, 300), dtype=np.uint8)

# 2
cv2.rectangle(img, (50, 50), (100, 200), 255, -1)

# 3
noise = np.random.choice([0, 255], size=img.shape, p=[0.95, 0.05]).astype(np.uint8)
noisy = cv2.bitwise_or(img, noise)

# 4
gaussian = cv2.GaussianBlur(noisy, (3, 3), 0)
median = cv2.medianBlur(noisy, 3)

# 5
ret, thres = cv2.threshold(gaussian, 254, 255, cv2.THRESH_BINARY)

# 6
kernel = np.ones((5, 5), np.uint8)
clean = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)

# %%
cv2.imshow("Noisy", gaussian)
cv2.imshow("Clean", clean)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
