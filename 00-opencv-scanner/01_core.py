# 1. Load the image chip.png from disk.
# 2. Prints its height, width, and number of channels.
# 3. Converts it to grayscale.
# 4. Shows both the color and grayscale versions in two separate windows.
# 5. Save the grayscale image as chip_gray.png.

# %% Imports
import cv2

# Read the image
image = cv2.imread("assets/base.png")
if image is None:
    raise Exception("Image failed to load.")

print(f"{image.shape = }")
image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Image", image)
cv2.imshow("Grayscale Image", image_grayscale)

cv2.imwrite("assets/base_grayscale.png", image_grayscale)

cv2.waitKey(0)
cv2.destroyAllWindows()

# This is needed for Mac for some reason
cv2.waitKey(0)
