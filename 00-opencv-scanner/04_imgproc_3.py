# 1. Create a 400×400 black image.
# 2. Draws two white shapes on it: a filled rectangle and a filled circle.
# 3. Adds a little noise, then cleans it with thresholding and morphology so you get a clean binary mask.
# 4. Finds the contours of the shapes.
# 5. Filters out any contour with an area smaller than 500 pixels.
# 6. For each remaining contour:
# 7. Draw the contour in green.
# 8. Draw a red bounding rectangle around it.
# 9. Print the area and perimeter.
# 10. Show the result.

# %% Imports
import cv2
import numpy as np

# 1
image = np.zeros((400, 400), np.uint8)

# 2
cv2.rectangle(image, (50, 50), (100, 350), 255, -1)
cv2.circle(image, (250, 150), 100, 255, -1)

# 3
noise = np.random.choice([0, 255], size=image.shape, p=[0.95, 0.05]).astype(np.uint8)
noisy_image = np.bitwise_or(noise, image)

blur = cv2.GaussianBlur(noisy_image, (3, 3), 0)
ret, thres = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((3, 3), np.uint8)
morph = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)

# 4
cnt, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt_img = cv2.drawContours(np.zeros((400, 400), np.uint8), cnt, -1, 255, 1)

# 5 6 7 8 9
res = np.zeros((400, 400, 3), np.uint8)
res_cnt = []
for c in cnt:
    if cv2.contourArea(c) < 500:
        continue

    x, y, w, h = cv2.boundingRect(c)
    cv2.drawContours(res, c, -1, (0, 255, 0), 1)
    cv2.rectangle(res, (x, y), (x + w, y + h), (0, 0, 255), 2)

    print("Contour found:")
    print(f"{cv2.arcLength(c, True) = }")
    print(f"{cv2.contourArea(c) = }\n")

# 10
cv2.imshow("Image", image)
cv2.imshow("Noise", noise)
cv2.imshow("Noisy Image", noisy_image)
cv2.imshow("Blur", blur)
cv2.imshow("Threshold", thres)
cv2.imshow("Morph", morph)
cv2.imshow("Contour Image", cnt_img)
cv2.imshow("Result", res)

cv2.waitKey(0)
cv2.destroyAllWindows()
