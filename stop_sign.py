import cv2 as cv
import numpy as np

img = cv.imread("stop_sign_dataset/premium_photo-1731192705955-f10a8e7174d2.jpg")
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lower_red1 = np.array([0, 60, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 60, 50])
upper_red2 = np.array([180, 255, 255])

mask1 = cv.inRange(hsv, lower_red1, upper_red1)
mask2 = cv.inRange(hsv, lower_red2, upper_red2)
mask = cv.bitwise_or(mask1, mask2)

kernel = np.ones((7, 7), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

image_area = img.shape[0] * img.shape[1]
detected = False

for cnt in contours:
    area = cv.contourArea(cnt)
    if area < image_area * 0.001: 
        continue

    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.04 * peri, True)
    x, y, w, h = cv.boundingRect(approx)
    aspect_ratio = w / h


    if (7 <= len(approx) <= 9) and (0.8 <= aspect_ratio <= 1.2):
        cx, cy = x + w // 2, y + h // 2
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.circle(img, (cx, cy), 5, (255, 0, 0), -1)
        print(f"Stop sign detected at center: ({cx}, {cy})")
        detected = True
        break

if not detected:
    print("No stop sign detected.")

cv.imshow("Mask", mask)
cv.imshow("Detected Stop Sign", img)
cv.waitKey(0)

