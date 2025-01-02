import cv2
import numpy as np



image = cv2.imread('snow_example.png')
original = image.copy()


hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_white = np.array([0, 10, 120])  # H, S, V 순서
upper_white = np.array([180, 60, 255])  # H, S, V 순서

mask = cv2.inRange(hsv, lower_white, upper_white)


kernel = np.ones((2, 2), np.uint8)

mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 열림(작은 흰색 점 제거)
mask_cleaned_2 = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)  # 닫힘(작은 검은색 구멍 채우기)



cv2.imshow("Original Image", original)
cv2.imshow("Mask", mask_cleaned_2)

cv2.waitKey(0)
cv2.destroyAllWindows()