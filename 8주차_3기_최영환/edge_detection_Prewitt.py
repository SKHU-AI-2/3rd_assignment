import cv2
import numpy as np

prewitt_detector_gy = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
], dtype=np.float32)

prewitt_detector_gx = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

image = cv2.imread("lena.png")
original = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float32)

edges_x = cv2.filter2D(gray, -1, prewitt_detector_gy)
edges_y = cv2.filter2D(gray, -1, prewitt_detector_gx)
edges = cv2.magnitude(edges_x, edges_y)

edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow("Original Image", original)
cv2.imshow("Prewitt", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()