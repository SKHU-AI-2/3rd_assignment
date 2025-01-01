# -*- coding: utf-8 -*-
"""edge_detection_Sobel

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GpcYJf7UNpKIvqeg3iYigIcYQd8hGYLh
"""

import cv2
import numpy as np

sobel_gy = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

sobel_gx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

image = cv2.imread("lena.png")
original = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float32)

edges_x = cv2.filter2D(gray, -1, sobel_gx)
edges_y = cv2.filter2D(gray, -1, sobel_gy)
edges = cv2.magnitude(edges_x, edges_y)

edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow("Original Image", original)
cv2.imshow("Sobel Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()