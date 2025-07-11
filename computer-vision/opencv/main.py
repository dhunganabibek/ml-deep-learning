import cv2
import numpy as np

image = cv2.imread('./image.png')
image = np.where(image < 100, image, 0)
cv2.imshow('Image', image)
cv2.waitKey(0)