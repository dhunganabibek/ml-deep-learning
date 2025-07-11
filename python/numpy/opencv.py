import cv2
import numpy as np

img = np.array([np.arange(0,10),np.arange(10,20), np.arange(30, 40)])
img = np.random.rand(100,200)* 100
img = img.astype(np.uint8)
cv2.imshow('Random',img)
cv2.waitKey(0)
