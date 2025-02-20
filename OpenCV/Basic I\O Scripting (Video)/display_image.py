import cv2
import numpy as np

img = cv2.imread('slickback.png')
cv2.imshow('slickback', img)
cv2.waitKey()
cv2.destroyAllWindows()