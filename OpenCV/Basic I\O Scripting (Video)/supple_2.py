import cv2

img = cv2.imread('slickback.jpg')
cv2.imshow("character_face", img[150:451, 192:499])
cv2.waitKey()
cv2.destroyAllWindows()