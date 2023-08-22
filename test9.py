import cv2
import imutils
import numpy as np


frame = cv2.imread('pruebas2.jpg')
frame = imutils.resize(frame, width=960,height=720)
print(frame.shape[0:2])
area_pts = np.array([[190, 120],[230, 120],[230, 150], [190, 150]])
area_pts2 = np.array([[240, 120],[280, 120],[280, 150], [240, 150]])
img_aux = np.zeros(shape=(720, 960), dtype=np.uint8)
img_aux = cv2.drawContours(img_aux, [area_pts2], -1, (255), -1)
image_area = cv2.bitwise_and(frame, frame, mask=img_aux) #esta 
cv2.imshow('template', image_area)
cv2.waitKey(0)
cv2.destroyAllWindows()