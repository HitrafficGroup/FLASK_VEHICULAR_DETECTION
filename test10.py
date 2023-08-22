import numpy as np
import cv2
import imutils
from ultralytics import YOLO

area_pts = np.array([[193, 129],[470, 121],[631, 345], [30, 350]])
camera = cv2.VideoCapture("pruebas.mp4")
while (camera.isOpened()):
    success,frame=camera.read()
    if not success:
        break
    else:
        frame = imutils.resize(frame, width=640,height=360)
        print(frame.shape[0:2])
        mask = np.zeros(shape=((360, 640)), dtype=np.uint8)
        mask = cv2.drawContours(mask, [area_pts], -1, (255), -1)
        dst = cv2.bitwise_or(frame,frame,mask=mask)
        #image_area = cv2.bitwise_and(frame, frame, mask=mask) #esta 
        cv2.imshow('Frame',dst)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        
 