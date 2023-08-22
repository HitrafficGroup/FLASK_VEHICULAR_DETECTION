import numpy as np
import cv2
import imutils
from ultralytics import YOLO
from sort import Sort
model = YOLO('best_2.pt')
area_pts = np.array([[193, 129],[470, 121],[651, 345], [30, 350]])
camera2 = cv2.VideoCapture("pruebas2.mp4")
tracker = Sort()
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
car_counter = 0
isClosed = True
 
# Blue color in BGR
color = (255, 0, 0)
 
# Line thickness of 2 px
thickness = 2
flag2 = False
while (camera2.isOpened()):
    success,frame=camera2.read()
    if not success:
        break
    else:
        frame = imutils.resize(frame, width=640,height=480)
        imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
        imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
     
        image_area = cv2.bitwise_and(frame, frame, mask=imAux) #esta 
        image = cv2.polylines(frame, [area_pts],
                      isClosed, color, thickness)
        results = model(image_area,stream=True)
        for res in results:
            filtered_indices = np.where((np.isin(res.boxes.cls.cpu().numpy(),[4])) & (res.boxes.conf.cpu().numpy() > 0.5))[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)
            for xmin, ymin, xmax, ymax, track_id in tracks:
                cv2.putText(img=frame, text=f"Id: {track_id}", org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

        cv2.imshow('Frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        
 