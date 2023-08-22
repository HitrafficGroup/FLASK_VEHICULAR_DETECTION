import numpy as np
import cv2
import imutils
from ultralytics import YOLO
from sort import Sort

resolution = [960,540]
def main():
    model = YOLO('best_5.pt')
    area_pts = np.array([[370, 200],[650, 200],[740, 540], [100, 540]])
    camera = cv2.VideoCapture("corto1.mp4")
    tracker = Sort()

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = imutils.resize(frame, width=resolution[0], height=resolution[1])
        mask = np.zeros(shape=(resolution[1], resolution[0]), dtype=np.uint8)
        cv2.drawContours(mask, [area_pts], -1, (255), -1)
        #image_area = cv2.bitwise_and(frame, frame, mask=mask)
        
        results = model(frame,conf=0.7)       # Visualize the results on the frame
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


        
 