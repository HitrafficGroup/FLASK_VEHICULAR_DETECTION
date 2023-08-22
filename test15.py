from ultralytics import YOLO
import cv2
import supervision as sv
import imutils
import numpy as np
resolution = [960,540]


'''
dispone de masacara , y rastreador byte track
'''
def main():
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    model = YOLO('best_2_1.pt')
    camera = cv2.VideoCapture("corto1.mp4")
    area_pts = np.array([[400, 200],[840, 200],[940, 700], [20, 700]])

    while (camera.isOpened()):
        success,frame=camera.read()
        if not success:
            break
        else:
         
            mask = np.zeros(shape=(frame.shape[0:2]), dtype=np.uint8)
            cv2.drawContours(mask, [area_pts], -1, (255), -1)
            image_area = cv2.bitwise_and(frame, frame, mask=mask)
            result = model.track(source=image_area,stream=True,agnostic_nms=True,conf=0.65)
        
            detections = sv.Detections.from_yolov8(result[0])


            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            labels = [
                f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
                ]
            frame = box_annotator.annotate(
                scene=frame, 
                detections=detections,
                labels=labels
            )
            
            cv2.imshow("yolov8 inference",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__ == "__main__":
    main()


        