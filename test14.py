from ultralytics import YOLO
import cv2
import supervision as sv
import imutils
import numpy as np
resolution = [960,540]
LINE_START = sv.Point(460, 200)
LINE_END = sv.Point(820, 200)
'''
contiene contador de vehiculos , y rastreador bytetrack
'''
def main():
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
   
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    model = YOLO('best_2_1.pt')
    for result in model.track(source="corto1.mp4",stream=True,agnostic_nms=True):
        #image_area = cv2.bitwise_and(frame, frame, mask=mask)

        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)


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
        line_counter.trigger(detections=detections)
        line_annotator.annotate(frame=frame, line_counter=line_counter)
        cv2.imshow("yolov8 inference",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__ == "__main__":
    main()


        