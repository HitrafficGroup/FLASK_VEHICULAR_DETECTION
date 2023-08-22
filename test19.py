
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import imutils
import threading
import time

###

###
polygons = [

np.array([
[123, 347],[163, 139],[411, 141],[395, 361],[123, 347]
])
,


np.array([
[423, 363],[451, 181],[559, 181],[573, 387],[419, 363]
])

    
]
###
model = YOLO('best_2_1.pt')
terminar_proceso = False
cantidad_vehiculos = 0
umbral = 1
time_peticion = 1
resolution = [640,480]


# Open the video file
video_path = "videos/pruebas1.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])
area_pts = np.array([[0, 0],[550, 0],[600,540], [0, 540]])
# Loop through the video frames
LINE_START = sv.Point(120, 150)
LINE_END = sv.Point(450, 150)
line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.4,text_padding=1)
###
colors = sv.ColorPalette.default()
###



zones = [
    sv.PolygonZone(
        polygon=polygon,
        frame_resolution_wh=(640,480)
    )
    for polygon
    in polygons
]
zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=zone,
        color=colors.by_idx(index),
        thickness=4,
        text_thickness=8,
        text_scale=4
    )
    for index, zone
    in enumerate(zones)
]
box_annotators = [
    sv.BoxAnnotator(
        color=colors.by_idx(index),
        thickness=4,
        text_thickness=4,
        text_scale=2
        )
    for index
    in range(len(polygons))
]

def process_frame(frame: np.ndarray, i) -> np.ndarray:
    results = model(frame, imgsz=640, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
        mask = zone.trigger(detections=detections)
        detections_filtered = detections[mask]
        frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
        frame = zone_annotator.annotate(scene=frame)

    return frame
while cap.isOpened():
    success, frame_read = cap.read()
    frame_read = imutils.resize(frame_read, width=resolution[0], height=resolution[1])
    if success:
        results = model.track(frame_read,agnostic_nms=True ,conf=0.6, persist=True,verbose=False)
        frame = results[0].orig_img
        detections = sv.Detections.from_yolov8(results[0])
        cantidad_vehiculos = len(detections)
        if results[0].boxes.id is not None:
            detections.tracker_id = results[0].boxes.id.cpu().numpy().astype(int)

        labels = [
                    f"id:{tracker_id} {model.model.names[class_id]} {confidence:0.1f}"
                    for _, confidence, class_id, tracker_id
                    in detections
                ]
        for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered,labels=labels)
            frame = zone_annotator.annotate(scene=frame)
       

        # line_counter.trigger(detections=detections)
        # line_annotator.annotate(frame=frame, line_counter=line_counter)
        # cv2.putText(frame, str(cantidad_vehiculos), [500, 150], cv2.FONT_HERSHEY_DUPLEX, 3,(231, 76, 60), 2, cv2.LINE_AA)
        cv2.imshow("yolov8 inference",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            terminar_proceso = True
            break
    else:
        terminar_proceso = True
        break
cap.release()
cv2.destroyAllWindows()


