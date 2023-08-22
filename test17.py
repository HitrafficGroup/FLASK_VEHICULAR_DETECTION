
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import imutils
import threading
import time
model = YOLO('best_2_1.pt')
terminar_proceso = False
cantidad_vehiculos = 0
umbral = 1
time_peticion = 1
resolution = [960,540]
def mi_funcion():
    while True:
        # Coloca aquí el código que deseas ejecutar cada 3 segundos
        if cantidad_vehiculos > umbral:
            print(f'se a detectado un vehiculo semaforo en verde por {cantidad_vehiculos*10}')
            time_peticion = cantidad_vehiculos *10
            time.sleep(time_peticion)
        else:
            print(f'no se a detectado un vehiculo semaforo en rojo')
        if terminar_proceso:
            print("Finalizo el programa")
            break
          # Espera 3 segundos

# Crea un hilo y comienza a ejecutarlo
t = threading.Thread(target=mi_funcion)
t.start()
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

box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=0.2,
        text_padding=1
    )
while cap.isOpened():
    success, frame_read = cap.read()
    frame_read = imutils.resize(frame_read, width=resolution[0], height=resolution[1])
    mask = np.zeros(shape=(frame_read.shape[0:2]), dtype=np.uint8)
    cv2.drawContours(mask, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(frame_read, frame_read, mask=mask)
    image_area = imutils.resize(image_area, width=640,height=480)
    if success:

        results = model.track(image_area,agnostic_nms=True ,conf=0.6, persist=True,verbose=False)
       
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

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
            
        )

        line_counter.trigger(detections=detections)
        line_annotator.annotate(frame=frame, line_counter=line_counter)
        cv2.putText(frame, str(cantidad_vehiculos), [500, 150], cv2.FONT_HERSHEY_DUPLEX, 3,(231, 76, 60), 2, cv2.LINE_AA)
        cv2.imshow("yolov8 inference",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            terminar_proceso = True
            break
    else:
        terminar_proceso = True
        break
cap.release()
cv2.destroyAllWindows()
