#este test es para usar la api en vez de dejar correr nuestra propia red

import json
import requests
import numpy as np
import cv2
import imutils
from ultralytics import YOLO
from sort import Sort

camera2 = cv2.VideoCapture("pruebas.mp4")
tracker = Sort()

# Run inference on an image
url = "https://api.ultralytics.com/v1/predict/qVwusF28GI44Jvh5E868"
headers = {"x-api-key": "a8b29c45669c6d2f8a2637acee48e9f83d3eef0db7"}
data = {"size": 640, "confidence": 0.25, "iou": 0.45}
while (camera2.isOpened()):
        success,frame=camera2.read()
        if not success:
            break
        else:
            frame = imutils.resize(frame, width=640,height=480)
            _, buffer = cv2.imencode('.jpg', frame)
            data_procesed = buffer.tobytes()
            files = {"image": data_procesed}
            response = requests.post(url, headers=headers, files=files, data=data)
            #response.raise_for_status()
            parsed_response = response.json()
            data_from_results = parsed_response['data']
            for box in data_from_results:
                cv2.putText(img=frame, text="Car", org=(int(box['box']['x1']), int(box['box']['y1'])-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,0), thickness=2)
                cv2.rectangle(img=frame, pt1=(int(box['box']['x1']), int(box['box']['y1'])), pt2=(int(box['box']['x2']), int(box['box']['y2'])), color=(0, 255, 0), thickness=2)
            cv2.imshow('Frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

