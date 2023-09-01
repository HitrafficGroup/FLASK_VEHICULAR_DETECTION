from flask import Flask,Response,jsonify,request
from flask_cors import CORS
from flask_socketio import SocketIO,send
from ultralytics import YOLO
from waitress import serve
import imutils
import numpy as np
import cv2
from sort import Sort
from collections import defaultdict
import supervision as sv
import json
import time
import serial_ht200
import sys
import trace
import threading
#esta linea de codigo es para abrir el archivo de configuraciones guardadas
controlador_ht200 = serial_ht200.MySerial(com="COM3")
app = Flask(__name__)
CORS(app)
model = YOLO('modelos/epoca_90.pt')

file_contents = {}
with open('data.json') as config_file:
    aux = config_file.read()
    file_contents = json.loads(aux)

flag = False
# Loop through the video frames
LINE_START = sv.Point(106, 113)
LINE_END = sv.Point(340, 113)
line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.4,text_padding=1)
colors = sv.ColorPalette.default()
colores = ["rgba(14, 98, 81 ,  1)"]
polygons = [np.array([[0,0],[640, 0],[640, 360],[0, 360],[0, 0]])]
datos_ia = [{"detecciones":0,"conteo":0,"color":'rgba(14, 98, 81 ,  1)'}]

#condicional para saber si existen datos en el archivo de configuraciones
if len(file_contents) != 0:
    polygons = []
    colores = []
    for data in file_contents:
        polygons.append(np.array((data['points'])))
        colores.append(data['stroke'])

#caso contrario emepzamos con los valores por defect
zones = [
    sv.PolygonZone(
        polygon=polygon,
        frame_resolution_wh=(640,360)
    )
    for polygon
    in polygons
]

box_annotators = [
    sv.BoxAnnotator(
        color=colors.by_idx(0),
        thickness=1,
        text_thickness=1,
        text_scale=0.2,
        text_padding=1
        )
    for index
    in range(len(polygons))
]

#variables para enviar
terminar_proceso = False

class thread_with_trace(threading.Thread):
    def __init__(self, *args, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False
    
    def start(self):
        self.__run_backup = self.run
        self.run = self.__run     
        threading.Thread.start(self)
    
    def __run(self):
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup
    
    def globaltrace(self, frame, event, arg):
        if event == 'call':
            return self.localtrace
        else:
            return None
    
    def localtrace(self, frame, event, arg):
        if self.killed:
            if event == 'line':
                raise SystemExit()
        return self.localtrace
    
    def kill(self):
        self.killed = True


umbral = 1

def setParamsController():
    global datos_ia
   
    
    while True:
        global umbral
        global terminar_proceso
        # Coloca aquí el código que deseas ejecutar cada 3 segundos
        cantidad_vehiculos = datos_ia[0]['detecciones']
        if cantidad_vehiculos >= umbral:
           
            time_peticion = cantidad_vehiculos +10
            print(f'se a detectado {cantidad_vehiculos} vehiculo semaforo en verde por {time_peticion} segundos')
            #json_data = {"trama":[15, 1, 49, 51, 0, 0, 15, 0],"ip":"192.168.1.123"}
            print(f'el umbral es {umbral}')
            controlador_ht200.sendData(fase=2,tiempo=time_peticion)
            # controlador_ht200.setControlManual(data_target=json_data['trama'],ip_controller=json_data['ip'])

            time.sleep(time_peticion+30)

        if terminar_proceso:
            break
        time.sleep(1)
    print("Finalizo el programa")


def generatePrediction():
    global flag 
    global zones
    global zone_annotators
    global box_annotators
    global datos_ia
    global colores
    global line_counter
    video_path = "videos/pruebas2.mp4"
    camera = cv2.VideoCapture(video_path)
    while (camera.isOpened()):
        success,frame=camera.read()
        if not success:
            break
        else:
            
            frame = imutils.resize(frame, width=640)
            if flag:
                results = model.track(frame,agnostic_nms=True ,imgsz=640,conf=0.6, persist=True,verbose=False)
                frame = results[0].orig_img
                detections = sv.Detections.from_yolov8(results[0])
                if results[0].boxes.id is not None:
                    detections.tracker_id = results[0].boxes.id.cpu().numpy().astype(int)

                labels = [
                    f"id:{tracker_id} {model.model.names[class_id]} {confidence:0.1f}"
                    for _, confidence, class_id, tracker_id
                    in detections
                ]
                aux_datos_ia = []

                for zone, box_annotator,color in zip(zones, box_annotators,colores):
                    mask = zone.trigger(detections=detections)
                    detections_filtered = detections[mask]
                    frame = box_annotator.annotate(scene=frame, detections=detections_filtered,labels=labels)
                    line_counter.trigger(detections=detections_filtered)
                    line_annotator.annotate(frame=frame, line_counter=line_counter)
                    aux_datos_ia.append({"detecciones":len(detections_filtered),"conteo":line_counter.out_count,"color":color})

                datos_ia = aux_datos_ia
                    
                _, buffer = cv2.imencode('.jpg', frame)
                data_procesed = buffer.tobytes()
            else:
                _, buffer = cv2.imencode('.jpg', frame)
                data_procesed = buffer.tobytes()
                datos_ia = [{"detecciones":0,"conteo":0,"color":'rgba(14, 98, 81 ,  1)'}]
                line_counter.out_count = 0
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + data_procesed + b'\r\n')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Crea un hilo y comienza a ejecutarlo
thread_controller = thread_with_trace(target = setParamsController)

thread_controller.start()
# variables del hilo




@app.route('/desactivateIA', methods=['POST'])
def disconnectIA():
    global flag
    flag = False
    return jsonify({"status":"DESACTIVATED IA"})

@app.route('/activateIA', methods=['POST'])
def connectIA():
    global flag
    flag = True
    return jsonify({"status":"ACTIVATED IA"})



@app.route('/setParams', methods=['POST'])
def setParameters():
    global colores
    global zones
    global zone_annotators
    global box_annotators
    aux_colores = []
    aux_areas = []
    json_data = request.get_json(force=True) 
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    for data in json_data:
        aux_areas.append(np.array((data['points'])))
        aux_colores.append(data['stroke'])


    if len(aux_areas) == 0:
        aux_areas = polygons
    else:
        colores = aux_colores
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=(640,360)
        )
        for polygon
        in aux_areas
    ]
    box_annotators = [
        sv.BoxAnnotator(
            color=colors.by_idx(0),
            thickness=1,
            text_thickness=1,
            text_scale=0.2,
            text_padding=1
            )
        for index
        in range(len(aux_areas))
    ]


    dictToReturn = {"status":"ok"}
    return jsonify(dictToReturn)


@app.route('/predict',methods=['GET'])
def video():
    return Response(generatePrediction(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/values',methods=['GET'])
def send_values():
    global contador
    return jsonify({"data":datos_ia})

@app.route('/setLinePos', methods=['POST'])
def setLinePosition():
    global line_counter
    global umbral   
    json_data = request.get_json(force=True) 
    line_points = json_data['line_position']
    umbral = json_data['umbral']
    print(line_points)
    LINE_START = sv.Point(line_points[0][0],line_points[0][1])
    LINE_END = sv.Point(line_points[1][0],line_points[1][1])
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)

    dictToReturn = {"status":"ok"}
    return jsonify(dictToReturn)

@app.route('/config', methods=['GET'])
def sendCurrentConfig():
    file_contents = {}
    with open('data.json') as config_file:
        file_contents = config_file.read()
    return file_contents

@app.route('/kill', methods=['POST'])
def killtreath():
    global terminar_proceso 
    terminar_proceso =True
    thread_controller.kill()
    thread_controller.join()
    dictToReturn = {"status":"ok"}
    return jsonify(dictToReturn)


@app.route('/umbral', methods=['POST'])
def updateUmbral():
    global umbral   
    json_data = request.get_json(force=True) 
    umbral = json_data['umbral']
    dictToReturn = {"status":"ok"}
    return jsonify(dictToReturn)
    

mode = "dev"
if __name__ == '__main__':
    if mode == "dev":
        app.run(host='0.0.0.0', port=50100, debug=True)
    else:
        serve(app, host='0.0.0.0', port=50100, threads=4)

# Parametros de la red neuronal
#  tipo de vehiculos que se quiere detectar
#  cantidad de zonas de inferencia
#  parametros del sistema de trackeo
#  activacion y desactivacion de la IA
#  logica de tiempos, si detecta 5 vehiculos cuantos segundos debe mandar al controlador
