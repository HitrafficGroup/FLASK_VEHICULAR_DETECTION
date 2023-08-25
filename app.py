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
#esta linea de codigo es para abrir el archivo de configuraciones guardadas
file_contents = {}
with open('data.json') as config_file:
    aux = config_file.read()
    file_contents = json.loads(aux)
flag = False
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app,cors_allowed_origins="*")
model = YOLO('modelos/best_2_1.pt')

# Store the track history
track_history = defaultdict(lambda: [])
# Loop through the video frames
LINE_START = sv.Point(106, 113)
LINE_END = sv.Point(340, 113)
line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.4,text_padding=1)
colors = sv.ColorPalette.default()
colores = ["rgba(14, 98, 81 ,  1)"]
polygons = [
np.array([
[0,0],[640, 0],[640, 360],[0, 360],[0, 0]
])
]
print(file_contents)
aux_areas = []
aux_colores = []
for data in file_contents:
    aux_areas.append(np.array((data['points'])))
    aux_colores.append(data['stroke'])
    if len(aux_areas) == 0:
        aux_areas = polygons
    else:
        colores = aux_colores
#caja donde se mostraran los datos
zones = [
    sv.PolygonZone(
        polygon=polygon,
        frame_resolution_wh=(640,360)
    )
    for polygon
    in aux_areas
]
zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=zone,
        color=colors.by_idx(1),
        thickness=1,
        text_thickness=1,
        text_scale=0.4
    )
    for index, zone
    in enumerate(zones)
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

#variables para enviar
datos_ia = [{"detecciones":0,"conteo":0,"color":'rgba(14, 98, 81 ,  1)'}]


def generatePrediction():
    global flag 
    global zones
    global zone_annotators
    global box_annotators
    global datos_ia
    global colores
    global line_counter
    video_path = "videos/pruebas1.mp4"
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

                for zone, zone_annotator, box_annotator,color in zip(zones, zone_annotators, box_annotators,colores):

                    mask = zone.trigger(detections=detections)
                    detections_filtered = detections[mask]
                    frame = box_annotator.annotate(scene=frame, detections=detections_filtered,labels=labels)
                    #frame = zone_annotator.annotate(scene=frame)
                    line_counter.trigger(detections=detections_filtered)
                    line_annotator.annotate(frame=frame, line_counter=line_counter)
                    aux_datos_ia.append({"detecciones":len(detections_filtered),"conteo":line_counter.out_count,"color":color})

                datos_ia = aux_datos_ia
                    
                _, buffer = cv2.imencode('.jpg', frame)
                data_procesed = buffer.tobytes()
            else:
                _, buffer = cv2.imencode('.jpg', frame)
                data_procesed = buffer.tobytes()
                line_counter.out_count = 0
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + data_procesed + b'\r\n')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break




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
    zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=zone,
        color=colors.by_idx(index+1),
        thickness=1,
        text_thickness=1,
        text_scale=0.4
    )
    for index, zone
    in enumerate(zones)
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
    json_data = request.get_json(force=True) 
    line_points = json_data['line_position']
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
