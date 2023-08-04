from flask import Flask,render_template,Response,jsonify,request
from waitress import serve
from flask_cors import CORS
import imutils
import numpy as np
import cv2
flag = False
area_pts = np.array([[150, 160],[500, 160],[500, 360],[150, 360]])
app = Flask(__name__)
#camera=cv2.VideoCapture(0)
CORS(app)
from ultralytics import YOLO
model = YOLO('best.pt')

def generate_frames():
    global flag
    camera = cv2.VideoCapture("pruebas.mp4")    
    while (camera.isOpened()):    
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame = imutils.resize(frame, width=720,height=480)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




def generatePrediction():
    global flag 
    global area_pts
    flag= True
    camera2 = cv2.VideoCapture("pruebas.mp4")
    while (camera2.isOpened()):
        success,frame=camera2.read()
        if not success:
            break
        else:
            frame = imutils.resize(frame, width=736,height=480)
            imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
            imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
            image_area = cv2.bitwise_and(frame, frame, mask=imAux) #esta 
            model.predict(image_area, save=False, imgsz=736, conf=0.8)
            res = model(image_area)
            res_plotted = res[0].plot()
            _, buffer = cv2.imencode('.jpg', res_plotted)
            data_procesed = buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + data_procesed + b'\r\n')
            if flag == False:
                break



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/setCameraInactive',methods=['POST'])
def disconnectCamera():
    global flag
    flag = False
    return jsonify({"status":"ok"})

@app.route('/setCameraActive')
def connectCamera():
    return Response(generatePrediction(),mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/setParams', methods=['POST'])
def setParameters():
    global area_pts
    json_data = request.get_json(force=True) 
    area_pts = np.array([json_data['p1'],json_data['p2'],json_data['p3'],json_data['p4']])
    dictToReturn = {"status":"ok"}
    return jsonify(dictToReturn)


@app.route('/test', methods=['POST'])
def test():
    input_json = request.get_json(force=True) 
    # force=True, above, is necessary if another developer 
    # forgot to set the MIME type to 'application/json'
    print('data from client:', input_json)
    dictToReturn = {'answer':42}
    return jsonify(dictToReturn)

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

mode = "dev"
if __name__ == '__main__':
    if mode == "dev":
        app.run(host='0.0.0.0', port=50100, debug=True)
    else:
        serve(app, host='0.0.0.0', port=50100, threads=4)


