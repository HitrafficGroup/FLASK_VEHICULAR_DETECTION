from roboflow import Roboflow
import cv2 
import imutils

rf = Roboflow(api_key="puhMpjdS8tFGClvbvXKl")
project = rf.workspace().project("vehicles-q0x2v")
model = project.version(1).model

cap = cv2.VideoCapture('pruebas.mp4')

while(cap.isOpened()):
  # f.read() methods returns a tuple, first element is a bool 
  # and the second is frame
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Frame',frame)
        predictions = model.predict(frame)
        predictions_json = predictions.json()
  
        # printing all detection results from the image
        print(predictions_json)

        # accessing individual predicted boxes on each image
        for bounding_box in predictions:
            # x0 = bounding_box['x'] - bounding_box['width'] / 2#start_column
            # x1 = bounding_box['x'] + bounding_box['width'] / 2#end_column
            # y0 = bounding_box['y'] - bounding_box['height'] / 2#start row
            # y1 = bounding_box['y'] + bounding_box['height'] / 2#end_row
            class_name = bounding_box['class']
            confidence_score = bounding_box['confidence']
        
            detection_results = bounding_box
            class_and_confidence = (class_name, confidence_score)
            print(class_and_confidence, '\n')
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()