from flask import Flask, Response
import cv2
import imutils
from mask_detect import *

faceNet = cv2.dnn.readNet("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel") #changed
maskNet = load_model('mask_detector.model')




app = Flask(__name__)
video = cv2.VideoCapture(0)
@app.route('/')
def index():
    return "Mask Detection"
def gen(video):
    while True:
        # success, image = video.read()
        success, frame = video.read()
        frame = imutils.resize(frame, width=800)

        #code for detecting and predicting
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        alert_label='No Face Detected'
        distance_from_cam=[]

        for ID,(box, pred) in enumerate(zip(locs, preds)):
            alert_label=''
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            #storing distance of each detected face
            x=(startX + endX)/2
            y=(startY + endY)/2
            z=Distance_finder(endX-startX)
            distance_from_cam.append((x,y,z))


            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
            # include the probability in the label
            label = "ID = {}  {}: {:.2f}%".format(ID,label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        if(len(distance_from_cam)>1):
            flag,dist=check_social_distancing(distance_from_cam)
            if(flag):
                alert_label=f'Maintain Social Distancing (distance between people = {dist}cm)'


        textsize = cv2.getTextSize(alert_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        textX = (frame.shape[1] - textsize[0]) // 2
        textY = (frame.shape[0] + textsize[1]) // 2
        if(alert_label!=''):
            cv2.rectangle(frame, (textX,textY), (textX+textsize[0], textY-textsize[1]), (0,0,0), -1)
        cv2.putText(frame, alert_label, (textX,textY),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)




        #for returning
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(threaded=True,debug=True)