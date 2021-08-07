from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import random
from math import sqrt

def Distance_finder( face_width_in_frame, Focal_Length=1200, real_face_width=15):
 
    distance = (real_face_width * Focal_Length)/face_width_in_frame
    return distance

def check_social_distancing(distances,dist_thresh=250):
	for i in range(len(distances)):
		for j in range(i+1, len(distances)):
			dist=sqrt((distances[i][0]-distances[j][0])**2 + (distances[i][1]-distances[j][1])**2 + (distances[i][1]-distances[j][1])**2)
			if(dist<=dist_thresh):
				return True,dist
	return False,-1


def detect_and_predict_mask(frame, faceNet, maskNet,default_confidence=0.5):
	# grab the dimensions of the frame and then construct a blob
	# from it

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > default_confidence:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)





#starting main
if __name__=='__main__':
	#load facenet and masknet
	faceNet = cv2.dnn.readNet("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel") #changed
	maskNet = load_model('mask_detector.model')

	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=800)

		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

		# loop over the detected face locations and their corresponding
		# locations
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

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()