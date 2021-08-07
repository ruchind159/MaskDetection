# install opencv "pip install opencv-python"
import cv2

# distance from camera to object(face) measured
# centimeter
Known_distance = 74

# width of face in the real world or Object Plane
# centimeter
Known_width = 15
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):

	# finding the focal length
	focal_length = (width_in_rf_image * measured_distance) / real_width
	return focal_length



def face_data(image):

	face_width = 0 # making face width to zero

	# converting color image ot gray scale image
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detecting face in the image
	faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

	# looping through the faces detect in the image
	# getting coordinates x, y , width and height
	for (x, y, h, w) in faces:

		# draw the rectangle on the face
		cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)

		# getting face width in the pixels
		face_width = w

	# return the face width in pixel
	return face_width


# reading reference_image from directory
ref_image = cv2.imread("ref_image.jpg")

# find the face width(pixels) in the reference_image
ref_image_face_width = face_data(ref_image)

# get the focal by calling "Focal_Length_Finder"
# face width in reference(pixels),
# Known_distance(centimeters),
# known_width(centimeters)
Focal_length_found = Focal_Length_Finder(
	Known_distance, Known_width, ref_image_face_width)

print(Focal_length_found)

# show the reference image
cv2.imshow("ref_image", ref_image)



##########################
#Focal length =1258