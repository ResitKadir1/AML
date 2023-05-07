
from keras.models import Sequential,Model

import keras
import numpy as np
import cv2
import tensorflow as tf
#from tensorflow.keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

from tensorflow.python.framework import tensor_util
from PIL import Image
#from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from os import listdir
from helperScript import loadVggFaceModel
from keras.models import model_from_json
#-----------------------

def is_tensor(x):
    return tensor_util.is_tensor(x)
    #return isinstance(x, tf_ops._TensorLike) or tf_ops.is_dense_tensor_like(x)

color = (67,67,67) #gray

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


model = loadVggFaceModel()

base_model_output = keras.Sequential()
base_model_output = Convolution2D(6, (1, 1), name='predictions')(model.layers[-4].output)
base_model_output = Flatten()(base_model_output)
base_model_output = Activation('softmax')(base_model_output)

race_model = Model(inputs=model.input, outputs=base_model_output)

race_model.load_weights('race_model_single_batch.h5')

#------------------------

races = ['Asian', 'Indian', 'Black', 'White', 'Middle-Eastern', 'Latino_Hispanic']

#------------------------
##record
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
videoWriter = cv2.VideoWriter('video1.mp4', fourcc, 30.0, (660,440)) #30 FPS

##
cap = cv2.VideoCapture("faces.mp4") #webcam

while(True):
	ret, img = cap.read()
	#img = cv2.imread("test_image/pogba.jpg")#F011_T01.mp4
	#img = cv2.imread("F011_T01.mp4")
	img = cv2.resize(img, (660, 440))
	faces = face_cascade.detectMultiScale(img, 1.3, 5)

	for (x,y,w,h) in faces:
		if w > 130:
			cv2.rectangle(img,(x,y),(x+w,y+h),(67, 67, 67),2) #draw rectangle to main image

			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face

			#add margin
			margin_rate = 30
			try:
				margin_x = int(w * margin_rate / 100)
				margin_y = int(h * margin_rate / 100)

				detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
				detected_face = cv2.resize(detected_face, (224, 224)) #resize to 224x224

				#display margin added face
				cv2.rectangle(img,(x-margin_x,y-margin_y),(x+w+margin_x,y+h+margin_y),(67, 67, 67),1)

			except Exception as err:
				#print("margin cannot be added (",str(err),")")
				detected_face = img[int(y):int(y+h), int(x):int(x+w)]
				detected_face = cv2.resize(detected_face, (224, 224))

			#print("shape: ",detected_face.shape)

			if detected_face.shape[0] > 0 and detected_face.shape[1] > 0 and detected_face.shape[2] >0: #sometimes shape becomes (264, 0, 3)


				#img_pixels = image.img_to_array(detected_face)
				img_pixels =tf.keras.utils.img_to_array(detected_face)
				img_pixels = np.expand_dims(img_pixels, axis = 0)
				img_pixels /= 255

				prediction_proba = race_model.predict(img_pixels)
				prediction = np.argmax(prediction_proba)

				if False: # activate to dump
					for i in range(0, len(races)):
						if np.argmax(prediction_proba) == i:
							print("* ", end='')
						print(races[i], ": ", prediction_proba[0][i])
					print("----------------")

				race = races[prediction]
				#--------------------------
				#background
				overlay = img.copy()
				opacity = 0.8
				cv2.rectangle(img,(x+w,y-40),(x+w+200,y+10),(64,64,64),cv2.FILLED)
				cv2.addWeighted(overlay, opacity, img, 1 - opacity, 3, img)

				color = (0,255,0)
				proba = round(100*prediction_proba[0, prediction], 2)

				if proba >= 51:
					label = str(race+" ("+str(proba)+"%)")
					cv2.putText(img, label, (int(x+w+10), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

					#connect face and text
					cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),(67, 67, 67),1)
					cv2.line(img,(x+w,y),(x+w+10,y-20),(67, 67, 67),1)
					videoWriter.write(img)

	cv2.imshow('AML',img)
	cv2.putText(img,"AML ",(220,700),cv2.FONT_HERSHEY_SIMPLEX,2,(67,67,67))
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break

#kill open cv things
cap.release()
videoWriter.release()
cv2.destroyAllWindows()
