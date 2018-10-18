#Kaivalya Vohra 2018
#Train face recognition model

#import libraries
import cv2
import os
import numpy as np
from PIL import Image
import pickle

#set up directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

#point to face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Create new recognizer object
recognizer = cv2.face.LBPHFaceRecognizer_create()

#inintialize variables
current_id = 0
label_ids = {}
y_labels = []
x_train = []

#loop through image directory
for root, dirs, files in os.walk(image_dir):
	for file in files:
                #only look at image files
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			#label is the folder name
			label = os.path.basename(root).replace(" ", "-").lower()
			#if label doesn't already exist create new label
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]

			#convert image to grayscale
			pil_image = Image.open(path).convert("L") # grayscale
                        #resize image
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			#convert image into numpy array
			image_array = np.array(final_image, "uint8")
			#detetct faces in images
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
                        # append region of interest to x_train
                        #append label's id to y_labels
			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)


#pickle converts objects to bytestream and vice versa
#converts labels dictionary to byte stream to be passed to faces.py
with open("pickles/face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)
#train recognizer and save trained model in face-trainner.yml
recognizer.train(x_train, np.array(y_labels))
recognizer.save("/face-trainner.yml")
