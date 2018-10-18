#Kaivalya Vohra 2018
#Run facial recognition on live video stream

#import libraries
import numpy as np
import cv2
import pickle

#main detection function
def detect():
    #point to face cascade(used later in program) 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #create a new recognizer object
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #read training file created by faces-train.py
    recognizer.read("recognizers/face-trainner.yml")
    
    labels = {}
    with open("pickles/face-labels.pickle", 'rb') as f:
        #pickle converts objects to bytestream and vice versa
        #converts the byte stream of labels created in faces-train.py to a dictionary.
        og_labels = pickle.load(f)
        #reverse loaded labels' key/value so they look like {0:"kv",1:"john"}
        labels = {v: k for k, v in og_labels.items()}

    #start video streamm from defualt camera
    cap = cv2.VideoCapture(0)
    #run infinite loop
    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()
        #grayscale video feed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Faces is array of all faces detected using haar casacade
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.5, minNeighbors=5)
        #for x coord, y coord, width, height in faces
        for (x, y, w, h) in faces:
            #region of interest gray.colour
            roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
            roi_color = frame[y:y + h, x:x + w]

            #id(0,1,2 etc) and confidence when running facerecog
            id_, conf = recognizer.predict(roi_gray)

            #values of confidence can be tweaked in the if statement here
            if  conf >= 40 and conf<=85:
                #Write name
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                final = name
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x, y), font, 1,color, stroke, cv2.LINE_AA)
            #draw box around face
            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x,
                                          end_cord_y), color, stroke)
         
        # Display the resulting frame
        cv2.imshow('frame', frame)

        #quit if 'q' is pressed
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

#run main function
detect()
