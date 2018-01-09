
import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
path='dataSet'

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids


faces,Ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')


import cv2
import numpy as np
import pyttsx3

k=pyttsx3.init()

rec = cv2.face.LBPHFaceRecognizer_create()
rec.train(faces, np.array(Ids))
#rec.train('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceDetect = cv2.CascadeClassifier(cascadePath);
Id=0
cam = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        #rec.train(faces, np.array(Ids))

        Id,conf = rec.predict(gray[y:y+h,x:x+w])
        def say(text):
            k.say(text)
            k.runAndWait()
        if(Id==1):
            Id="Urvish"
            say('Urvish')
        elif(Id==2):
            Id="Utkarsh"
            say('Utkarsh')
        
        elif(Id==4):
            Id="Harsh"
            say('Harsh')
        else:
            Id="Unknown"
            say('unkown')
        cv2.putText((im),str(Id), (x,y+h),font, 255,255)
        
    if cv2.waitKey(1)==ord('q'):
        break
    cv2.imshow('im',im) 
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
