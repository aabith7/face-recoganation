import cv2
import numpy as np
import os 
from sklearn.neighbors import KNeighborsClassifier
data = np.load("face_data.npy")

x = data[:, 1:].astype(int)
y = data[:, 0]
model = KNeighborsClassifier()
model.fit(x,y)
cam =cv2.VideoCapture(0)
rec = cv2.CascadeClassifier('haarcascade_frontalface_default (1).xml')


while True :

    ret,frame = cam.read()

    if ret :
        faces = rec.detectMultiScale(frame)
        for face in faces:
            x,y,w,h = face
            crop = frame[y:y+h , x:x+w]
            cut =  cv2.resize(crop , (200,200))
            gray = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
            out = model.predict([gray.flatten()])
            cv2.rectangle(frame, (x,y), (x+w , y+h), (0,0,255), 4 )
            cv2.putText(frame, str(out[0]), (x,y-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0), 2  ),
            
            cv2.imshow('croped' , gray)

        cv2.imshow("this is the main camera", frame)


    key = cv2.waitKey(1)

    if key == ord('e'):
        break
   

 

cam.release()
cv2.destroyAllWindows()        
