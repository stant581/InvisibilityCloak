import numpy as np
import cv2

detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

c = cv2.VideoCapture(0)

while True:
    ret,image = c.read()

    gray = cv2.cvtColor(image, cv2.COLOUR_BGR2GRAY)

    faces = detect.detectMultiScale(gray, 1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0),2)


    cv2.imshow('frame', image)


    if cv2.waitKey(1):
        break

    c.release()

    cv2.destroyAllWindows()

