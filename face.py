import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')


while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    #smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x1, y1, w1, h1) in faces: 
        cv2.rectangle(img, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 2)
        cv2.putText(img, 'Face', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155,100,0), 2)
        
    for (x2,y2,w2,h2) in eyes:
        cv2.rectangle(img, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 1)
    
    '''for(x3,y3,w3,h3) in smiles:
        cv2.rectangle(img, (x3,y3), (x3+w3, y3+h3), (0,0,255), 2)'''
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        cv2.destroyAllWindows()
        break
        
cap.release()

