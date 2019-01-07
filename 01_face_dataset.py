from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import os
import numpy

cam = PiCamera()
cam.resolution = (640, 480)
cam.framerate = 32
rawCapture = PiRGBArray(cam, size=(640, 480))

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    #ret, img = cam.read()
    img = frame.array
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
        print(30-count)
        
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        print('User interrupted.')
        break
    elif count >= 30: # Take 30 face sample and stop video
         print('Data collected!')
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.stop_preview()
cv2.destroyAllWindows()
