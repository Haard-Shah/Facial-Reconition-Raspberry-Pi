
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
from RPLCD import CharLCD


#Google sheets initialziation
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(creds)
sheet = client.open('Attandence').sheet1

#LCD Initialzation
lcd = CharLCD(cols=16, rows=2, pin_rs=37, pin_e=35, pins_data=[33, 31, 29, 23])


#Face Recognition initialzation 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml') #load faces
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names (string value) related to ids: 
names = ['None', 'Haard', 'Kaushal']
StudentsAttend = {"None":0, "Haard":0, "Kaushal":0}

# Initialize and start realtime video capture
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# Define min window size to be recognized as a face
minW = 0.1*640
minH = 0.1*480

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Look for faces 
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))

            if StudentsAttend[id]==0: #Checking their presence
                StudentsAttend[id]=1 #Marking them present
                sheet.append_row([id, str(datetime.date.today()), "Present"]) #uplaoding data to the google sheets
                lcd.clear()
                lcd.write_string(u'  Good Morning \n\r  '+id+' :)')
            
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            #sheet.append_row(["Unknown", str(datetime.date.today()), "Present"])

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('camera',img)

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
camera.stop_preview()
cv2.destroyAllWindows()

#installed opencv-contrib-python library for cv2.face to work
#Also completly reinstally ananconda version 5.1.0 for the now compadible with virutal env support and anaconda
# navigator gui package manager.
