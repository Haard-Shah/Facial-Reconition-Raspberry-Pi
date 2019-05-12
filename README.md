# Facial-Reconition-Raspberry-Pi
Automatically update attendence sheet for students who are walking into the class room using a trained facial recogniction model.  

## Libraries
   1. cv2 from OpenCV
   2. Numpy
   3. PiCamera
   4. PiRGBArray
   5. PIL from Image 
   6. gspread
   7. ServiceAccountCredentials
   
## Results
The final results of the project can be viewed at https://www.youtube.com/watch?v=JDegd-iHBjI. The raspberry pi with trained models recognises student, greet them with a custom welcome message and updates their presence in the class attendence marksheet

## How to use
To run my project on your local machine please first install the above mentioned libraries, updated the Raspbian OS on your raspberry pi and then upload the scripts to your pi. Next run '1_face_dataset.py' to collect traning data using raspberry pi. Please take training data in conditions similar to expect final working environment. Next, run '02_face_training.py' to train a facial recognition model (I am using Haar Cascades classifier to detect faces in an image). Finally run, '03_Real_Time_Face_Recognition.py' to run real time facial recognition. Enjoy! :)
   
