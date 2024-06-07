#importing libraries
import cv2
import os
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from joblib import load, dump

# VARIABLES
datetoday = date.today().strftime("%m_%d_%y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
  os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
  os.makedirs('static/faces')

# Function to get a number of total registered users
def totalreg():
  return len(os.listdir('static/faces'))

# Function to extract the face from an image
def extract_faces(img):
  if img != []:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points
  else:
    return [ ]

# Function to identify face using ML model
def identify_face(facearray):
  model = load('static/face_recognition_model.pkl')
  return model.predict(facearray)
# Function to train the model on all the faces available in faces folder
def train_model():
  faces = []
  labels = []
  userlist = os.listdir('static/faces')
  for user in userlist:
    for imgname in os.listdir(f'static/faces/{user}'):
      img = cv2.imread(f'static/faces/{user}/{imgname}')
      resized_face = cv2.resize(img, (50, 50))
      faces.append(resized_face.ravel())
      labels.append(user)
  faces = np.array(faces)
  knn = KNeighborsClassifier(n_neighbors=5)
  knn.fit(faces, labels)
  dump(knn, 'static/face_recognition_model.pkl')

# Main loop for face recognition and attendance marking
while True:
  # Capture video from webcam
  cap = cv2.VideoCapture(0)
  ret = True
  while ret:
    ret, frame = cap.read()

 # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 # Detect faces in the grayscale frame
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

 # Process each detected face
    for (x, y, w, h) in faces:
      # Extract face region
      face = frame[y:y+h, x:x+w]
 # Identify the face
      identified_person = identify_face(face.reshape(1, -1))[0]
 # Draw rectangle and display name
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.putText(frame, f'{identified_person}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

      # Add functionality to mark attendance (e.g., press 'a' key)
      if cv2.waitKey(1) == ord('a'):
        add_attendance(identified_person) # Assuming you have an add_attendance function

    # Display the resulting frame
    cv2.imshow('Attendance Check, press "q" to exit', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
      break

  # Release resources
  cap.release()
  cv2.destroyAllWindows()
