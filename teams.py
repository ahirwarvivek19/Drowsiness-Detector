import os
import time
import cv2
import dlib
from playsound import playsound
from scipy.spatial import distance


def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio


cap = cv2.VideoCapture(0)

# Capture or input video frame-by-frame
# cap=cv2.VideoCapture("Video.mp4")


hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")
count = 0
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 2)
        if EAR < 0.18:
            cv2.putText(frame, "DROWSY", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
            cv2.putText(frame, "Are you Sleepy?", (20, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            print("Drowsy")

        if EAR < 0.18:
            time.sleep(.1)

        if EAR < 0.18:
            playsound('alarm_clock_tonight2.mp3')
            count=count+1

        if count>10:
            playsound('Industrial Alarm-SoundBible.com-1012301296.mp3')
            os.system('taskkill/im Teams.exe')
           # os.system('taskkill/im Zoom.exe')
          #  os.system('taskkill/im chrome.exe')
            count = 0


        print(EAR)

    cv2.imshow("Are you Sleepy", frame)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break


cap.release()
cv2.destroyAllWindows()