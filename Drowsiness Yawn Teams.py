# Drowsiness Detector Final with Yawn Detector and Online meeting app closer

import os
import time
import cv2
import dlib
from playsound import playsound
from imutils import face_utils
from scipy.spatial import distance
from threading import Thread
import numpy as np
# For Exceptions
import sys


try:

    def alarm(msg):
        global alarm_status
        global alarm_status2
        global saying

        while alarm_status:
            print('call')
            s = 'espeak "'+msg+'"'
            os.system(s)

        if alarm_status2:
            print('call')
            saying = True
            s = 'espeak "' + msg + '"'
            os.system(s)
            saying = False


    def calculate_EAR(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear_aspect_ratio = (A + B) / (2.0 * C)
        return ear_aspect_ratio


    def lip_distance(shape):
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))

        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))

        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)

        yawn = abs(top_mean[1] - low_mean[1])
        return yawn


    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 60
    YAWN_THRESH = 30
    alarm_status = False
    alarm_status2 = False
    saying = False
    COUNTER = 0


    cap = cv2.VideoCapture(0)

    # Capture or input video frame-by-frame
    # cap=cv2.VideoCapture("Video.mp4")

    #  Alternate of face detector
    # detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate

    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat")


    frameCounter = 0
    # Threshold for closing teams if faces not found
    frameCounterThreshold = 60

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = hog_face_detector(gray)

        if faces:
            frameCounter = 0

            for face in faces:

                face_landmarks = dlib_facelandmark(gray, face)
                leftEye = []
                rightEye = []

                # Drawing Eye Shape
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

                # EAR
                left_ear = calculate_EAR(leftEye)
                right_ear = calculate_EAR(rightEye)
                EAR = (left_ear + right_ear) / 2
                EAR = round(EAR, 2)

                # Yawn
                shape = face_utils.shape_to_np(face_landmarks)
                yawn = lip_distance(shape)
                yawn = round(yawn,2)
                # Drawing lip shape
                lip = shape[48:60]
                cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

                # Conditions to check drowsy
                if EAR < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if alarm_status == False:
                            alarm_status = True
                            t = Thread(target=alarm, args=('wake up sir',))
                            t.deamon = True
                            t.start()
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print("Drowsy")

                        # playsound('Industrial Alarm-SoundBible.com-1012301296.mp3')
                        os.system('taskkill/im Teams.exe')
                        # os.system('taskkill/im Zoom.exe')
                        #  os.system('taskkill/im chrome.exe')

                else:
                    COUNTER = 0
                    alarm_status = False

                # Yawn Condition
                if (yawn > YAWN_THRESH):
                    cv2.putText(frame, "Yawn Alert", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if alarm_status2 == False and saying == False:
                        alarm_status2 = True
                        t = Thread(target=alarm, args=('Take some fresh air sir',))
                        t.deamon = True
                        t.start()
                else:
                    alarm_status2 = False

                print("EAR = ", EAR, " Yawn = ", yawn)
                print()

                # # # Old Code
                # if EAR < 0.18:
                #     time.sleep(.1)

                # if EAR < 0.18:
                #     playsound('alarm_clock_tonight2.mp3')
                #     count=count+1
                #
                # if count>10:
                #     playsound('Industrial Alarm-SoundBible.com-1012301296.mp3')
                #     os.system('taskkill/im Teams.exe')
                #    # os.system('taskkill/im Zoom.exe')
                #   #  os.system('taskkill/im chrome.exe')
                #     count = 0

        # Face not found condition
        else:
            frameCounter+=1
            print("Face not found")
            cv2.putText(frame, "Face not found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if frameCounter>frameCounterThreshold:
                os.system('taskkill/im Teams.exe')
                #    # os.system('taskkill/im Zoom.exe')
                #   #  os.system('taskkill/im chrome.exe')



        cv2.imshow("Are you Sleepy", frame)

        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    cap.release()
    cv2.destroyAllWindows()


except Exception as e:
    print("Oops!", e.__class__, "occurred.")
    print()
