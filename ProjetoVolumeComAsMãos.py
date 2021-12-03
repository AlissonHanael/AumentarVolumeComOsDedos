import cv2
import mediapipe as mp
import math as mt
import comtypes as cot
import numpy as np
from pycaw.pycaw import IAudioEndpointVolume, AudioUtilities
from ctypes import cast, POINTER

cap = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

dispositivos = AudioUtilities.GetSpeakers()
interface = dispositivos.Activate(
    IAudioEndpointVolume._iid_, cot.CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volMin, volMax = volume.GetVolumeRange()[:2]

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultados = hands.process(imgRGB)

    lmList = []

    if resultados.multi_hand_landmarks:
        for handlandmark in resultados.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handlandmark, mphands.HAND_CONNECTIONS)
    if lmList != []:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = mt.hypot(x2-x1, y2-y1)

        vol = np.interp(length, [15, 220], [volMin, volMax])
        print(vol, length)
        volume.SetMasterVolumeLevel(vol, None)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
