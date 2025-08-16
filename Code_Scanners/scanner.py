#With Auth and Time Logic

import cv2
import numpy as np
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load authorized codes
with open('myDataFile.txt') as f:
    myDataList = f.read().splitlines()

prevVisibleCodes = set()  # Barcodes detected in previous frame

while True:
    success, img = cap.read()
    currentVisibleCodes = set()

    for barcode in decode(img):
        myData = barcode.data.decode('utf-8')
        currentVisibleCodes.add(myData)

        if myData in myDataList:
            myOutput = 'Authorized'
            myColor = (0, 255, 0)
        else:
            myOutput = 'Unauthorized'
            myColor = (0, 0, 255)

        pts = np.array([barcode.polygon], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, myColor, 5)
        pts2 = barcode.rect
        cv2.putText(img, f"{myOutput}: {myData}", (pts2[0], pts2[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, myColor, 2)

        # Check if it's a newly seen code
        if myData not in prevVisibleCodes:
            print(f"Logging once: {myData}")  # Replace this with DB logging logic

    # Update previous frame codes
    prevVisibleCodes = currentVisibleCodes

    cv2.imshow('Gate Scanner', img)
    cv2.waitKey(1)
